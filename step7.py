"""
step7.py -- Run the draft and the target AT THE SAME TIME on one GPU and measure how they interfere.

Overlapped speculative decoding (PEARL-style, or SpecIso) runs the draft for the next round while
the target verifies the current one. This script measures the GPU-level cost of that overlap,
separately from the scheduling question of how to break the round dependency.

How
  * Both models are captured as CUDA graphs with torch.cuda.graph(), each with its own memory
    pool, so they can safely run at the same time. (torch.compile's graph trees share one pool and
    assume graphs run one after another, so they are not safe for this.)
      - draft graph(k)  : reset cache -> k draft passes, each feeding the next (one graph)
      - target graph(k) : reset cache -> one verify pass over k+1 tokens
  * Each graph runs on its own CUDA stream. A short GPU spin (torch.cuda._sleep) holds the GPU
    while both are queued, so they really start together.
  * Timing uses CUDA events from that common start, median over many trials.

Prints
  Part A: for each k, draft alone / target alone vs both at once: each one's slowdown, the
          makespan (when both finish), and how much of the serial time the overlap saves.
  Part B: the same at k = 4 with CUDA stream priorities (target first / draft first).
  Part C: SM clock and power when running alone vs together (power/clock throttling check).

Nsight Systems timeline (optional, short run):
  nsys profile -t cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi \
       -o step7_k4 python step7.py --nsys
"""
import sys, time, statistics, threading, torch
import transformers.masking_utils as mu
# For multi-token passes (verify), build the attention mask explicitly: HF's shortcut that skips it
# reads a GPU value on the CPU (a sync), which is not allowed inside CUDA-graph capture. 1-token
# passes (draft) keep HF's default path, which has no sync. Results are identical either way.
_orig_skip = mu._ignore_causal_mask_sdpa
def _skip_mask(padding_mask, q_length, *a, **k):
    return False if q_length > 1 else _orig_skip(padding_mask, q_length, *a, **k)
mu._ignore_causal_mask_sdpa = _skip_mask
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

dev = "cuda"
NSYS = "--nsys" in sys.argv
KS = (1, 2, 3, 4, 6)
TRIALS = 50

T = "Qwen/Qwen2.5-7B-Instruct"     # target
D = "Qwen/Qwen2.5-0.5B-Instruct"   # draft

tok = AutoTokenizer.from_pretrained(T)
target = AutoModelForCausalLM.from_pretrained(T, dtype=torch.float16, device_map="cuda").eval()
draft  = AutoModelForCausalLM.from_pretrained(D, dtype=torch.float16).to(dev).eval()

msgs = [{"role": "user", "content": "Write a Python function to check if a string is a palindrome, then explain it."}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
prompt = tok(text, return_tensors="pt").to(dev)["input_ids"]
L = prompt.shape[1]
MAX_LEN = L + 64


# ---------------------------------------------------------------------------
# Caches, static buffers, and the work each graph does
# ---------------------------------------------------------------------------
@torch.no_grad()
def prefill(model):
    cache = StaticCache(config=model.config, max_cache_len=MAX_LEN)
    out = model(input_ids=prompt, past_key_values=cache, use_cache=True)
    return cache, out.logits[:, -1:].argmax(-1)

def set_len(cache, n):
    for layer in cache.layers:
        layer.cumulative_length.fill_(n)

t_cache, first = prefill(target)
d_cache, _ = prefill(draft)

d_start = first.clone()                                          # draft's first input token
d_tok = torch.zeros(1, 1, dtype=torch.long, device=dev)          # draft's current input
d_out = torch.zeros(1, max(KS), dtype=torch.long, device=dev)    # draft tokens produced
v_in = first.repeat(1, max(KS) + 1)                              # target verify input
t_out = torch.zeros(1, max(KS) + 1, dtype=torch.long, device=dev)

def draft_work(k):
    @torch.no_grad()
    def fn():
        set_len(d_cache, L)                  # every replay starts from the same state
        d_tok.copy_(d_start)
        for i in range(k):                   # k sequential passes: each needs the previous token
            nxt = draft(input_ids=d_tok, past_key_values=d_cache, use_cache=True).logits[:, -1:].argmax(-1)
            d_tok.copy_(nxt)
            d_out[:, i:i + 1].copy_(nxt)
    return fn

def verify_work(k):
    @torch.no_grad()
    def fn():
        set_len(t_cache, L)
        p = target(input_ids=v_in[:, :k + 1], past_key_values=t_cache, use_cache=True).logits.argmax(-1)
        t_out[:, :k + 1].copy_(p)
    return fn

def capture(fn, s):
    # Standard recipe: warm up on a side stream, then record. Each graph gets its own memory pool.
    # Each MODEL is captured on its own stream `s`: cuBLAS keeps one scratch workspace per stream and
    # bakes its address into the graph, so graphs captured on the same stream share that workspace
    # and corrupt each other when they run at the same time.
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn()
    torch.cuda.synchronize()
    return g

print("capturing CUDA graphs ...")
cap_d, cap_t = torch.cuda.Stream(), torch.cuda.Stream()
g_draft = {k: capture(draft_work(k), cap_d) for k in KS}
g_ver = {k: capture(verify_work(k), cap_t) for k in KS}

lo, hi = torch.cuda.Stream.priority_range()                     # (lowest, highest) priority values
s_d, s_t = torch.cuda.Stream(), torch.cuda.Stream()             # normal priority
s_d_hi, s_t_hi = torch.cuda.Stream(priority=hi), torch.cuda.Stream(priority=hi)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def launch(jobs):
    """jobs: list of (stream, graph). Returns one end event per job and the common start event."""
    torch.cuda._sleep(2_000_000)             # keep the GPU busy ~1 ms while we queue everything
    start = torch.cuda.Event(enable_timing=True)
    start.record()
    ends = []
    for st, g in jobs:
        st.wait_event(start)
        with torch.cuda.stream(st):
            g.replay()
            e = torch.cuda.Event(enable_timing=True)
            e.record(st)
        ends.append(e)
    return start, ends

def measure(jobs, trials=TRIALS):
    """Median ms from the common start to each job's end."""
    for _ in range(5):
        launch(jobs)
    torch.cuda.synchronize()
    per = [[] for _ in jobs]
    for _ in range(trials):
        start, ends = launch(jobs)
        torch.cuda.synchronize()
        for i, e in enumerate(ends):
            per[i].append(start.elapsed_time(e))
    return [statistics.median(p) for p in per]


# ---------------------------------------------------------------------------
# Optional: short run for an Nsight Systems timeline
# ---------------------------------------------------------------------------
if NSYS:
    k = 4
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    for name, jobs in (("alone: draft", [(s_d, g_draft[k])]),
                       ("alone: target verify", [(s_t, g_ver[k])]),
                       ("together", [(s_t, g_ver[k]), (s_d, g_draft[k])])):
        torch.cuda.nvtx.range_push(name)
        for _ in range(5):
            launch(jobs)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    torch.cuda.profiler.stop()
    print("nsys capture done")
    sys.exit()


# ---------------------------------------------------------------------------
# Correctness: running together must not change either model's output
# ---------------------------------------------------------------------------
k = 4
measure([(s_d, g_draft[k])], 1); ref_d = d_out[:, :k].clone()
measure([(s_t, g_ver[k])], 1);   ref_t = t_out[:, :k + 1].clone()
measure([(s_t, g_ver[k]), (s_d, g_draft[k])], 3)
same_d = torch.equal(ref_d, d_out[:, :k]); same_t = torch.equal(ref_t, t_out[:, :k + 1])
print(f"outputs identical when running together: draft {same_d}, target {same_t}")
if not (same_d and same_t):
    print("  WARNING: running together changed the results -> the graphs share memory somewhere;"
          " timings below are suspect")


# ---------------------------------------------------------------------------
# Part A: alone vs together, for each k
# ---------------------------------------------------------------------------
def compare(k, sd, st, label=""):
    d_alone, = measure([(sd, g_draft[k])])
    t_alone, = measure([(st, g_ver[k])])
    t_co, d_co = measure([(st, g_ver[k]), (sd, g_draft[k])])
    make = max(d_co, t_co)
    serial = d_alone + t_alone
    best = max(d_alone, t_alone)                     # perfect overlap, no interference
    eff = (serial - make) / (serial - best) if serial > best else 0.0
    return dict(k=k, d_alone=d_alone, t_alone=t_alone, d_co=d_co, t_co=t_co,
                make=make, serial=serial, best=best, eff=eff, label=label)

def row(r):
    print(f"{r['label']:>14} k={r['k']} | draft {r['d_alone']:6.2f} -> {r['d_co']:6.2f} ms "
          f"({r['d_co']/r['d_alone']:4.2f}x) | target {r['t_alone']:6.2f} -> {r['t_co']:6.2f} ms "
          f"({r['t_co']/r['t_alone']:4.2f}x) | serial {r['serial']:6.2f}, together {r['make']:6.2f}, "
          f"ideal {r['best']:6.2f} ms | overlap gets {100*r['eff']:5.1f}% of the ideal saving")

print("\n=== Part A: draft (k passes) and target verify (k+1 tokens), alone vs together ===")
print("x = slowdown when running together;  ideal = max(alone times), i.e. perfect overlap")
for k in KS:
    row(compare(k, s_d, s_t, "same priority"))

# ---------------------------------------------------------------------------
# Part B: does stream priority change who pays?
# ---------------------------------------------------------------------------
print("\n=== Part B: k = 4 with CUDA stream priorities ===")
row(compare(4, s_d, s_t, "same priority"))
row(compare(4, s_d, s_t_hi, "target first"))
row(compare(4, s_d_hi, s_t, "draft first"))


# ---------------------------------------------------------------------------
# Part C: clocks and power, alone vs together (k = 4)
# ---------------------------------------------------------------------------
print("\n=== Part C: SM clock and power, k = 4 (sustained ~2 s each) ===")
try:
    import pynvml
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

    def sustained(jobs, seconds=2.0):
        samples, stop = [], threading.Event()
        def sampler():
            while not stop.is_set():
                samples.append((pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
                                pynvml.nvmlDeviceGetPowerUsage(h) / 1000))
                time.sleep(0.01)
        th = threading.Thread(target=sampler); th.start()
        t0 = time.time()
        while time.time() - t0 < seconds:       # back-to-back replays, no idle gaps
            for _ in range(10):
                for st, g in jobs:
                    with torch.cuda.stream(st):
                        g.replay()
            torch.cuda.synchronize()
        stop.set(); th.join()
        clk = statistics.median(s[0] for s in samples)
        pw = statistics.median(s[1] for s in samples)
        return clk, pw

    limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000
    print(f"power limit: {limit:.0f} W")
    for name, jobs in (("draft alone", [(s_d, g_draft[4])]),
                       ("target alone", [(s_t, g_ver[4])]),
                       ("together", [(s_t, g_ver[4]), (s_d, g_draft[4])])):
        clk, pw = sustained(jobs)
        print(f"{name:>13}: SM clock {clk:5.0f} MHz, power {pw:5.1f} W")
except ImportError:
    print("skipped: pip install nvidia-ml-py   (then rerun)")
except Exception as ex:
    print(f"skipped: NVML error {type(ex).__name__}: {ex}")
