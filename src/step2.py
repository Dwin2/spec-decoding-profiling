"""
step2.py (v2) -- Where does the time go in speculative decoding?

Part A: time ONE forward pass of each model (draft 1 token, target 1 and 5 tokens)
Part B: profile one speculative generation  -> rounds, time split, and (fixed) GPU busy/idle,
        including GPU busy % *during draft passes* vs *during target passes*
Part C: fix the number of draft guesses per round (k) and see how the speedup changes
"""
import time, statistics, torch
from torch.profiler import profile, ProfilerActivity, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer

dev = "cuda"
def sync():
    torch.cuda.synchronize()

T = "Qwen/Qwen2.5-7B-Instruct"     # target
D = "Qwen/Qwen2.5-0.5B-Instruct"   # draft

tok = AutoTokenizer.from_pretrained(T)
target = AutoModelForCausalLM.from_pretrained(T, dtype=torch.float16, device_map="cuda").eval()
draft  = AutoModelForCausalLM.from_pretrained(D, dtype=torch.float16).to(dev).eval()
draft.resize_token_embeddings(target.config.vocab_size)   # 7B pads its vocab; same tokenizer

msgs = [{"role": "user", "content": "Write a Python function to check if a string is a palindrome, then explain it."}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt").to(dev)
prompt_len = inputs["input_ids"].shape[1]

# ============================================================
# Part A: time ONE forward pass
# ============================================================
@torch.no_grad()
def time_pass(model, n_tokens, iters=30):
    out = model(**inputs, use_cache=True)                    # prefill the prompt once
    past = out.past_key_values
    x = out.logits[:, -1:].argmax(-1).repeat(1, n_tokens)    # n_tokens new tokens
    times = []
    for i in range(iters + 5):                               # first 5 = warmup
        sync(); t0 = time.perf_counter()
        model(input_ids=x, past_key_values=past, use_cache=True)
        sync(); dt = (time.perf_counter() - t0) * 1000
        past.crop(-n_tokens)                                 # undo, so each pass starts the same
        if i >= 5:
            times.append(dt)
    return statistics.median(times)

print("\n=== Part A: time per forward pass (median, ms) ===")
d1 = time_pass(draft, 1)
t1 = time_pass(target, 1)
t5 = time_pass(target, 5)
print(f"draft  pass, 1 token : {d1:6.2f} ms")
print(f"target pass, 1 token : {t1:6.2f} ms")
print(f"target pass, 5 tokens: {t5:6.2f} ms")
print(f"draft/target cost ratio      : {d1/t1:.2f}")
print(f"5-token / 1-token target pass: {t5/t1:.2f}x")

# ============================================================
# Part B: profile one speculative generation
# ============================================================
LABELS = ("WHOLE_GENERATE", "TARGET", "DRAFT")

def label(model, name):
    orig = model.forward
    def wrapped(*a, **k):
        with record_function(name):
            return orig(*a, **k)
    model.forward = wrapped
    return orig

target_forward = label(target, "TARGET")
draft_forward  = label(draft, "DRAFT")

orig_k = draft.generation_config.num_assistant_tokens
orig_sched = draft.generation_config.num_assistant_tokens_schedule

def reset_draft_schedule():
    # HF writes the adapted k back into the draft's config after every generate(); undo that
    draft.generation_config.num_assistant_tokens = orig_k
    draft.generation_config.num_assistant_tokens_schedule = orig_sched

reset_draft_schedule()
target.generate(**inputs, max_new_tokens=8, do_sample=False, assistant_model=draft)   # warmup
sync()
reset_draft_schedule()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with record_function("WHOLE_GENERATE"):
        out = target.generate(**inputs, max_new_tokens=50, do_sample=False, assistant_model=draft)
    sync()

prof.export_chrome_trace("trace.json")
print("\nsaved trace.json  (open it at https://ui.perfetto.dev)")

def ranges(name):
    return [(e.time_range.start, e.time_range.end) for e in prof.events() if e.name == name]

def merge(iv):
    iv = sorted(iv); out_iv = []
    for s, e in iv:
        if out_iv and s <= out_iv[-1][1]:
            out_iv[-1][1] = max(out_iv[-1][1], e)
        else:
            out_iv.append([s, e])
    return out_iv

def length(iv):
    return sum(e - s for s, e in merge(iv))

def overlap(a, b):
    a, b = merge(a), merge(b); i = j = 0; tot = 0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0]); e = min(a[i][1], b[j][1])
        if e > s: tot += e - s
        if a[i][1] < b[j][1]: i += 1
        else: j += 1
    return tot

wall = length(ranges("WHOLE_GENERATE"))
tgt, drf = ranges("TARGET"), ranges("DRAFT")
new_tokens = out.shape[1] - prompt_len
rounds = len(tgt) - 1

# FIX vs v1: only real GPU work (kernels / copies), not the GPU-side copies of our labels
gpu = [(e.time_range.start, e.time_range.end) for e in prof.events()
       if e.device_type == torch.autograd.DeviceType.CUDA
       and e.time_range.elapsed_us() > 0
       and e.name not in LABELS]
busy = length(gpu)

print(f"\n=== Part B: one generation of {new_tokens} tokens (profiler adds overhead; trust % not ms) ===")
print(f"total time            : {wall/1000:8.1f} ms")
print(f"rounds                : {rounds}   (TARGET calls {len(tgt)} incl. 1 prefill)")
print(f"draft passes per round: {len(drf)/max(1,rounds):.1f}")
print(f"tokens per round      : {new_tokens/max(1,rounds):.2f}")
print(f"time inside TARGET    : {100*length(tgt)/wall:5.1f}%")
print(f"time inside DRAFT     : {100*length(drf)/wall:5.1f}%")
print(f"time outside both     : {100*(wall-length(tgt+drf))/wall:5.1f}%")
print(f"GPU busy overall      : {100*busy/wall:5.1f}%   (GPU idle {100*(wall-busy)/wall:5.1f}%)")
print(f"GPU busy during TARGET: {100*overlap(gpu, tgt)/max(1,length(tgt)):5.1f}%   (approx.)")
print(f"GPU busy during DRAFT : {100*overlap(gpu, drf)/max(1,length(drf)):5.1f}%   (approx.)")
print(f"GPU kernels/copies    : {len(gpu)}  (~{len(gpu)/max(1,len(tgt)+len(drf)):.0f} per forward pass)")

# ============================================================
# Part C: fixed number of draft guesses per round (k)
# ============================================================
target.forward = target_forward      # remove the labels (no profiler here)
draft.forward = draft_forward

def tok_per_s(**kw):
    target.generate(**inputs, max_new_tokens=8, do_sample=False, **kw)    # warmup
    sync(); t0 = time.perf_counter()
    o = target.generate(**inputs, max_new_tokens=200, do_sample=False, **kw)
    sync()
    return (o.shape[1] - prompt_len) / (time.perf_counter() - t0)

print("\n=== Part C: speedup vs number of draft guesses per round ===")
base = tok_per_s()
print(f"target alone          : {base:5.1f} tok/s")
reset_draft_schedule()
spec = tok_per_s(assistant_model=draft)
print(f"HF default (adaptive) : {spec:5.1f} tok/s  -> {spec/base:.2f}x")
for k in (1, 2, 3, 4, 6):
    draft.generation_config.num_assistant_tokens = k
    draft.generation_config.num_assistant_tokens_schedule = "constant"
    s = tok_per_s(assistant_model=draft)
    print(f"fixed k = {k}           : {s:5.1f} tok/s  -> {s/base:.2f}x")
reset_draft_schedule()
