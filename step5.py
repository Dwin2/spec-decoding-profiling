"""
step5.py -- Is the draft slow because of the GPU, or because of framework overhead?

Times one 1-token decode pass three ways, for both models:
  1. eager  + DynamicCache   (what HF generate() normally does; same as step2 Part A)
  2. eager  + StaticCache    (fixed-size cache; isolates the cache's own overhead)
  3. CUDA graphs + StaticCache  (torch.compile mode="reduce-overhead": the whole pass is
     recorded once and replayed, removing almost all Python / kernel-launch overhead)

If (3) is much faster than (1) for the draft, the draft was overhead-bound, not GPU-bound.
Then we plug the new draft time into a simple model to estimate the speedup SD could reach.
"""
import time, statistics, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

dev = "cuda"
def sync():
    torch.cuda.synchronize()

T = "Qwen/Qwen2.5-7B-Instruct"     # target
D = "Qwen/Qwen2.5-0.5B-Instruct"   # draft

tok = AutoTokenizer.from_pretrained(T)
target = AutoModelForCausalLM.from_pretrained(T, dtype=torch.float16, device_map="cuda").eval()
draft  = AutoModelForCausalLM.from_pretrained(D, dtype=torch.float16).to(dev).eval()

msgs = [{"role": "user", "content": "Write a Python function to check if a string is a palindrome, then explain it."}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt").to(dev)
L = inputs["input_ids"].shape[1]


def median_ms(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(iters):
        sync(); t0 = time.perf_counter()
        fn()
        sync(); ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


@torch.no_grad()
def eager_dynamic(model):
    out = model(**inputs, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[:, -1:].argmax(-1)
    def step():
        model(input_ids=nxt, past_key_values=past, use_cache=True)
        past.crop(-1)
    return median_ms(step)


@torch.no_grad()
def static_setup(model):
    cache = StaticCache(config=model.config, max_cache_len=L + 16)
    out = model(**inputs, past_key_values=cache, use_cache=True)       # prefill into the static cache
    nxt = out.logits[:, -1:].argmax(-1)
    return cache, nxt


def rewind(cache):
    # The static cache keeps a per-layer counter of how many tokens it holds. Set it back to the
    # prompt length so every decode step writes the same slot and starts from the same state.
    for layer in cache.layers:
        layer.cumulative_length.fill_(L)


@torch.no_grad()
def eager_static(model):
    cache, nxt = static_setup(model)
    def step():
        rewind(cache)
        model(input_ids=nxt, past_key_values=cache, use_cache=True)
    return median_ms(step)


@torch.no_grad()
def graphed_static(model):
    cache, nxt = static_setup(model)
    def fwd(ids):
        rewind(cache)                       # captured inside the CUDA graph too
        return model(input_ids=ids, past_key_values=cache, use_cache=True).logits
    compiled = torch.compile(fwd, mode="reduce-overhead", fullgraph=False)
    for _ in range(5):                      # compile + record the CUDA graph (slow the first time)
        compiled(nxt)
    sync()
    return median_ms(lambda: compiled(nxt))


results = {}
for name, model in (("draft", draft), ("target", target)):
    print(f"\n--- {name} ---")
    r = {}
    r["eager+dynamic"] = eager_dynamic(model);  print(f"eager + DynamicCache      : {r['eager+dynamic']:6.2f} ms")
    r["eager+static"]  = eager_static(model);   print(f"eager + StaticCache       : {r['eager+static']:6.2f} ms")
    try:
        r["graphs+static"] = graphed_static(model)
        print(f"CUDA graphs + StaticCache : {r['graphs+static']:6.2f} ms   (compile can take a minute)")
    except Exception as ex:
        print(f"CUDA graphs failed: {type(ex).__name__}: {str(ex)[:200]}")
    results[name] = r

# ---- Summary + a simple speedup estimate ------------------------------------------------
d_e, t_e = results["draft"]["eager+dynamic"], results["target"]["eager+dynamic"]
print("\n=== Summary ===")
print(f"draft/target cost, eager      : {d_e/t_e:.2f}")
if "graphs+static" in results["draft"] and "graphs+static" in results["target"]:
    d_g, t_g = results["draft"]["graphs+static"], results["target"]["graphs+static"]
    print(f"draft speedup from CUDA graphs: {d_e/d_g:.1f}x   ({d_e:.2f} -> {d_g:.2f} ms)")
    print(f"target speedup from CUDA graphs: {t_e/t_g:.1f}x   ({t_e:.2f} -> {t_g:.2f} ms)")
    print(f"draft/target cost, CUDA graphs: {d_g/t_g:.2f}")

    # Estimate: each round = k draft passes + 1 target verify (~ target pass), producing
    # E = 1 + a + a^2 + ... + a^k tokens, where a = per-token acceptance rate.
    print("\nEstimated SD speedup if both models ran with CUDA graphs (verify pass ~ 1 target pass):")
    for a in (0.6, 0.7, 0.8):
        best = max(((sum(a**i for i in range(k + 1)) * t_g) / (k * d_g + t_g), k) for k in range(1, 9))
        print(f"  acceptance {a:.1f}: best k = {best[1]}, speedup ~ {best[0]:.2f}x")
