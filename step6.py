"""
step6.py -- Our own speculative-decoding loop with BOTH models on CUDA graphs.

step5 estimated ~2x if the draft's framework overhead were removed. This turns that
estimate into a measurement, and gives the real serial baseline SpecIso would improve on.

What it does
  * Both models use a StaticCache (fixed-size KV buffers) and torch.compile(mode="reduce-overhead"),
    so each forward pass is a replayed CUDA graph.
  * One round = k draft passes (1 token each) -> 1 target verify pass (k+1 tokens)
    -> accept the longest matching prefix + 1 target token -> roll both caches back.
  * Rolling back a StaticCache is just resetting its length counter: rejected entries are
    masked out by position and overwritten next round (like EAGLE's preallocated KV).
  * Greedy only. One CPU sync per round (reading how many tokens were accepted) -- that read is
    the draft<->target dependency itself.

Prints
  Part A: tok/s for the target alone (graphed) and for SD at k = 1..6, plus acceptance stats.
  Part B: per-round time split (draft / verify / accept+rollback) at the best k, and an
          upper bound for what perfect draft/verify overlap (the SpecIso goal) could give.

The first run of each k compiles and records CUDA graphs: expect several minutes in total.
"""
import os, time, statistics, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

dev = "cuda"
COMPILE = os.environ.get("NO_COMPILE") is None        # NO_COMPILE=1 -> plain eager (for testing)
def sync():
    torch.cuda.synchronize()

for name in ("recompile_limit", "cache_size_limit"):     # one compiled graph per input shape
    if hasattr(torch._dynamo.config, name):
        setattr(torch._dynamo.config, name, 64)

T = "Qwen/Qwen2.5-7B-Instruct"     # target
D = "Qwen/Qwen2.5-0.5B-Instruct"   # draft (no vocab resize needed: we only compare token ids)
MAX_NEW = 200
KS = (1, 2, 3, 4, 5, 6)

tok = AutoTokenizer.from_pretrained(T)
target = AutoModelForCausalLM.from_pretrained(T, dtype=torch.float16, device_map="cuda").eval()
draft  = AutoModelForCausalLM.from_pretrained(D, dtype=torch.float16).to(dev).eval()

msgs = [{"role": "user", "content": "Write a Python function to check if a string is a palindrome, then explain it."}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
prompt = tok(text, return_tensors="pt").to(dev)["input_ids"]
L = prompt.shape[1]
MAX_LEN = L + MAX_NEW + max(KS) + 8


# ---------------------------------------------------------------------------
# Static caches + compiled passes
# ---------------------------------------------------------------------------
@torch.no_grad()
def prefill(model):
    cache = StaticCache(config=model.config, max_cache_len=MAX_LEN)
    out = model(input_ids=prompt, past_key_values=cache, use_cache=True)   # eager prefill
    return cache, out.logits[:, -1:].argmax(-1)

def set_len(cache, n):
    # Number of tokens the cache holds. Everything at positions >= n is ignored and overwritten.
    for layer in cache.layers:
        layer.cumulative_length.fill_(n)

def make_pass(model, cache):
    def fwd(ids):                                   # ids: [1, n] -> greedy next token at every position
        return model(input_ids=ids, past_key_values=cache, use_cache=True).logits.argmax(-1)
    f = torch.compile(fwd, mode="reduce-overhead", dynamic=False) if COMPILE else fwd
    def call(ids):
        if COMPILE:
            torch.compiler.cudagraph_mark_step_begin()
        return f(ids)
    return torch.no_grad()(call)

t_cache, first_tok = prefill(target)
d_cache, _ = prefill(draft)                         # draft's own prompt KV; its guess is unused
t_pass = make_pass(target, t_cache)
d_pass = make_pass(draft, d_cache)

# Persistent input buffers: CUDA graphs want inputs at fixed addresses
d_in = torch.zeros(1, 1, dtype=torch.long, device=dev)
t_in1 = torch.zeros(1, 1, dtype=torch.long, device=dev)
v_in = {k: torch.zeros(1, k + 1, dtype=torch.long, device=dev) for k in KS}


# ---------------------------------------------------------------------------
# Target alone (graphed): the fair baseline
# ---------------------------------------------------------------------------
@torch.no_grad()
def target_alone(n_new):
    set_len(t_cache, L)
    t_in1.copy_(first_tok)
    out = torch.empty(n_new, dtype=torch.long, device=dev)
    out[0] = first_tok[0, 0]
    for i in range(1, n_new):
        nxt = t_pass(t_in1)
        t_in1.copy_(nxt)
        out[i] = nxt[0, 0]
    return out.tolist()


# ---------------------------------------------------------------------------
# Speculative decoding (graphed)
# ---------------------------------------------------------------------------
@torch.no_grad()
def spec_decode(k, n_new, phases=False):
    """Returns (tokens, rounds, total accepted drafts, per-round phase times or None)."""
    set_len(t_cache, L); set_len(d_cache, L)
    N = L + 1                         # committed tokens (prompt + first token); caches hold N-1
    toks = [first_tok[0, 0].item()]
    v = v_in[k]
    v[0, 0] = first_tok[0, 0]         # v = [cur, d1..dk]
    d_in.copy_(first_tok)
    rounds = accepted = 0
    times = [] if phases else None

    while len(toks) < n_new:
        if phases: sync(); t0 = time.perf_counter()
        # 1) draft k tokens, one pass each (input of pass i = output of pass i-1)
        for i in range(k):
            nxt = d_pass(d_in)
            d_in.copy_(nxt)
            v[0, i + 1] = nxt[0, 0]
        if phases: sync(); t1 = time.perf_counter()

        # 2) target verifies [cur, d1..dk] in ONE pass -> its own greedy token at each position
        pred = t_pass(v)                                          # [1, k+1]
        if phases: sync(); t2 = time.perf_counter()

        # 3) accept the longest matching prefix, then take the target's next token
        match = (v[0, 1:] == pred[0, :k]).long().cumprod(0).sum()
        res = torch.cat([match.view(1), pred[0]]).tolist()        # the one CPU<->GPU sync per round
        m, pred_l = res[0], res[1:]
        toks += pred_l[:m + 1]        # accepted drafts equal the target's own tokens, + 1 target token
        accepted += m; rounds += 1

        # 4) roll back: caches should hold the N + m tokens before the new `cur`
        if m == k:                     # draft never saw d_k: feed it once (output ignored)
            d_in.copy_(v[:, k:k + 1]); d_pass(d_in)
        set_len(t_cache, N + m); set_len(d_cache, N + m)
        N += m + 1
        cur = pred[:, m:m + 1]
        d_in.copy_(cur); v[0, 0] = cur[0, 0]
        if phases:
            sync(); t3 = time.perf_counter()
            times.append(((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3))

    return toks[:n_new], rounds, accepted, times


def tok_per_s(fn):
    fn()                                   # warmup (+ compile / graph capture the first time)
    fn()
    sync(); t0 = time.perf_counter()
    r = fn()
    sync()
    return r, MAX_NEW / (time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------
print("\n=== Part A: throughput, both models on CUDA graphs (200 new tokens, greedy) ===")
print("(compiling: first k can take a few minutes)")
base_toks, base = tok_per_s(lambda: target_alone(MAX_NEW))
print(f"target alone (graphed)       : {base:5.1f} tok/s")

best = None
for k in KS:
    (toks, rounds, acc, _), s = tok_per_s(lambda: spec_decode(k, MAX_NEW))
    diff = next((i for i, (a, b) in enumerate(zip(toks, base_toks)) if a != b), None)
    same_s = "same output" if diff is None else f"differs from token {diff} (fp16 near-tie?)"
    print(f"SD k = {k}: {s:5.1f} tok/s -> {s/base:.2f}x | {MAX_NEW/rounds:4.2f} tok/round | "
          f"{100*acc/(rounds*k):4.1f}% drafts accepted | {same_s}")
    if best is None or s > best[1]:
        best = (k, s)

print("\n(for reference, step2 Part C, eager HF on this GPU: target alone 28.2 tok/s, best SD 0.95x)")

# ---------------------------------------------------------------------------
# Part B: where does a round's time go, at the best k?
# ---------------------------------------------------------------------------
k = best[0]
_, rounds, acc, times = spec_decode(k, MAX_NEW, phases=True)
dr = statistics.median(t[0] for t in times)
vf = statistics.median(t[1] for t in times)
bk = statistics.median(t[2] for t in times)
rnd = dr + vf + bk
tpr = MAX_NEW / rounds
print(f"\n=== Part B: one round at best k = {k} (median over {len(times)} rounds; extra syncs added) ===")
print(f"draft  ({k} passes)     : {dr:6.2f} ms  ({100*dr/rnd:4.1f}%)   = {dr/k:.2f} ms per draft pass")
print(f"verify (1 target pass): {vf:6.2f} ms  ({100*vf/rnd:4.1f}%)")
print(f"accept + rollback     : {bk:6.2f} ms  ({100*bk/rnd:4.1f}%)")
print(f"round total           : {rnd:6.2f} ms  for {tpr:.2f} tokens")
serial = tpr / rnd * 1000
overlap = tpr / (max(dr, vf) + bk) * 1000
print(f"\nserial SD (this loop)            : {serial:5.1f} tok/s  -> {serial/base:.2f}x")
print(f"if draft fully hidden under verify: {overlap:5.1f} tok/s  -> {overlap/base:.2f}x  "
      f"(upper bound, same acceptance; real overlap must break the round dependency and avoid interference)")
