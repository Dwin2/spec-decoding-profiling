"""
step4.py -- How much do the "handoffs" between draft and target cost?

Measures the small operations that happen every speculative-decoding round, and compares
each one to the time of one target decode step:
  (a) a GPU->CPU sync (.item())                  -- HF does several per round
  (b) KV cache cleanup: HF-style crop vs EAGLE-style copy of the accepted tokens
  (c) hidden-state handoff (EAGLE feeds target features to the draft)
  (d) draft-logits handoff (HF passes the draft's logits to the target)
  (e) EAGLE-style tree-mask building on the CPU (Python loop)
"""
import time, statistics, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)

T = "Qwen/Qwen2.5-7B-Instruct"   # target
tok = AutoTokenizer.from_pretrained(T)
target = AutoModelForCausalLM.from_pretrained(T, dtype=torch.float16, device_map="cuda").eval()
cfg = target.config

L = cfg.num_hidden_layers
H_kv = cfg.num_key_value_heads
Dh = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
hidden = cfg.hidden_size
vocab = cfg.vocab_size
print(f"target config: {L} layers, {H_kv} KV heads, head dim {Dh}, hidden {hidden}, vocab {vocab}")

msgs = [{"role": "user", "content": "Write a Python function to check if a string is a palindrome, then explain it."}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt").to(dev)


def sync():
    if dev == "cuda":
        torch.cuda.synchronize()

def time_us(fn, iters=200, warmup=20):
    """Median time of fn() in microseconds (waits for the GPU each time)."""
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(iters):
        sync(); t0 = time.perf_counter()
        fn()
        sync(); ts.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(ts)


# ---- Reference: one target decode step --------------------------------------------
with torch.no_grad():
    out = target(**inputs, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[:, -1:].argmax(-1)

    def target_step():
        target(input_ids=nxt, past_key_values=past, use_cache=True)
        past.crop(-1)
    step_us = time_us(target_step, iters=30, warmup=5)
print(f"\nreference: one target decode step = {step_us/1000:.2f} ms\n")

results = []
def add(name, us, note=""):
    results.append((name, us, note))


# ---- (a) GPU -> CPU sync -------------------------------------------------------------
x = torch.randn(1, device=dev)
# launch 1000 tiny kernels back to back, wait once at the end -> cost of just launching one kernel
for _ in range(100): x.sum()
sync(); t0 = time.perf_counter()
for _ in range(1000): x.sum()
sync(); launch_only = (time.perf_counter() - t0) * 1e6 / 1000
# same tiny kernel, but pull the result to the CPU every time -> launch + full GPU<->CPU round trip
with_item = time_us(lambda: x.sum().item(), iters=1000)
add("(a) launch one tiny kernel (no waiting)", launch_only, "x ~550 kernels per decode step")
add("(a) tiny kernel + .item() sync", with_item, "HF does several of these per round")


# ---- (b) KV cache cleanup --------------------------------------------------------------
# HF style: crop the dynamic cache after rejected guesses
with torch.no_grad():
    five = nxt.repeat(1, 5)
    def add_and_crop():
        target(input_ids=five, past_key_values=past, use_cache=True)   # adds 5 tokens
    # time only the crop: add 5 tokens, then time crop(-5)
    crop_times = []
    for i in range(60):
        add_and_crop(); sync()
        t0 = time.perf_counter(); past.crop(-5); sync()
        if i >= 10:
            crop_times.append((time.perf_counter() - t0) * 1e6)
add("(b) HF: crop 5 rejected tokens from KV cache", statistics.median(crop_times))

# EAGLE style: copy the accepted path's K/V entries into place in a preallocated buffer
max_len = 2048
kv = torch.randn(2 * L, 1, H_kv, max_len, Dh, dtype=torch.float16, device=dev)
src = torch.tensor([300, 302, 305, 309, 314], device=dev)            # scattered tree positions
def eagle_copy():
    kv[..., 200:205, :].copy_(kv[..., src, :])
add("(b) EAGLE: copy 5 accepted tokens' K/V", time_us(eagle_copy),
    f"{2*L*H_kv*5*Dh*2/1024:.0f} KB moved")


# ---- (c) hidden-state handoff (EAGLE) --------------------------------------------------
for n_tok, mult, label in [(1, 1, "1 token"), (60, 1, "60-token tree"), (60, 3, "60-token tree, EAGLE-3 (3 layers)")]:
    h = torch.randn(1, n_tok, hidden * mult, dtype=torch.float16, device=dev)
    kb = h.numel() * 2 / 1024
    add(f"(c) hidden states, {label}: GPU copy", time_us(lambda: h.clone()), f"{kb:.1f} KB")
    if dev == "cuda":
        add(f"(c) hidden states, {label}: GPU->CPU", time_us(lambda: h.cpu()), f"{kb:.1f} KB")


# ---- (d) draft-logits handoff (HF passes k x vocab logits, converted to fp32) ------------
for k in (4, 10):
    lg = torch.randn(1, k, vocab, dtype=torch.float16, device=dev)
    mb = k * vocab * 4 / 1e6
    add(f"(d) draft logits, k={k}: fp16->fp32 copy", time_us(lambda: lg.to(torch.float32)),
        f"{mb:.1f} MB in fp32")


# ---- (e) EAGLE-style tree mask built on the CPU ------------------------------------------
n_nodes = 60
parents_gpu = torch.randint(-1, n_nodes, (n_nodes,), device=dev)
def build_mask():
    parents = parents_gpu.tolist()                 # GPU -> CPU (sync)
    mask = torch.eye(n_nodes + 1).bool()           # built on the CPU
    mask[:, 0] = True
    for i in range(n_nodes):                       # Python loop over tree nodes
        p = parents[i]
        if 0 <= p < i:
            mask[i + 1].add_(mask[p + 1])
    return mask.to(dev)                            # CPU -> GPU
add("(e) EAGLE tree mask: .tolist() + Python loop + copy back", time_us(build_mask, iters=100),
    f"{n_nodes} nodes")


# ---- Report ----------------------------------------------------------------------------
print(f"{'operation':58s} {'time':>10s} {'% of target step':>17s}  note")
print("-" * 110)
for name, us, note in results:
    print(f"{name:58s} {us:8.1f} us {100*us/step_us:15.2f}%  {note}")
