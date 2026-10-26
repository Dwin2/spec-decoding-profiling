"""
step3.py -- When does each layer finish inside ONE forward pass?

For both the target and the draft, we time one decode step (1 new token) and record a
GPU timestamp right after each piece of the model finishes:
    embedding  ->  layer 0 ... layer N-1  ->  final norm + lm_head (vocab projection)

This tells us:
  1. How early intermediate features (e.g. a middle layer's output) are ready
     -> how early a "layer-pipelined" draft could start.
  2. How the pass time splits between the transformer layers and the vocab head.
"""
import time, statistics, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)

T = "Qwen/Qwen2.5-7B-Instruct"   # target
D = "Qwen/Qwen2.5-0.5B-Instruct"   # draft

tok = AutoTokenizer.from_pretrained(T)
target = AutoModelForCausalLM.from_pretrained(T, dtype=torch.float16, device_map="cuda").eval()
draft  = AutoModelForCausalLM.from_pretrained(D, dtype=torch.float16).to(dev).eval()

msgs = [{"role": "user", "content": "Write a Python function to check if a string is a palindrome, then explain it."}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt").to(dev)


# ------------------------------------------------------------
# A tiny timer: GPU timestamps on CUDA, wall clock otherwise
# ------------------------------------------------------------
def stamp():
    if dev == "cuda":
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e
    return time.perf_counter()

def ms_between(a, b):
    if dev == "cuda":
        return a.elapsed_time(b)          # milliseconds, measured on the GPU
    return (b - a) * 1000

def finish():
    if dev == "cuda":
        torch.cuda.synchronize()


# ------------------------------------------------------------
# Time one decode step of `model`, with a timestamp after every layer
# ------------------------------------------------------------
@torch.no_grad()
def layer_timeline(model, trials=30):
    layers = model.model.layers
    marks = {}                                            # name -> timestamp

    def make_hook(name):
        def hook(module, inp, out):
            marks[name] = stamp()
        return hook

    handles = [model.model.embed_tokens.register_forward_hook(make_hook("embed"))]
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(i)))

    out = model(**inputs, use_cache=True)                 # prefill the prompt once
    past = out.past_key_values
    nxt = out.logits[:, -1:].argmax(-1)

    runs = []
    for t in range(trials + 5):                           # first 5 = warmup
        marks.clear()
        start = stamp()
        model(input_ids=nxt, past_key_values=past, use_cache=True)   # ONE decode step
        end = stamp()
        finish()
        past.crop(-1)                                     # undo, so every step starts the same
        if t >= 5:
            row = {"embed": ms_between(start, marks["embed"])}
            for i in range(len(layers)):
                row[i] = ms_between(start, marks[i])
            row["end"] = ms_between(start, end)
            runs.append(row)

    for h in handles:
        h.remove()

    # median over trials for every timestamp
    keys = runs[0].keys()
    return {k: statistics.median(r[k] for r in runs) for k in keys}, len(layers)


def report(name, model):
    tl, n = layer_timeline(model)
    total = tl["end"]
    cfg = model.config
    print(f"\n=== {name}: {n} layers, hidden size {cfg.hidden_size}, vocab {cfg.vocab_size} ===")
    print(f"one decode step: {total:.2f} ms")
    print(f"  embedding done        at {tl['embed']:6.2f} ms ({100*tl['embed']/total:5.1f}%)")
    for i in range(n):
        t = tl[i]
        mark = ""
        if i == n // 4:      mark = "  <- ~1/4 of layers (low)"
        if i == n // 2:      mark = "  <- middle layer"
        if i == 3 * n // 4:  mark = "  <- ~3/4 of layers (high)"
        print(f"  layer {i:2d} done        at {t:6.2f} ms ({100*t/total:5.1f}%){mark}")
    layers_ms = tl[n - 1] - tl["embed"]
    head_ms = total - tl[n - 1]
    print(f"  ---")
    print(f"  all {n} layers      : {layers_ms:6.2f} ms ({100*layers_ms/total:5.1f}%), "
          f"{layers_ms/n:.3f} ms per layer")
    print(f"  final norm + lm_head: {head_ms:6.2f} ms ({100*head_ms/total:5.1f}%)")
    return total, layers_ms / n, head_ms


t_total, t_per_layer, t_head = report("TARGET (Qwen2.5-7B)", target)
d_total, d_per_layer, d_head = report("DRAFT  (Qwen2.5-0.5B)", draft)

print("\n=== Comparison ===")
print(f"time per layer : target {t_per_layer:.3f} ms vs draft {d_per_layer:.3f} ms "
      f"(draft/target = {d_per_layer/t_per_layer:.2f})")
print(f"lm_head        : target {t_head:.2f} ms vs draft {d_head:.2f} ms")
print(f"whole step     : target {t_total:.2f} ms vs draft {d_total:.2f} ms "
      f"(draft/target = {d_total/t_total:.2f})")
