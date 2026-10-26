import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- 1. Pick the device ----
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", dev)

def sync():
    # GPU work is asynchronous: wait for it to finish before reading the clock
    if dev == "cuda": torch.cuda.synchronize()
    elif dev == "mps": torch.mps.synchronize()

# ---- 2. Load the models (same family, so they share a tokenizer) ----
T = "Qwen/Qwen2.5-7B-Instruct"     # target
D = "Qwen/Qwen2.5-0.5B-Instruct"   # draft

tok = AutoTokenizer.from_pretrained(T)
target = AutoModelForCausalLM.from_pretrained(T, dtype=torch.float16, device_map="cuda").eval()
draft  = AutoModelForCausalLM.from_pretrained(D, dtype=torch.float16).to(dev).eval()
draft.resize_token_embeddings(target.config.vocab_size)   # 7B pads its vocab to 152064; match it (same tokenizer)

# ---- 3. Build the prompt ----
msgs = [{"role": "user", "content": "Write a Python function to check if a string is a palindrome, then explain it."}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt").to(dev)
prompt_len = inputs["input_ids"].shape[1]

# ---- 4. Time one generation ----
def run(**kw):
    target.generate(**inputs, max_new_tokens=8, do_sample=False, **kw)    # warmup, not timed
    sync(); t0 = time.perf_counter()
    out = target.generate(**inputs, max_new_tokens=200, do_sample=False, **kw)
    sync(); dt = time.perf_counter() - t0
    n = out.shape[1] - prompt_len                                         # new tokens only
    return n / dt, n, dt, out

# ---- 5. Run both ways ----
alone, n_a, dt_a, out_a = run()                          # normal decoding
spec,  n_s, dt_s, out_s = run(assistant_model=draft)     # speculative decoding

# ---- 6. Results ----
print(f"target alone : {n_a} tokens in {dt_a:.2f}s = {alone:.1f} tok/s")
print(f"with draft   : {n_s} tokens in {dt_s:.2f}s = {spec:.1f} tok/s")
print(f"speedup      : {spec/alone:.2f}x")
print("same output? ", torch.equal(out_a, out_s))
if dev == "cuda":
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

print("\n--- generated text ---")
print(tok.decode(out_s[0][prompt_len:], skip_special_tokens=True))
