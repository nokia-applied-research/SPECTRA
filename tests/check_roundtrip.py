"""GPU check: compress -> save -> load gives the same model, and the exact window is restored.

    python tests/check_roundtrip.py --model NousResearch/Meta-Llama-3.1-8B-Instruct --preset 8.8x
"""
import argparse, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import spectra

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
ap.add_argument("--preset", default="8.8x")
ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()

model, tok = spectra.compress(a.model, preset=a.preset, device=a.device)
meta = model._spectra
print(f"[check] preset={meta['preset']} config={meta['config']} latent ratio={meta['compression']}x "
      f"r_k={meta['r_k']} r_v={meta['r_v']} window={meta['window']}")

text = ("The KV cache stores one key and one value vector per token and layer. " * 120
        + "\nIn one sentence, what does the KV cache store?")
ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                              return_tensors="pt", return_dict=True).to(a.device)
print(f"[check] prompt {ids['input_ids'].shape[1]} tokens")
with torch.no_grad():
    logits_a = model(**ids).logits[0, -1].float()
    out_a = model.generate(**ids, max_new_tokens=40, do_sample=False)[0, ids["input_ids"].shape[1]:]

with tempfile.TemporaryDirectory() as d:
    spectra.save(model, d)
    size = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)) / 2**20
    del model; torch.cuda.empty_cache()
    model2, _ = spectra.load(d, device=a.device)
with torch.no_grad():
    logits_b = model2(**ids).logits[0, -1].float()
    out_b = model2.generate(**ids, max_new_tokens=40, do_sample=False)[0, ids["input_ids"].shape[1]:]

ok = True
diff = (logits_a - logits_b).abs().max().item()
print(f"[check] delta size {size:.0f} MiB; max |logit diff| after load {diff:.2e}")
ok &= diff < 1e-2
ok &= torch.equal(out_a, out_b)
ok &= (model2._spectra["window"] == meta["window"])
print("[check] output:", repr(tok.decode(out_a, skip_special_tokens=True)))
print("[check] same tokens after load:", torch.equal(out_a, out_b))
print("ALL OK" if ok else "CHECK FAILED")
sys.exit(0 if ok else 1)
