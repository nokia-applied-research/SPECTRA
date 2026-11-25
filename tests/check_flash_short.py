"""Short sequences through the SPECTRA-Flash HF path (the exact window not yet full): pod check, H100.

A short chat prompt; HF dense decodes `--tg` greedy tokens, so the positions cross sink + recent + 1 = 128. The
continuation is teacher-forced through HF dense and the Flash HF path (backends fast and torch), and the per-step
logits are compared in two ranges:
  positions < 128 : every key is still in the exact window, so Flash must equal dense attention (bf16 noise only);
  positions >= 128: the record holds the older tokens (the approximation).
usage: python tests/check_flash_short.py --params <params dir or HF Hub repo>
"""
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import spectra.flash as BF_

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
ap.add_argument("--params", required=True)
ap.add_argument("--tg", type=int, default=160)
a = ap.parse_args()

dev = torch.device("cuda")
tok = AutoTokenizer.from_pretrained(a.model)
msg = [{"role": "user", "content": "Write a short story about a lighthouse keeper who finds a message in a bottle."}]
ids = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt")
ids = (ids["input_ids"] if not torch.is_tensor(ids) else ids)[0].to(dev)
Tp = ids.numel()
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
with torch.inference_mode():
    cont = model.generate(ids[None], max_new_tokens=a.tg, min_new_tokens=a.tg, do_sample=False, temperature=None,
                          top_p=None)[0, Tp:]


@torch.inference_mode()
def forced(cache):
    out = model(ids[None], past_key_values=cache, use_cache=True)
    rows = [out.logits[0, -1].float()]
    for t in cont[:-1]:
        out = model(t.view(1, 1), past_key_values=out.past_key_values, use_cache=True)
        rows.append(out.logits[0, -1].float())
    return torch.stack(rows)


def kl(x, y):
    lx, ly = F.log_softmax(x, -1), F.log_softmax(y, -1)
    return (lx.exp() * (lx - ly)).sum(-1)


ld = forced(None)
pos = torch.arange(Tp - 1, Tp - 1 + cont.numel(), device=dev)        # position of the token each row is computed at
print(f"[short] prompt {Tp} tokens, {cont.numel()} teacher-forced steps (positions {Tp - 1}..{int(pos[-1])})", flush=True)
ok = True
for backend in ("fast", "torch"):
    BF_.enable(model, a.params, backend=backend)
    lf = forced(model.spectra_flash.new_cache())
    BF_.disable(model)
    for name, sel in (("positions < 128", pos < 128), ("positions >= 128", pos >= 128)):
        if sel.any():
            agree = (lf[sel].argmax(-1) == ld[sel].argmax(-1)).float().mean().item()
            k = kl(ld[sel], lf[sel]).mean().item()
            print(f"[short] {backend:5s} vs dense, {name:17s}: argmax agree {agree:.3f}  KL {k:.5f}  "
                  f"({int(sel.sum())} steps)", flush=True)
            if name.startswith("positions <"):
                ok &= agree > 0.97 and k < 5e-3
print("SHORT_OK" if ok else "SHORT_FAIL", flush=True)
