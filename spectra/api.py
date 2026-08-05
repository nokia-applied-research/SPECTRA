"""SPECTRA easy API — compress an HF model's KV cache in one call, then save / load / generate.

    import spectra
    model, tok = spectra.compress("meta-llama/Llama-2-7b-hf", ratio="9x")   # one line
    print(tok.decode(model.generate(**tok("The KV cache is", return_tensors="pt").to(model.device),
                                    max_new_tokens=40)[0]))
    spectra.save(model, "spectra_llama2_8x")          # tiny delta (factors only, ~tens of MB)
    model, tok = spectra.load("spectra_llama2_8x")    # rebuilds from the HF base + delta

`ratio` picks a tested operating point (see PRESETS below / the paper). Override with `bits=`/`B=` for a
custom point. Compression replaces k_proj/v_proj with absorbed low-rank+quantized latent
projections and heals o_proj in closed form — training-free, standard `model.generate` works.
"""
from __future__ import annotations
import json, os, time
from types import SimpleNamespace
import torch

from .core import AttnHeads, collect_grams_and_mean, make_windows
from .method import build_spectra, LatentProjV2, LatentProjWF

# Tested operating points from the HP sweep (Llama-3.1-8B, WikiText-2 PPL proxy; generalizes across
# models since B scales with kv_dim). Each preset carries the FULL config (not just b_frac/bits):
#   mode      "wf"      = full-rank(-ish) + group water-filling (near-lossless, 3-6x)
#             "lowrank" = energy-ordered rank truncation + quant (aggressive, 9-12x)
#   b_frac    latent width B as a fraction of the fp16 KV width (2*kv_dim)
#   bits      avg bits/elem;  kv_ratio  K-rank fraction (V = 1-kv_ratio)
#   group     quant group size;  heavy_k  #top channels kept at 8-bit;  hadamard  rotate light block
# Regime findings: WF wants kv_ratio=0.5 (equal K/V); lowrank wants kv_ratio=0.33 (less-key-more-value)
# + heavy_k=32; 12x additionally benefits from a Hadamard rotation on the light block.
PRESETS = {
    "3x":  dict(mode="wf",      b_frac=1.0,   bits=4, kv_ratio=0.5,  group=64,  heavy_k=0,  hadamard=False),  # ~3.6x, +0.03 PPL
    "5x":  dict(mode="wf",      b_frac=1.0,   bits=3, kv_ratio=0.5,  group=64,  heavy_k=0,  hadamard=False),  # ~4.6x, +0.14
    "6x":  dict(mode="wf",      b_frac=0.75,  bits=3, kv_ratio=0.5,  group=128, heavy_k=0,  hadamard=False),  # ~6.6x, +0.51
    "9x":  dict(mode="lowrank", b_frac=0.5,   bits=3, kv_ratio=0.33, group=128, heavy_k=32, hadamard=False),  # ~9.5x, +1.07
    "12x": dict(mode="lowrank", b_frac=0.375, bits=3, kv_ratio=0.33, group=64,  heavy_k=32, hadamard=True),   # ~12.2x, +1.70
    # Paging-friendly symmetric uniform-bit LR for the vLLM int4 latent cache (spectra.vllm). Trades a
    # little quality (kv_ratio=0.5, heavy_k=0, uniform 4-bit) for a fixed-width, byte-aligned latent.
    "vllm-7x": dict(mode="lowrank", b_frac=0.5, bits=4, kv_ratio=0.5, group=64, heavy_k=0, hadamard=False),   # ~7.1x
}


def _default_args(**ov):
    a = dict(kv_ratio=0.5, wo_rank=4, mla_ridge=1e-3, wo_ridge=1e-2, group=64,
             center=False, heavy_k=0, heavy_bits=8, hadamard=False,
             water_filling=False, wf_bmax=8, wf_clip=4.0, wf_residual=0)
    a.update(ov)
    return SimpleNamespace(**a)


def _resolve_device(device):
    if device is not None:
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _calib_windows(tok, seq_len, n, text=None):
    if text is None:
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(t for t in ds["text"] if t.strip())
    return make_windows(tok, text, seq_len, n)


@torch.no_grad()
def compress(model, tokenizer=None, ratio="9x", *, bits=None, B=None,
             calib_text=None, calib_windows=64, seq_len=2048,
             device=None, dtype=torch.float16, verbose=True):
    """Compress `model` (an HF model or a HF model-id string) in place. Returns (model, tok).

    ratio: one of PRESETS ("3x","5x","6x","9x","12x","vllm-7x") — or override with bits=/B=.
    calib_*: tiny activation calibration (default 64 wikitext-2 windows; ~seconds on a GPU).
    """
    torch.set_grad_enabled(False)
    dev = _resolve_device(device)
    if isinstance(model, str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_id = model
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(dev).eval()
    else:
        model_id = getattr(model.config, "_name_or_path", "unknown")
        if tokenizer is None:
            raise ValueError("pass a tokenizer when compressing an already-loaded model")
        model = model.to(dev).eval()
    tok = tokenizer

    preset = dict(PRESETS.get(ratio, {}))
    if not preset and (B is None or bits is None):
        raise ValueError(f"unknown ratio '{ratio}'; choose {list(PRESETS)} or pass B= and bits=")
    mode = preset.get("mode", "wf" if (B and bits and bits >= 4) else "lowrank")
    eff_bits = bits if bits is not None else preset["bits"]

    heads = AttnHeads(model)
    kv_full = 2 * heads.kv_dim                                   # fp16 KV width
    eff_B = B if B is not None else max(1, int(round(preset["b_frac"] * kv_full)))

    args = _default_args(
        water_filling=(mode == "wf"),
        kv_ratio=preset.get("kv_ratio", 0.5 if mode == "wf" else 0.33),
        center=(mode == "wf"),
        group=preset.get("group", 64),
        heavy_k=preset.get("heavy_k", 0),
        hadamard=preset.get("hadamard", False),
    )

    if verbose:
        print(f"[spectra] {model_id}  arch={heads.arch} L={heads.L} kv_dim={heads.kv_dim}", flush=True)
        print(f"[spectra] ratio={ratio} mode={mode} B={eff_B} bits={eff_bits}", flush=True)

    t0 = time.time()
    cw = _calib_windows(tok, seq_len, calib_windows, calib_text)
    G, hmean = collect_grams_and_mean(model, heads, cw, dev)
    WKo = [l.self_attn.k_proj.weight.detach().clone() for l in heads.layers]
    WVo = [l.self_attn.v_proj.weight.detach().clone() for l in heads.layers]
    WOo = [l.self_attn.o_proj.weight.detach().clone() for l in heads.layers]
    kb = [(l.self_attn.k_proj.bias.detach().clone() if l.self_attn.k_proj.bias is not None else None)
          for l in heads.layers]
    vb = [(l.self_attn.v_proj.bias.detach().clone() if l.self_attn.v_proj.bias is not None else None)
          for l in heads.layers]

    r_k, r_v, cf = build_spectra(model, heads, G, hmean, WKo, WVo, WOo, kb, vb,
                                 cw, dev, dtype, eff_B, eff_bits, args)
    model._spectra = dict(base_model=model_id, ratio=ratio, mode=mode, B=eff_B, bits=eff_bits,
                          r_k=r_k, r_v=r_v, cache_frac=cf, compression=round(1.0 / cf, 2),
                          group=args.group)
    if verbose:
        print(f"[spectra] done in {time.time()-t0:.1f}s  cache={cf:.4f}  "
              f"compression={1/cf:.2f}x  (r_k={r_k} r_v={r_v})", flush=True)
    return model, tok


# --------------------------------------------------------------------- save / load (lean delta)
def _module_spec(m):
    """Serialize a LatentProj module to (kind, scalars, tensors)."""
    t = dict(W_DOWN=m.W_DOWN, W_UP=m.W_UP, mu=m.mu, beta=m.beta)
    if m.bias is not None:
        t["bias"] = m.bias
    if isinstance(m, LatentProjWF):
        t["group_bits"] = m.group_bits
        return "wf", dict(group_size=m.group_size), t
    return "v2", dict(bits=m.bits, group=m.group, heavy_k=m.heavy_k, heavy_bits=m.heavy_bits), t


def _build_module(kind, sc, t):
    bias = t.get("bias")
    if kind == "wf":
        return LatentProjWF(t["W_DOWN"], t["W_UP"], t["mu"], t["beta"],
                            t["group_bits"], sc["group_size"], bias)
    return LatentProjV2(t["W_DOWN"], t["W_UP"], t["mu"], t["beta"],
                        sc["bits"], sc["group"], sc["heavy_k"], sc["heavy_bits"], bias)


def save(model, path):
    """Save a SPECTRA-compressed model as a lean delta (factors + healed o_proj only). The HF
    base weights are NOT duplicated — `load` re-fetches them by id. Writes <path>/spectra.pt and
    <path>/meta.json."""
    if not hasattr(model, "_spectra"):
        raise ValueError("model was not compressed by spectra.compress()")
    os.makedirs(path, exist_ok=True)
    heads = AttnHeads(model)
    layers = []
    for l in heads.layers:
        sa = l.self_attn
        kk, ks, kt = _module_spec(sa.k_proj)
        vk, vs, vt = _module_spec(sa.v_proj)
        layers.append(dict(k_kind=kk, k_scalars=ks, k_tensors={n: x.cpu() for n, x in kt.items()},
                           v_kind=vk, v_scalars=vs, v_tensors={n: x.cpu() for n, x in vt.items()},
                           o_proj_weight=sa.o_proj.weight.detach().cpu()))
    meta = dict(format="spectra-delta-v1", **model._spectra)
    torch.save(dict(meta=meta, layers=layers), os.path.join(path, "spectra.pt"))
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return path


@torch.no_grad()
def load(path, tokenizer=None, *, device=None, dtype=torch.float16, verbose=True):
    """Rebuild a compressed model from a delta saved by `save`: fetch the HF base by id, overlay
    the SPECTRA factors and healed o_proj. Returns (model, tok)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_grad_enabled(False)
    dev = _resolve_device(device)
    blob = torch.load(os.path.join(path, "spectra.pt"), map_location="cpu", weights_only=False)
    meta, layers = blob["meta"], blob["layers"]
    base = meta["base_model"]
    if verbose:
        print(f"[spectra] loading base {base} + delta ({meta['ratio']}, {meta['compression']}x)", flush=True)
    tok = tokenizer or AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(dev).eval()
    heads = AttnHeads(model)
    for l, spec in zip(heads.layers, layers):
        sa = l.self_attn
        kt = {n: x.to(dev, dtype) for n, x in spec["k_tensors"].items()}
        vt = {n: x.to(dev, dtype) for n, x in spec["v_tensors"].items()}
        if "group_bits" in kt:
            kt["group_bits"] = kt["group_bits"].long()
        if "group_bits" in vt:
            vt["group_bits"] = vt["group_bits"].long()
        sa.k_proj = _build_module(spec["k_kind"], spec["k_scalars"], kt).to(dev)
        sa.v_proj = _build_module(spec["v_kind"], spec["v_scalars"], vt).to(dev)
        sa.o_proj.weight.copy_(spec["o_proj_weight"].to(dev, dtype))
    model._spectra = {k: v for k, v in meta.items() if k != "format"}
    return model, tok


# ----------------------------------------------------------------------------------------- CLI
def _main():
    import argparse
    ap = argparse.ArgumentParser(description="SPECTRA easy CLI: compress an HF model & save the delta.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compress", help="compress an HF model and save a SPECTRA delta")
    c.add_argument("--model", required=True)
    c.add_argument("--ratio", default="9x", help=f"one of {list(PRESETS)} or use --B/--bits")
    c.add_argument("--bits", type=int, default=None)
    c.add_argument("--B", type=int, default=None)
    c.add_argument("--calib-windows", type=int, default=16)
    c.add_argument("--device", default=None)
    c.add_argument("--out", required=True)
    c.add_argument("--prompt", default=None, help="optional: generate a sample after compressing")
    g = sub.add_parser("generate", help="load a saved delta and generate")
    g.add_argument("--path", required=True)
    g.add_argument("--prompt", default="The KV cache is")
    g.add_argument("--max-new-tokens", type=int, default=60)
    g.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.cmd == "compress":
        model, tok = compress(args.model, ratio=args.ratio, bits=args.bits, B=args.B,
                              calib_windows=args.calib_windows, device=args.device)
        save(model, args.out)
        print(f"[spectra] saved delta -> {args.out}")
        if args.prompt:
            ids = tok(args.prompt, return_tensors="pt").to(model.device)
            out = model.generate(**ids, max_new_tokens=60, do_sample=False)
            print("[sample]", tok.decode(out[0], skip_special_tokens=True))
    else:
        model, tok = load(args.path, device=args.device)
        ids = tok(args.prompt, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=args.max_new_tokens, do_sample=False)
        print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    _main()
