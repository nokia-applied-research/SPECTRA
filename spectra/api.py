"""SPECTRA easy API: compress an HF model's KV cache in one call, then save / load / generate.

    import spectra
    model, tok = spectra.compress("NousResearch/Meta-Llama-3.1-8B-Instruct", preset="8.8x")
    spectra.save(model, "spectra_l31_8.8x")          # small delta: factors + healed o_proj
    model, tok = spectra.load("spectra_l31_8.8x")    # rebuilds from the HF base + delta

`preset` is one of the paper's operating points, named by its KV compression on Llama-3.1-8B (NAMED), or an
internal config name (PRESETS; a `-B` suffix keeps the first 4 and the last 64 prompt tokens of K and V exact).
See docs/presets.md. Override with `B=` and `bits=` for a custom point. Compression replaces k_proj / v_proj with
latent projections and heals o_proj in closed form; no training, and `model.generate` works.
"""
from __future__ import annotations
import json, os, time
from types import SimpleNamespace
import torch

from .core import AttnHeads, collect_grams_and_mean, make_windows
from .method import build_spectra, set_exact_window, LatentProjV2, LatentProjWF
from .accounting import honest_ratio

# The paper's operating points. B is the latent width summed over K and V (capped per side at the KV width of the
# model), bits the average bits per latent value. mode "wf" = water-filling bit allocation over the energy-ordered
# latent; "lowrank" = rank truncation with uniform bits, the top `heavy_k` channels at 8 bits, and optionally a
# random orthogonal rotation of the rest (`hadamard`, the historical name).
# The same preset gives different ratios on different models; the README lists the measured ones.
PRESETS = {
    "s35":   dict(mode="wf",      B=2048, bits=4, kv_ratio=0.5,  group=64,  heavy_k=0,  hadamard=False, center=True),
    "t40b":  dict(mode="wf",      B=1792, bits=4, kv_ratio=0.5,  group=64,  heavy_k=0,  hadamard=False, center=True),
    "s45":   dict(mode="wf",      B=2048, bits=3, kv_ratio=0.5,  group=64,  heavy_k=0,  hadamard=False, center=True),
    "s65":   dict(mode="wf",      B=1536, bits=3, kv_ratio=0.5,  group=128, heavy_k=0,  hadamard=False, center=True),
    "L48":   dict(mode="lowrank", B=2048, bits=3, kv_ratio=0.33, group=128, heavy_k=32, hadamard=False, center=False),
    "L55":   dict(mode="lowrank", B=1792, bits=3, kv_ratio=0.33, group=128, heavy_k=32, hadamard=False, center=False),
    "L65":   dict(mode="lowrank", B=1536, bits=3, kv_ratio=0.33, group=128, heavy_k=32, hadamard=False, center=False),
    "L70":   dict(mode="lowrank", B=1408, bits=3, kv_ratio=0.33, group=128, heavy_k=32, hadamard=False, center=False),
    "t80e":  dict(mode="lowrank", B=1216, bits=3, kv_ratio=0.33, group=128, heavy_k=32, hadamard=False, center=False),
    "s95":   dict(mode="lowrank", B=1024, bits=3, kv_ratio=0.33, group=128, heavy_k=32, hadamard=False, center=False),
    "s12":   dict(mode="lowrank", B=768,  bits=3, kv_ratio=0.33, group=64,  heavy_k=32, hadamard=True,  center=False),
    # narrow-KV models (Qwen2.5-7B: 4 KV heads) reach the 8x tier only with a narrower latent
    "w544":  dict(mode="lowrank", B=544,  bits=3, kv_ratio=0.33, group=128, heavy_k=32, hadamard=False, center=False),
    "w512":  dict(mode="lowrank", B=512,  bits=3, kv_ratio=0.33, group=128, heavy_k=32, hadamard=False, center=False),
    "w512h": dict(mode="lowrank", B=512,  bits=3, kv_ratio=0.33, group=64,  heavy_k=32, hadamard=True,  center=False),
    "w448":  dict(mode="lowrank", B=448,  bits=3, kv_ratio=0.33, group=128, heavy_k=32, hadamard=False, center=False),
    # symmetric uniform 4-bit latent with a byte-aligned width, for the vLLM int4 cache (spectra.vllm); the width
    # scales with the model (b_frac of the fp16 KV width)
    "vllm-7x": dict(mode="lowrank", b_frac=0.5, bits=4, kv_ratio=0.5, group=64, heavy_k=0, hadamard=False, center=False),
}
# Public names: the KV compression each paper point reaches on Llama-3.1-8B (LongBench, exact window counted).
NAMED = {"3.6x": "s35", "4x": "t40b-B", "4.6x": "s45", "5.5x": "L48-B", "5.8x": "L55-B", "6x": "L65-B",
         "6.7x": "L70-B", "7.7x": "t80e-B", "8.8x": "s95-B", "11x": "s12-B"}
# v0.1 names (no exact window), kept for compatibility.
ALIASES = {"3x": "s35", "5x": "s45", "9x": "s95", "12x": "s12"}
SINK, RECENT = 4, 64                       # the `-B` exact window: 68 tokens of K and V kept in fp16
CALIB_RECIPES = ("wikitext+code", "wikitext")


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


def parse_preset(name):
    """'11x' or 's12-B' -> ('s12', config dict, window on). Also accepts the v0.1 names ('9x', ...)."""
    name = NAMED.get(name, name)
    window = name.endswith("-B")
    base = name[:-2] if window else name
    base = ALIASES.get(base, base)
    if base not in PRESETS:
        raise ValueError(f"unknown preset '{name}'; choose one of {list(NAMED)} (see docs/presets.md), "
                         f"or pass B= and bits=")
    return base, dict(PRESETS[base]), window


def make_calib_windows(tok, recipe="wikitext+code", n=16, seq_len=2048, text=None):
    """Calibration windows of `seq_len` tokens. 'wikitext+code' (default, the paper's recipe): n/2 windows from
    WikiText-2 train and n/2 from codeparrot-clean-valid. 'wikitext': n windows of WikiText-2 train (the recipe of
    arXiv v1). `text` overrides both with your own corpus."""
    if text is not None:
        return make_windows(tok, text, seq_len, n)
    from datasets import load_dataset
    nl = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"]
                     if t.strip())
    if recipe == "wikitext":
        return make_windows(tok, nl, seq_len, n)
    if recipe != "wikitext+code":
        raise ValueError(f"calib must be one of {CALIB_RECIPES}")
    code = "\n\n".join(str(x) for x in
                       load_dataset("codeparrot/codeparrot-clean-valid", split="train")[:800]["content"])
    nl_ids = tok(nl, return_tensors="pt").input_ids[0]
    code_ids = tok(code, return_tensors="pt").input_ids[0]
    n_code = n // 2; n_nl = n - n_code
    wins = ([nl_ids[i * seq_len:(i + 1) * seq_len] for i in range(n_nl)]
            + [code_ids[i * seq_len:(i + 1) * seq_len] for i in range(n_code)])
    return [w for w in wins if w.numel() == seq_len]


@torch.no_grad()
def compress(model, tokenizer=None, preset="8.8x", *, ratio=None, B=None, bits=None, window=None,
             calib="wikitext+code", calib_windows=16, calib_text=None, seq_len=2048,
             device=None, dtype=torch.float16, verbose=True):
    """Compress `model` (an HF model or a model id) in place. Returns (model, tok).

    preset : a NAMED preset ('3.6x' ... '11x') or a PRESETS name. `ratio=` is the v0.1 name of this argument.
    B, bits: a custom operating point (latent width, average bits) instead of a preset.
    window : keep the first 4 and last 64 prompt tokens exact (default: on for `-B` presets).
    calib  : 'wikitext+code' (the paper's recipe, 8 + 8 windows of 2048 tokens) or 'wikitext'.
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
    heads = AttnHeads(model)

    name = ratio if ratio is not None else preset
    if B is not None and bits is not None:
        mode = "wf" if bits >= 4 else "lowrank"
        cfg = dict(mode=mode, B=B, bits=bits, kv_ratio=0.5 if mode == "wf" else 0.33, group=64,
                   heavy_k=0, hadamard=False, center=(mode == "wf"))          # as in v0.1
        base, keep = f"custom-B{B}-b{bits}", bool(window)
    else:
        base, cfg, keep = parse_preset(name)
        keep = keep if window is None else bool(window)
    eff_B = cfg["B"] if "B" in cfg else max(1, int(round(cfg["b_frac"] * 2 * heads.kv_dim)))
    args = _default_args(water_filling=(cfg["mode"] == "wf"), kv_ratio=cfg["kv_ratio"], center=cfg["center"],
                         group=cfg["group"], heavy_k=cfg["heavy_k"], heavy_bits=8, hadamard=cfg["hadamard"])
    config = base + ("-B" if keep else "")
    label = name if (name in NAMED and NAMED[name] == config) else config
    if verbose:
        print(f"[spectra] {model_id}  arch={heads.arch} L={heads.L} kv_dim={heads.kv_dim}", flush=True)
        print(f"[spectra] preset={label} mode={cfg['mode']} B={eff_B} bits={cfg['bits']} calib={calib}", flush=True)

    t0 = time.time()
    cw = make_calib_windows(tok, calib, calib_windows, seq_len, calib_text)
    G, hmean = collect_grams_and_mean(model, heads, cw, dev)
    WKo = [l.self_attn.k_proj.weight.detach().clone() for l in heads.layers]
    WVo = [l.self_attn.v_proj.weight.detach().clone() for l in heads.layers]
    WOo = [l.self_attn.o_proj.weight.detach().clone() for l in heads.layers]
    kb = [(l.self_attn.k_proj.bias.detach().clone() if l.self_attn.k_proj.bias is not None else None)
          for l in heads.layers]
    vb = [(l.self_attn.v_proj.bias.detach().clone() if l.self_attn.v_proj.bias is not None else None)
          for l in heads.layers]
    r_k, r_v, cf = build_spectra(model, heads, G, hmean, WKo, WVo, WOo, kb, vb,
                                 cw, dev, dtype, eff_B, cfg["bits"], args)
    if keep:
        set_exact_window(heads, WKo, WVo, kb, vb, SINK, RECENT)
    comp = round(1.0 / cf, 2)
    model._spectra = dict(base_model=model_id, preset=label, config=config, mode=cfg["mode"], B=eff_B, bits=cfg["bits"],
                          r_k=r_k, r_v=r_v, cache_frac=cf, compression=comp, group=args.group,
                          window=[SINK, RECENT] if keep else None, calib=calib, calib_windows=len(cw))
    if verbose:
        msg = f"[spectra] done in {time.time() - t0:.1f}s  latent ratio {comp:.2f}x (r_k={r_k} r_v={r_v})"
        if keep:
            w = SINK + RECENT
            msg += (f"; with the {w}-token exact window: {honest_ratio(comp, [8192], w):.2f}x at 8K, "
                    f"{honest_ratio(comp, [32768], w):.2f}x at 32K")
        print(msg, flush=True)
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
    """Save a compressed model as a lean delta (factors + healed o_proj). The HF base weights are not duplicated:
    `load` fetches them by id. Writes <path>/spectra.pt and <path>/meta.json."""
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
    meta = dict(format="spectra-delta-v2", **model._spectra)
    torch.save(dict(meta=meta, layers=layers), os.path.join(path, "spectra.pt"))
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return path


@torch.no_grad()
def load(path, tokenizer=None, *, device=None, dtype=torch.float16, verbose=True):
    """Rebuild a compressed model from a delta saved by `save`: fetch the HF base by id, overlay the SPECTRA
    factors and healed o_proj, and restore the exact window from the base k/v weights. Returns (model, tok)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_grad_enabled(False)
    dev = _resolve_device(device)
    blob = torch.load(os.path.join(path, "spectra.pt"), map_location="cpu", weights_only=False)
    meta, layers = blob["meta"], blob["layers"]
    base = meta["base_model"]
    if verbose:
        print(f"[spectra] loading base {base} + delta ({meta.get('preset', meta.get('ratio'))}, "
              f"latent ratio {meta['compression']}x)", flush=True)
    tok = tokenizer or AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(dev).eval()
    heads = AttnHeads(model)
    WKo = [l.self_attn.k_proj.weight.detach() for l in heads.layers]
    WVo = [l.self_attn.v_proj.weight.detach() for l in heads.layers]
    kb = [(None if l.self_attn.k_proj.bias is None else l.self_attn.k_proj.bias.detach()) for l in heads.layers]
    vb = [(None if l.self_attn.v_proj.bias is None else l.self_attn.v_proj.bias.detach()) for l in heads.layers]
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
    if meta.get("window"):
        set_exact_window(heads, WKo, WVo, kb, vb, *meta["window"])
    model._spectra = {k: v for k, v in meta.items() if k != "format"}
    return model, tok


# ----------------------------------------------------------------------------------------- CLI
def _main():
    import argparse
    ap = argparse.ArgumentParser(description="SPECTRA CLI: compress an HF model and save the delta, or generate.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compress", help="compress an HF model and save a SPECTRA delta")
    c.add_argument("--model", required=True)
    c.add_argument("--preset", default="8.8x", help=f"one of {list(NAMED)} (see docs/presets.md)")
    c.add_argument("--bits", type=int, default=None)
    c.add_argument("--B", type=int, default=None)
    c.add_argument("--calib", default="wikitext+code", choices=CALIB_RECIPES)
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
        model, tok = compress(args.model, preset=args.preset, bits=args.bits, B=args.B, calib=args.calib,
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
