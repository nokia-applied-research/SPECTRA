"""niah.py — Needle-In-A-Haystack long-context retrieval benchmark.

Evaluates a model's ability to retrieve a single fact ("needle") planted at a
controlled depth inside a long filler context ("haystack"), across a grid of
(context_length x depth). Deterministic scoring: the needle is a unique magic
number; a cell is correct iff that number appears in the greedy generation.

Supports two methods on the SAME grid for apples-to-apples comparison:
  --method base                  fp16 model, no compression
  --method akvc --configs B:bits  AKVC-v2 latent-compressed + quantized KV
                                  (center+heavy-k+hadamard recipe, reused from akvc_v2)

Outputs a JSON grid per method and an ASCII heatmap (+ PNG if matplotlib present).
"""
from __future__ import annotations
import argparse, json, time, random
import torch

from ..core import (
    AttnHeads, collect_grams_and_mean, make_windows,
    fit_apply_mla_split,
    _accumulate_streaming_stats, _rrr_precompute, _rrr_apply,
)
from ..method import build_factor, LatentProjV2, build_spectra as build_akvc
from ..baselines import fake_quant_K, fake_quant_V, effective_bits
from .. import baselines as kvb_mod  # noqa: F401  (kept for parity; main() imports kv_baselines lazily)
from transformers import AutoModelForCausalLM, AutoTokenizer


# ----------------------------- NIAH grid -----------------------------------
@torch.no_grad()
def run_grid(model, tok, hay_ids, contexts, depths, magic, gen_tokens, dev, sys_prompt, question, label,
             gen_fn=None):
    """gen_fn(ids)->str overrides default fp16 generate (used for baselines)."""
    needle = f" One special detail to remember: the magic access code for this document is {magic}. "
    needle_ids = tok(needle, add_special_tokens=False)["input_ids"]
    q_full = f"\n\n{question}"
    grid = {}
    for ctx in contexts:
        grid[ctx] = {}
        budget = ctx - len(needle_ids) - 96  # leave room for chat template + question
        if budget < 64:
            for d in depths: grid[ctx][d] = None
            continue
        base_hay = hay_ids[:budget]
        for d in depths:
            pos = int(round(d / 100.0 * len(base_hay)))
            doc_ids = base_hay[:pos] + needle_ids + base_hay[pos:]
            doc_text = tok.decode(doc_ids)
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": doc_text + q_full},
            ]
            ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=False).to(dev)
            if gen_fn is not None:
                gen = gen_fn(ids)
            else:
                out = model.generate(ids, max_new_tokens=gen_tokens, do_sample=False,
                                     pad_token_id=tok.eos_token_id, use_cache=True)
                gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            ok = magic in gen.replace(",", "")
            grid[ctx][d] = int(ok)
            print(f"[{label}] ctx={ctx:>6} depth={d:>3}% real_len={ids.shape[1]:>6} -> "
                  f"{'HIT ' if ok else 'miss'} | {gen[:48]!r}", flush=True)
    return grid


def ascii_heatmap(grid, contexts, depths, title):
    lines = [f"\n=== {title} (rows=depth%, cols=ctx; #=hit .=miss) ==="]
    header = "depth\\ctx | " + " ".join(f"{c//1024 if c>=1024 else c}{'k' if c>=1024 else ''}".rjust(5) for c in contexts)
    lines.append(header)
    for d in depths:
        cells = []
        for c in contexts:
            v = grid.get(c, {}).get(d)
            cells.append("    #" if v == 1 else ("    ." if v == 0 else "    -"))
        lines.append(f"{str(d).rjust(8)} | " + " ".join(cells))
    # column accuracy
    accs = []
    for c in contexts:
        vals = [grid.get(c, {}).get(d) for d in depths]
        vals = [v for v in vals if v is not None]
        accs.append(f"{(sum(vals)/len(vals)*100):4.0f}" if vals else "  - ")
    lines.append("  acc(%) | " + " ".join(a.rjust(5) for a in accs))
    return "\n".join(lines)


def try_png(grid, contexts, depths, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        Z = np.full((len(depths), len(contexts)), np.nan)
        for i, d in enumerate(depths):
            for j, c in enumerate(contexts):
                v = grid.get(c, {}).get(d)
                if v is not None: Z[i, j] = v
        fig, ax = plt.subplots(figsize=(max(6, len(contexts)), max(4, len(depths) * 0.5)))
        im = ax.imshow(Z, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(contexts))); ax.set_xticklabels([f"{c//1024}k" if c >= 1024 else str(c) for c in contexts])
        ax.set_yticks(range(len(depths))); ax.set_yticklabels([f"{d}%" for d in depths])
        ax.set_xlabel("context length"); ax.set_ylabel("needle depth"); ax.set_title(title)
        fig.colorbar(im, ax=ax, label="retrieval (1=hit)")
        fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
        return path
    except Exception as e:
        print(f"[png] skipped ({type(e).__name__}: {e})", flush=True)
        return None


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--dtype", default="float16")
    ap.add_argument("--contexts", default="1024,2048,4096,8192,16384,32768")
    ap.add_argument("--depths", default="0,17,33,50,67,83,100")
    ap.add_argument("--gen-tokens", type=int, default=24)
    ap.add_argument("--magic", default="7429183")
    # akvc configs (optional): csv of B:bits. If empty, only baseline runs.
    ap.add_argument("--configs", default="")
    ap.add_argument("--kv-ratio", type=float, default=0.25); ap.add_argument("--wo-rank", type=int, default=4)
    ap.add_argument("--mla-ridge", type=float, default=1e-3); ap.add_argument("--wo-ridge", type=float, default=1e-2)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--center", action="store_true")
    ap.add_argument("--heavy-k", type=int, default=0); ap.add_argument("--heavy-bits", type=int, default=8)
    ap.add_argument("--hadamard", action="store_true")
    ap.add_argument("--water-filling", action="store_true",
                    help="rate-distortion per-channel bit allocation (3-5x recipe); 'B:bits' = width B, avg bits")
    ap.add_argument("--wf-bmax", type=int, default=8)
    ap.add_argument("--wf-clip", type=float, default=4.0)
    ap.add_argument("--wf-residual", type=int, default=0)
    ap.add_argument("--calib-windows", type=int, default=32); ap.add_argument("--seq-len", type=int, default=2048)
    # KIVI baseline (optional): csv of bit-widths, applied per-channel(K)/per-token(V) via hooks
    ap.add_argument("--kivi-configs", default="")
    ap.add_argument("--gk", type=int, default=128); ap.add_argument("--gv", type=int, default=32)
    ap.add_argument("--residual", type=int, default=128)
    # competing baselines (kv_baselines.py): quant = kivi/turboquant/polarquant ; evict = snapkv/pyramidkv
    # spec csv: "method:param" where param=bits (quant) or budget (evict), e.g. "turboquant:2,snapkv:1024"
    ap.add_argument("--baselines", default="")
    ap.add_argument("--out", required=True); ap.add_argument("--fig-prefix", default="")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    dev = args.device; dtype = getattr(torch, args.dtype)
    contexts = [int(x) for x in args.contexts.split(",") if x.strip()]
    depths = [int(x) for x in args.depths.split(",") if x.strip()]

    print(f"[load] {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(dev).eval()
    heads = AttnHeads(model)
    print(f"[cfg] arch={heads.arch} L={heads.L} kv_dim={heads.kv_dim} maxpos={getattr(model.config,'max_position_embeddings',None)}", flush=True)

    # haystack from wikitext-2 train (plenty of tokens for <=32k)
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train") \
        if max(contexts) > 60000 else load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    hay_text = "\n\n".join(t for t in ds["text"] if t.strip())
    hay_ids = tok(hay_text, add_special_tokens=False)["input_ids"]
    print(f"[data] haystack tokens={len(hay_ids)}", flush=True)

    sys_prompt = "You are a helpful assistant. Read the document carefully and answer the question using only information from the document."
    question = "What is the magic access code for this document? Answer with only the number."

    results = {}
    # ---- baseline ----
    t0 = time.time()
    g = run_grid(model, tok, hay_ids, contexts, depths, args.magic, args.gen_tokens, dev, sys_prompt, question, "base")
    results["base"] = {"grid": {str(c): g[c] for c in contexts}, "cache_frac": 1.0, "total_x": 1.0}
    print(ascii_heatmap(g, contexts, depths, "base (fp16)"), flush=True)
    if args.fig_prefix:
        try_png(g, contexts, depths, f"{args.fig_prefix}_base.png", f"{args.model.split('/')[-1]} NIAH — base fp16")
    print(f"[time] base {time.time()-t0:.1f}s", flush=True)

    # snapshot originals for restore between configs
    WKo = [l.self_attn.k_proj.weight.detach().clone() for l in heads.layers]
    WVo = [l.self_attn.v_proj.weight.detach().clone() for l in heads.layers]
    WOo = [l.self_attn.o_proj.weight.detach().clone() for l in heads.layers]
    kb = [(l.self_attn.k_proj.bias.detach().clone() if l.self_attn.k_proj.bias is not None else None) for l in heads.layers]
    vb = [(l.self_attn.v_proj.bias.detach().clone() if l.self_attn.v_proj.bias is not None else None) for l in heads.layers]
    orig_k = [l.self_attn.k_proj for l in heads.layers]
    orig_v = [l.self_attn.v_proj for l in heads.layers]

    def restore():
        for l in range(heads.L):
            heads.layers[l].self_attn.k_proj = orig_k[l]
            heads.layers[l].self_attn.v_proj = orig_v[l]
            orig_k[l].weight.copy_(WKo[l]); orig_v[l].weight.copy_(WVo[l])
            heads.layers[l].self_attn.o_proj.weight.copy_(WOo[l])

    cfgs = [s for s in args.configs.split(",") if s.strip()]
    if cfgs:
        G, hmean = collect_grams_and_mean(model, heads,
                                          make_windows(tok, hay_text, args.seq_len, args.calib_windows), dev)
        for spec in cfgs:
            B, bits = (int(x) for x in spec.split(":"))
            t0 = time.time(); restore()
            calib_w = make_windows(tok, hay_text, args.seq_len, args.calib_windows)
            r_k, r_v, cf = build_akvc(model, heads, G, hmean, WKo, WVo, WOo, kb, vb, calib_w, dev, dtype, B, bits, args)
            tag = f"B{B}_b{bits}"
            g = run_grid(model, tok, hay_ids, contexts, depths, args.magic, args.gen_tokens, dev, sys_prompt, question, tag)
            results[tag] = {"grid": {str(c): g[c] for c in contexts}, "cache_frac": round(cf, 4),
                            "total_x": round(1 / cf, 2), "r_k": r_k, "r_v": r_v}
            print(ascii_heatmap(g, contexts, depths, f"{tag}  ({1/cf:.1f}x cache)"), flush=True)
            if args.fig_prefix:
                try_png(g, contexts, depths, f"{args.fig_prefix}_{tag}.png",
                        f"{args.model.split('/')[-1]} NIAH — {tag} ({1/cf:.1f}x)")
            print(f"[time] {tag} {time.time()-t0:.1f}s", flush=True)
        restore()

    # ---- KIVI baseline: per-channel K / per-token V fake-quant via forward hooks ----
    kivi_cfgs = [s for s in args.kivi_configs.split(",") if s.strip()]
    if kivi_cfgs:
        restore()
        state = {"bits": None}

        def mk_k():
            def hook(mod, a_, out):
                if state["bits"] is None: return out
                shp = out.shape
                return fake_quant_K(out.reshape(-1, shp[-1]), state["bits"], args.gk, args.residual).reshape(shp)
            return hook

        def mk_v():
            def hook(mod, a_, out):
                if state["bits"] is None: return out
                shp = out.shape
                return fake_quant_V(out.reshape(-1, shp[-1]), state["bits"], args.gv, args.residual).reshape(shp)
            return hook

        hk = []
        for l in range(heads.L):
            hk.append(heads.layers[l].self_attn.k_proj.register_forward_hook(mk_k()))
            hk.append(heads.layers[l].self_attn.v_proj.register_forward_hook(mk_v()))
        try:
            for spec in kivi_cfgs:
                b = int(spec); t0 = time.time(); state["bits"] = b
                eb = effective_bits(b, min(args.gk, args.gv)); cf = eb / 16.0
                tag = f"kivi_int{b}"
                g = run_grid(model, tok, hay_ids, contexts, depths, args.magic, args.gen_tokens, dev, sys_prompt, question, tag)
                results[tag] = {"grid": {str(c): g[c] for c in contexts}, "eff_bits": round(eb, 2),
                                "cache_frac": round(cf, 4), "total_x": round(1 / cf, 2),
                                "gk": args.gk, "gv": args.gv, "residual": args.residual}
                print(ascii_heatmap(g, contexts, depths, f"{tag}  ({1/cf:.1f}x, eff {eb:.2f}b)"), flush=True)
                if args.fig_prefix:
                    try_png(g, contexts, depths, f"{args.fig_prefix}_{tag}.png",
                            f"{args.model.split('/')[-1]} NIAH — KIVI int{b} ({1/cf:.1f}x)")
                print(f"[time] {tag} {time.time()-t0:.1f}s", flush=True)
        finally:
            state["bits"] = None
            for h in hk: h.remove()

    # ---- competing baselines (faithful reimplementations in kv_baselines.py) ----
    bspecs = [s for s in args.baselines.split(",") if s.strip()]
    if bspecs:
        restore()
        from .. import baselines as kvb
        QUANT = {"kivi", "turboquant", "polarquant"}
        for spec in bspecs:
            method, param = spec.split(":"); param = int(param); t0 = time.time()
            if method in QUANT:
                cache, eb = kvb.make_quant_cache(method, heads.hd, param, dev,
                                                 gk=args.gk, gv=args.gv, residual=args.residual)
                tag = f"{method}_b{param}"; cf = eb / 16.0

                def gen_fn(ids, _m=method, _p=param):
                    c, _ = kvb.make_quant_cache(_m, heads.hd, _p, dev,
                                                gk=args.gk, gv=args.gv, residual=args.residual)
                    o = model.generate(ids, max_new_tokens=args.gen_tokens, do_sample=False,
                                       pad_token_id=tok.eos_token_id, use_cache=True, past_key_values=c)
                    return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True)
                meta = {"eff_bits": round(eb, 3), "cache_frac": round(cf, 4), "total_x": round(1 / cf, 2)}
                title = f"{method} int{param} ({1/cf:.1f}x, eff {eb:.2f}b)"
            else:  # evict: snapkv / pyramidkv ; param = token budget
                def gen_fn(ids, _m=method, _p=param):
                    txt, _ = kvb.evict_generate(model, tok, heads, ids, args.gen_tokens, _m, _p)
                    return txt
                tag = f"{method}_bud{param}"
                meta = {"budget": param, "note": "token-eviction; compression=ctx/budget per cell"}
                title = f"{method} budget={param} (ctx/budget x)"
            g = run_grid(model, tok, hay_ids, contexts, depths, args.magic, args.gen_tokens, dev,
                         sys_prompt, question, tag, gen_fn=gen_fn)
            entry = {"grid": {str(c): g[c] for c in contexts}}; entry.update(meta)
            results[tag] = entry
            print(ascii_heatmap(g, contexts, depths, title), flush=True)
            if args.fig_prefix:
                try_png(g, contexts, depths, f"{args.fig_prefix}_{tag}.png",
                        f"{args.model.split('/')[-1]} NIAH — {title}")
            print(f"[time] {tag} {time.time()-t0:.1f}s", flush=True)

    out = dict(model=args.model, opts=vars(args), contexts=contexts, depths=depths, results=results)
    with open(args.out, "w") as f: json.dump(out, f, indent=2)
    print(f"\n[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
