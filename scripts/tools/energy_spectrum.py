#!/usr/bin/env python3
"""Energy-concentration evidence figure for SPECTRA Section 3.1.

For each layer we measure how much of the energy of the keys/values the model *actually
produces* (E||hW||^2 = ||G^{1/2}W||_F^2) is retained by a rank-r factorization, under two bases:

  - G-SVD (ours):   keep the top-r directions of  M = G^{1/2} W   (optimal in produced energy)
                    retained(r) = sum_{j<r} s_j^2 / sum_j s_j^2 ,  M = U S V^T
  - plain SVD (Palu): keep the top-r directions of W itself (ordered by W's own singular values),
                    but scored in the SAME produced-energy metric:
                    retained(r) = sum_{j<r} sw_j^2 * (uw_j^T G uw_j) / sum_j s_j^2

The gap between the two curves is exactly "why G-weighting matters" (and previews SPECTRA > Palu).

Usage:
  python scripts/tools/energy_spectrum.py --model NousResearch/Meta-Llama-3.1-8B-Instruct \
      --calib-windows 24 --seq-len 2048 --out runs/energy_l31
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from spectra.core import AttnHeads, collect_grams_and_mean, make_windows, _sym_sqrt_inv_sqrt, _robust_svd


@torch.no_grad()
def side_curves(W, G, ridge):
    """W: (d, m) row-conv weight (d_in x m). Returns (cum_gsvd, cum_plain) cumulative produced-energy
    fractions over rank r=1..m, plus total produced energy."""
    Csqrt, _ = _sym_sqrt_inv_sqrt(G, ridge)
    M = Csqrt @ W                                   # whitened weight, (d, m)
    U, S, Vh = _robust_svd(M, full_matrices=False)
    s2 = (S.double() ** 2)
    total = s2.sum().clamp_min(1e-30)
    cum_gsvd = torch.cumsum(s2, 0) / total          # optimal (G-SVD)
    # plain SVD of W, scored in produced-energy metric
    Uw, Sw, Vwh = _robust_svd(W, full_matrices=False)
    # per-direction produced energy = sw_j^2 * (uw_j^T G uw_j)
    GU = G @ Uw                                      # (d, k)
    uGu = (Uw * GU).sum(0).double()                 # (k,)  diag(Uw^T G Uw)
    contrib = (Sw.double() ** 2) * uGu              # ordered by plain-SVD order already
    cum_plain = torch.cumsum(contrib, 0) / total
    return cum_gsvd.cpu(), cum_plain.cpu(), float(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--calib-windows", type=int, default=24)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--out", default="runs/energy")
    args = ap.parse_args()

    dev = args.device
    torch.set_grad_enabled(False)
    print(f"[load] {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).to(dev).eval()
    heads = AttnHeads(model)
    print(f"[cfg] arch={heads.arch} L={heads.L} kv_dim={heads.kv_dim}", flush=True)

    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    calib = make_windows(tok, text, args.seq_len, args.calib_windows)
    print(f"[calib] {len(calib)} windows", flush=True)
    G, _ = collect_grams_and_mean(model, heads, calib, dev)

    m = heads.kv_dim
    gk = torch.zeros(m, dtype=torch.float64); pk = torch.zeros(m, dtype=torch.float64)
    gv = torch.zeros(m, dtype=torch.float64); pv = torch.zeros(m, dtype=torch.float64)
    gk_all, pk_all = [], []
    for l in range(heads.L):
        Gl = G[l].double()
        WK = heads.layers[l].self_attn.k_proj.weight.t().double()   # (d, kv)
        WV = heads.layers[l].self_attn.v_proj.weight.t().double()
        cg_k, cp_k, _ = side_curves(WK, Gl, args.ridge)
        cg_v, cp_v, _ = side_curves(WV, Gl, args.ridge)
        gk += cg_k; pk += cp_k; gv += cg_v; pv += cp_v
        gk_all.append(cg_k); pk_all.append(cp_k)
        print(f"  L{l:02d} K: G-SVD@50%={cg_k[m//2]:.3f} plain@50%={cp_k[m//2]:.3f}", flush=True)
    gk/=heads.L; pk/=heads.L; gv/=heads.L; pv/=heads.L

    x = [(i+1)/m for i in range(m)]
    def rank_at(cum, thr):
        for i,v in enumerate(cum):
            if v>=thr: return (i+1)/m
        return 1.0
    r95k = rank_at(gk, 0.95); r99k = rank_at(gk, 0.99)
    plain_at_r95 = float(pk[int(r95k*m)-1])
    summary = dict(model=args.model, kv_dim=m, L=heads.L,
        K_gsvd_rank_at_95=r95k, K_gsvd_rank_at_99=r99k,
        K_plain_energy_at_that_rank=plain_at_r95,
        K_gsvd_energy_at_50pct=float(gk[m//2]), K_plain_energy_at_50pct=float(pk[m//2]),
        V_gsvd_energy_at_50pct=float(gv[m//2]), V_plain_energy_at_50pct=float(pv[m//2]))
    print("[summary]", json.dumps(summary, indent=2), flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(dict(summary=summary, x=x,
                   K_gsvd=gk.tolist(), K_plain=pk.tolist(),
                   V_gsvd=gv.tolist(), V_plain=pv.tolist()),
              open(args.out+".json","w"))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        gk_stack = torch.stack(gk_all); pk_stack = torch.stack(pk_all)
        fig, ax = plt.subplots(figsize=(5.4,3.8))
        ax.plot(x, gk.tolist(), color="#d62728", lw=2.4, label="G-SVD (ours), Keys", zorder=3)
        ax.fill_between(x, gk_stack.quantile(0.1,0).tolist(), gk_stack.quantile(0.9,0).tolist(),
                        color="#d62728", alpha=0.15, zorder=1)
        ax.plot(x, pk.tolist(), color="#9467bd", lw=2.0, ls="--", label="plain SVD (Palu), Keys", zorder=2)
        ax.fill_between(x, pk_stack.quantile(0.1,0).tolist(), pk_stack.quantile(0.9,0).tolist(),
                        color="#9467bd", alpha=0.12, zorder=1)
        ax.axhline(0.95, color="gray", lw=0.8, ls=":")
        ax.axvline(r95k, color="#d62728", lw=0.8, ls=":")
        ax.set_xlabel("retained rank fraction  r / d_kv")
        ax.set_ylabel("produced-key energy retained")
        ax.set_xlim(0,1); ax.set_ylim(0,1.02)
        ax.set_title(args.model.split("/")[-1])
        ax.legend(fontsize=8, loc="lower right", frameon=False)
        ax.grid(alpha=0.25)
        ax.annotate(f"G-SVD: 95% energy @ {r95k*100:.0f}% rank\nplain SVD there: {plain_at_r95*100:.0f}%",
                    xy=(r95k,0.95), xytext=(0.30,0.55), fontsize=8,
                    arrowprops=dict(arrowstyle="->", color="gray"))
        fig.tight_layout(); fig.savefig(args.out+".png", dpi=140)
        print(f"[fig] {args.out}.png", flush=True)
    except Exception as e:
        print("[fig] skipped:", e, flush=True)


if __name__ == "__main__":
    main()
