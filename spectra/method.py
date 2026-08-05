"""SPECTRA method — co-designed latent transform + quantization. Three composable,
training-free, fully-absorbed (zero inference overhead) components on top of the
data-aware low-rank latent (see core.py):

  (1) --center        mean-center the latent: cache quant(c - mu); add back mu@W_UP as a
                      folded constant beta. V-side is provably lossless+free; K-side mu is
                      cheap. Makes the cached latent zero-mean -> tighter quant range.
  (2) --heavy-k K     keep the top-K (SVD-ordered) latent channels at HIGH precision
                      (heavy_bits), quantize the rest at `bits`. --hadamard rotates the
                      light block to equalize variance (rotation absorbed into W_DOWN/W_UP).
  (3) --qaware-iters  after quantizing, refit W_UP by G-free LS on the QUANTIZED latent
                      (W_UP = (E[c_q^T c_q])^-1 E[c_q^T K]) so the up-projection compensates
                      quantization noise.

Eval: wikitext-2 PPL + lm-eval (see scripts/run_ppl_lmeval.py).
"""
from __future__ import annotations
import argparse, json, time, math
import torch
import torch.nn as nn

from .core import (
    AttnHeads, collect_grams_and_mean, make_windows, perplexity,
    lowrank_gweighted_factors, fit_apply_mla_split,
    _accumulate_streaming_stats, _rrr_precompute, _rrr_apply,
)
from .quant import water_filling_bits, group_water_filling_quant
from transformers import AutoModelForCausalLM, AutoTokenizer


def fake_quant(c, bits, group):
    if bits is None or bits >= 16:
        return c
    *lead, r = c.shape
    g = group if (group and 0 < group <= r and r % group == 0) else r
    x = c.reshape(*lead, r // g, g)
    lo = x.amin(-1, keepdim=True); hi = x.amax(-1, keepdim=True)
    qmax = (1 << bits) - 1
    scale = (hi - lo).clamp_min(1e-8) / qmax
    q = torch.round((x - lo) / scale).clamp_(0, qmax)
    return (q * scale + lo).reshape(*lead, r).to(c.dtype)


def rand_orthogonal(m, device, dtype, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(m, m, generator=g)
    Q, _ = torch.linalg.qr(A)
    return Q.to(device=device, dtype=dtype)


class LatentProjV2(nn.Module):
    """out = quant(x @ W_DOWN - mu) @ W_UP + beta + bias.  Heavy block (first heavy_k dims)
    quantized at heavy_bits; light block at bits. Rotation already absorbed into W_DOWN/W_UP."""
    def __init__(self, W_DOWN, W_UP, mu, beta, bits, group, heavy_k, heavy_bits, bias=None):
        super().__init__()
        self.register_buffer("W_DOWN", W_DOWN.contiguous())
        self.register_buffer("W_UP", W_UP.contiguous())
        self.register_buffer("mu", mu.contiguous())
        self.register_buffer("beta", beta.contiguous())
        self.bits, self.group = bits, group
        self.heavy_k, self.heavy_bits = heavy_k, heavy_bits
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)

    def forward(self, x):
        c = x.to(self.W_DOWN.dtype) @ self.W_DOWN - self.mu
        k = self.heavy_k
        if k > 0:
            heavy = fake_quant(c[..., :k], self.heavy_bits, self.group)
            light = fake_quant(c[..., k:], self.bits, self.group)
            c = torch.cat([heavy, light], dim=-1)
        else:
            c = fake_quant(c, self.bits, self.group)
        out = c @ self.W_UP + self.beta
        if self.bias is not None:
            out = out + self.bias
        return out


def run_lm_eval(model, tok, tasks, limit, bs=8):
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=bs)
    res = evaluator.simple_evaluate(model=lm, tasks=tasks, num_fewshot=0,
                                    batch_size=bs, limit=limit, verbosity="ERROR")
    out = {}
    for t in tasks:
        if t in res["results"]:
            m = res["results"][t]
            acc = m.get("acc_norm,none", m.get("acc,none", m.get("acc_norm", m.get("acc"))))
            out[t] = round(float(acc), 4) if acc is not None else None
    return out


@torch.no_grad()
def build_factor(W, Gl, hmean, r, ridge, hadamard, heavy_k, dtype, dev):
    """Return (W_DOWN, W_UP, mu) for one side. Rotation (Hadamard/orthogonal) on the LIGHT
    block is absorbed: c_light' = c_light R, W_UP_light' = R^T W_UP_light (same product)."""
    WD, WU = lowrank_gweighted_factors(W, Gl, r, ridge)   # (d,r_eff),(r_eff,m)
    r_eff = WD.shape[1]                                    # actual rank (<= r, capped by W cols)
    if hadamard and heavy_k < r_eff:
        m = r_eff - heavy_k
        R = rand_orthogonal(m, WD.device, WD.dtype)
        WD = WD.clone(); WU = WU.clone()
        WD[:, heavy_k:] = WD[:, heavy_k:] @ R              # rotate latent light block
        WU[heavy_k:, :] = R.t() @ WU[heavy_k:, :]          # inverse-rotate up-proj (R^T=R^-1)
    mu = (hmean.to(WD.dtype) @ WD)                          # (r,)
    return WD.to(dtype), WU.to(dtype), mu.to(dtype)


@torch.no_grad()
def qaware_refit(model, heads, dev, calib_w, ridge):
    """One pass: per layer collect quantized latent c_q (= LatentProjV2 internal, pre-W_UP)
    and target K/V; refit W_UP = (E[c_q^T c_q]+λI)^-1 E[c_q^T target]."""
    L = heads.L
    kmods = [heads.layers[l].self_attn.k_proj for l in range(L)]
    vmods = [heads.layers[l].self_attn.v_proj for l in range(L)]
    # need c_q (post-quant, pre-W_UP) and target (= original K). We capture module input h,
    # recompute c_q via the module's front half, and target via original (saved) weights.
    accs = {"k": [None]*L, "v": [None]*L}

    def cq_of(mod, x):
        c = x.to(mod.W_DOWN.dtype) @ mod.W_DOWN - mod.mu
        k = mod.heavy_k
        if k > 0:
            c = torch.cat([fake_quant(c[..., :k], mod.heavy_bits, mod.group),
                           fake_quant(c[..., k:], mod.bits, mod.group)], -1)
        else:
            c = fake_quant(c, mod.bits, mod.group)
        return c

    cur = {"h": [None]*L}
    hk = []
    def mk(l):
        def pre(mod, args): cur["h"][l] = args[0].reshape(-1, heads.d)
        return pre
    for l in range(L):
        hk.append(heads.layers[l].self_attn.q_proj.register_forward_pre_hook(mk(l)))
    # accumulators
    SccK=[None]*L; SckK=[None]*L; SccV=[None]*L; SckV=[None]*L
    try:
        for ids in calib_w:
            model(ids.to(dev).unsqueeze(0), use_cache=False)
            for l in range(L):
                h = cur["h"][l].double()
                # target K,V from ORIGINAL weights stored on the modules
                # target for W_UP fit is K MINUS the folded constant beta (= mu@W_UP),
                # since the forward adds beta back: out = c_q@W_UP + beta.
                tgtK = (h @ kmods[l]._WK_orig.double()) - kmods[l].beta.double()
                tgtV = (h @ vmods[l]._WV_orig.double()) - vmods[l].beta.double()
                cqk = cq_of(kmods[l], cur["h"][l]).double()
                cqv = cq_of(vmods[l], cur["h"][l]).double()
                SccK[l] = (cqk.t()@cqk) if SccK[l] is None else SccK[l]+cqk.t()@cqk
                SckK[l] = (cqk.t()@tgtK) if SckK[l] is None else SckK[l]+cqk.t()@tgtK
                SccV[l] = (cqv.t()@cqv) if SccV[l] is None else SccV[l]+cqv.t()@cqv
                SckV[l] = (cqv.t()@tgtV) if SckV[l] is None else SckV[l]+cqv.t()@tgtV
                cur["h"][l] = None
    finally:
        for h in hk: h.remove()
    for l in range(L):
        for Scc, Sck, mod in ((SccK[l],SckK[l],kmods[l]),(SccV[l],SckV[l],vmods[l])):
            A = Scc; B = Sck
            d = torch.diagonal(A).mean().clamp_min(1e-9)
            A = A + ridge*d*torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
            WUP = torch.linalg.solve(A, B)
            mod.W_UP.copy_(WUP.to(mod.W_UP.dtype))


def cache_frac(r_k, r_v, bits, args, kvb):
    """Honest cache fraction vs fp16: heavy block at heavy_bits, light at `bits`, plus
    fp16 group scale+zero overhead. kvb = 2 * kv_dim (the fp16 KV baseline width)."""
    def bits_side(r):
        k = min(args.heavy_k, r)
        ng = (r // args.group if args.group > 0 and r % args.group == 0 else 1)
        return k * args.heavy_bits + (r - k) * bits + ng * 2 * 16
    return (bits_side(r_k) + bits_side(r_v)) / (kvb * 16)


_cache_frac = cache_frac  # backward-compat alias


class LatentProjWF(nn.Module):
    """Water-filling LatentProj with GROUP-level rate allocation on the proven cache-friendly
    per-token group quant. out = group_wf_quant(x@W_DOWN - mu; group_bits, group) @ W_UP + beta.
    Each channel-group (SVD energy-ordered) gets a water-filled bit-width; 0-bit groups are the
    low-rank tail. Per-token min/max -> composes with the streaming KV cache."""
    def __init__(self, W_DOWN, W_UP, mu, beta, group_bits, group_size, bias=None):
        super().__init__()
        self.register_buffer("W_DOWN", W_DOWN.contiguous())
        self.register_buffer("W_UP", W_UP.contiguous())
        self.register_buffer("mu", mu.contiguous())
        self.register_buffer("beta", beta.contiguous())
        self.register_buffer("group_bits", group_bits.to(torch.long).contiguous())
        self.group_size = int(group_size)
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)

    def forward(self, x):
        c = x.to(self.W_DOWN.dtype) @ self.W_DOWN - self.mu
        c = group_water_filling_quant(c, self.group_bits, self.group_size)
        out = c @ self.W_UP + self.beta
        if self.bias is not None:
            out = out + self.bias
        return out


def _wf_cache_frac(gbits_k, gbits_v, group_size, kvb):
    """Honest effective rate for group-level water-filling (per-token group quant): each group
    of `group_size` channels stores group_bits bits/elem + one fp16 scale+zero per group/token.
    Rate over the fp16 KV baseline (kvb = 2*kv_dim)."""
    def side(gb):
        return float((gb.double() * group_size).sum()) + gb.numel() * 2 * 16   # bits + scale/zero
    return (side(gbits_k) + side(gbits_v)) / (kvb * 16)


@torch.no_grad()
def build_spectra_wf(model, heads, G, hmean, WKo, WVo, WOo, kb, vb, calib_w, dev, dtype, B, avg_bits, args):
    """SPECTRA with rate-distortion (water-filling) per-channel bit allocation — the
    low-compression (3-5x) recipe. Same low-rank split + W_O healing as build_spectra, but the
    latent is quantized with bits allocated per channel by 0.5*log2(variance), variance taken
    in closed form from the calibration covariance Cov(h)=G-h̄h̄ᵀ (no extra calibration pass).
    `avg_bits` is the target average bit-rate (the config's `bits` field)."""
    r_k = max(1, int(round(B * args.kv_ratio))); r_v = B - r_k
    layers = list(range(heads.L)); orig_snap = heads.snapshot(layers)
    fit_apply_mla_split(heads, G, layers, r_k, r_v, args.mla_ridge, dev)
    tied_snap = heads.snapshot(layers)
    if args.wo_rank > 0:
        GYY, GYZ, n = _accumulate_streaming_stats(heads, orig_snap, tied_snap, calib_w, dev)
        pc = _rrr_precompute(heads, GYY, GYZ, n, args.wo_ridge); del GYY, GYZ
        _rrr_apply(heads, pc, args.wo_rank); del pc; torch.cuda.empty_cache()
    wo_saved = [l.self_attn.o_proj.weight.detach().clone() for l in heads.layers]
    heads.restore(orig_snap)
    for l in range(heads.L):
        heads.layers[l].self_attn.o_proj.weight.copy_(wo_saved[l])
    del orig_snap, tied_snap; torch.cuda.empty_cache()
    bmax = int(getattr(args, "wf_bmax", 8)); gsz = int(args.group)

    def group_bits(WD, cov):
        """per-group water-filled bit-widths from per-group mean variance (SVD energy order)."""
        var = torch.diagonal(WD.double().t() @ cov @ WD.double()).clamp_min(1e-12)  # (r_eff,)
        r_eff = var.numel(); ng = r_eff // gsz
        if ng == 0:
            return water_filling_bits(var.mean().view(1), avg_bits, 0, bmax)
        gvar = var[:ng * gsz].reshape(ng, gsz).mean(dim=1)            # (ng,) mean energy per group
        return water_filling_bits(gvar, avg_bits, 0, bmax)

    tot_k = tot_v = None
    for l in range(heads.L):
        Gl = G[l].double(); hm = hmean[l].double()
        cov = Gl - torch.outer(hm, hm)
        WK = WKo[l].t().double(); WV = WVo[l].t().double()
        # pure low-rank factors (no heavy/Hadamard; water-filling subsumes the heavy block)
        WDk, WUk, muk = build_factor(WK, Gl, hmean[l], r_k, args.mla_ridge, False, 0, dtype, dev)
        WDv, WUv, muv = build_factor(WV, Gl, hmean[l], r_v, args.mla_ridge, False, 0, dtype, dev)
        gbk = group_bits(WDk, cov).to(dev); gbv = group_bits(WDv, cov).to(dev)
        if not args.center:
            muk = torch.zeros_like(muk); muv = torch.zeros_like(muv)
        betak = (muk.double() @ WUk.double()).to(dtype)
        betav = (muv.double() @ WUv.double()).to(dtype)
        km = LatentProjWF(WDk, WUk, muk, betak, gbk, gsz, kb[l]).to(dev)
        vm = LatentProjWF(WDv, WUv, muv, betav, gbv, gsz, vb[l]).to(dev)
        heads.layers[l].self_attn.k_proj = km
        heads.layers[l].self_attn.v_proj = vm
        tot_k = gbk.double() if tot_k is None else tot_k + gbk.double()
        tot_v = gbv.double() if tot_v is None else tot_v + gbv.double()
    cf = _wf_cache_frac(tot_k / heads.L, tot_v / heads.L, gsz, 2 * heads.kv_dim)
    return r_k, r_v, cf


@torch.no_grad()
def build_spectra(model, heads, G, hmean, WKo, WVo, WOo, kb, vb, calib_w, dev, dtype, B, bits, args):
    """Apply one SPECTRA config in place: low-rank K/V split (data-aware G-weighted SVD) +
    W_O reduced-rank compensation + LatentProj (center / heavy-block / Hadamard / quant).
    With --water-filling, dispatches to build_spectra_wf (rate-distortion per-channel bits).
    Returns (r_k, r_v, cache_fraction). `build_prism`/`build_akvc` are backward-compat aliases."""
    if getattr(args, "water_filling", False):
        return build_spectra_wf(model, heads, G, hmean, WKo, WVo, WOo, kb, vb, calib_w, dev, dtype, B, bits, args)
    r_k = max(1, int(round(B * args.kv_ratio))); r_v = B - r_k
    layers = list(range(heads.L)); orig_snap = heads.snapshot(layers)
    fit_apply_mla_split(heads, G, layers, r_k, r_v, args.mla_ridge, dev)
    tied_snap = heads.snapshot(layers)
    if args.wo_rank > 0:
        GYY, GYZ, n = _accumulate_streaming_stats(heads, orig_snap, tied_snap, calib_w, dev)
        pc = _rrr_precompute(heads, GYY, GYZ, n, args.wo_ridge); del GYY, GYZ
        _rrr_apply(heads, pc, args.wo_rank); del pc; torch.cuda.empty_cache()
    wo_saved = [l.self_attn.o_proj.weight.detach().clone() for l in heads.layers]
    heads.restore(orig_snap)
    for l in range(heads.L):
        heads.layers[l].self_attn.o_proj.weight.copy_(wo_saved[l])
    del orig_snap, tied_snap; torch.cuda.empty_cache()
    for l in range(heads.L):
        Gl = G[l]; hm = hmean[l]
        WK = WKo[l].t().double(); WV = WVo[l].t().double()
        WDk, WUk, muk = build_factor(WK, Gl, hm, r_k, args.mla_ridge, args.hadamard, args.heavy_k, dtype, dev)
        WDv, WUv, muv = build_factor(WV, Gl, hm, r_v, args.mla_ridge, args.hadamard, args.heavy_k, dtype, dev)
        if not args.center:
            muk = torch.zeros_like(muk); muv = torch.zeros_like(muv)
        betak = (muk.double() @ WUk.double()).to(dtype)
        betav = (muv.double() @ WUv.double()).to(dtype)
        km = LatentProjV2(WDk, WUk, muk, betak, bits, args.group, args.heavy_k, args.heavy_bits, kb[l]).to(dev)
        vm = LatentProjV2(WDv, WUv, muv, betav, bits, args.group, args.heavy_k, args.heavy_bits, vb[l]).to(dev)
        km._WK_orig = WK.to(dtype); vm._WV_orig = WV.to(dtype)
        heads.layers[l].self_attn.k_proj = km
        heads.layers[l].self_attn.v_proj = vm
    cf = cache_frac(r_k, r_v, bits, args, 2 * heads.kv_dim)
    return r_k, r_v, cf


build_prism = build_spectra   # backward-compat alias (earlier working name)
build_akvc = build_spectra    # backward-compat alias (original working name)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--dtype", default="float16")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--calib-windows", type=int, default=64); ap.add_argument("--eval-windows", type=int, default=40)
    ap.add_argument("--kv-ratio", type=float, default=0.25); ap.add_argument("--wo-rank", type=int, default=4)
    ap.add_argument("--mla-ridge", type=float, default=1e-3); ap.add_argument("--wo-ridge", type=float, default=1e-2)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--center", action="store_true")
    ap.add_argument("--heavy-k", type=int, default=0); ap.add_argument("--heavy-bits", type=int, default=8)
    ap.add_argument("--hadamard", action="store_true")
    ap.add_argument("--water-filling", action="store_true",
                    help="rate-distortion per-channel bit allocation (low-compression 3-5x recipe); "
                         "config 'B:bits' then means latent width B and TARGET AVERAGE bits")
    ap.add_argument("--wf-bmax", type=int, default=8)
    ap.add_argument("--wf-clip", type=float, default=4.0)
    ap.add_argument("--wf-residual", type=int, default=0)
    ap.add_argument("--qaware-iters", type=int, default=0)
    ap.add_argument("--qaware-ridge", type=float, default=1e-4)
    ap.add_argument("--configs", default="", help="csv B:bits (empty = base/fp16 PPL only)")
    ap.add_argument("--lm-eval", action="store_true")
    ap.add_argument("--lm-eval-tasks", default="arc_easy,arc_challenge,hellaswag,piqa,winogrande,openbookqa")
    ap.add_argument("--lm-eval-limit", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    dev = args.device; dtype = getattr(torch, args.dtype)
    print(f"[load] {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(dev).eval()
    heads = AttnHeads(model); kvb = 2*heads.kv_dim
    print(f"[cfg] arch={heads.arch} L={heads.L} kv_dim={heads.kv_dim}", flush=True)

    from datasets import load_dataset
    ds_tr = load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="train")
    ds_te = load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
    calib_w = make_windows(tok, "\n\n".join(t for t in ds_tr["text"] if t.strip()), args.seq_len, args.calib_windows)
    eval_w  = make_windows(tok, "\n\n".join(t for t in ds_te["text"] if t.strip()), args.seq_len, args.eval_windows)
    print(f"[data] calib={len(calib_w)} eval={len(eval_w)}", flush=True)
    G, hmean = collect_grams_and_mean(model, heads, calib_w, dev)

    lm_tasks = [t for t in args.lm_eval_tasks.split(",") if t.strip()]
    results = {}
    ppl_b = perplexity(model, eval_w, dev)
    results["base"] = {"ppl": round(ppl_b,4), "cache_frac":1.0}
    print(f"[ppl] base = {ppl_b:.4f}", flush=True)
    if args.lm_eval:
        results["base"]["acc"] = run_lm_eval(model, tok, lm_tasks, args.lm_eval_limit)
        print(f"[lm-eval] base {results['base']['acc']}", flush=True)

    # save original K/V weights + modules to restore between configs
    orig_k = [l.self_attn.k_proj for l in heads.layers]
    orig_v = [l.self_attn.v_proj for l in heads.layers]
    WKo = [k.weight.detach().clone() for k in orig_k]
    WVo = [v.weight.detach().clone() for v in orig_v]
    WOo = [l.self_attn.o_proj.weight.detach().clone() for l in heads.layers]
    kb = [(k.bias.detach().clone() if k.bias is not None else None) for k in orig_k]
    vb = [(v.bias.detach().clone() if v.bias is not None else None) for v in orig_v]

    def restore():
        for l in range(heads.L):
            heads.layers[l].self_attn.k_proj = orig_k[l]
            heads.layers[l].self_attn.v_proj = orig_v[l]
            orig_k[l].weight.copy_(WKo[l]); orig_v[l].weight.copy_(WVo[l])
            heads.layers[l].self_attn.o_proj.weight.copy_(WOo[l])

    for spec in args.configs.split(","):
        if not spec.strip(): continue
        B, bits = (int(x) for x in spec.split(":"))
        tb = time.time(); restore()
        r_k = max(1, int(round(B*args.kv_ratio))); r_v = B - r_k
        if args.water_filling:
            # real rate-distortion water-filling (same path as build_spectra / the API / LongBench)
            r_k, r_v, cf = build_spectra(model, heads, G, hmean, WKo, WVo, WOo, kb, vb,
                                         calib_w, dev, dtype, B, bits, args)
            ppl = perplexity(model, eval_w, dev)
            tag = f"B{B}_b{bits}_c{int(args.center)}_wf_qa{args.qaware_iters}"
            entry = {"ppl": round(ppl,4), "cache_frac": round(cf,4), "total_x": round(1/cf,2),
                     "r_k": r_k, "r_v": r_v}
            print(f"[ppl] {tag} cache={cf:.4f}x ({entry['total_x']}x) = {ppl:.4f} ({time.time()-tb:.1f}s)", flush=True)
            if args.lm_eval:
                entry["acc"] = run_lm_eval(model, tok, lm_tasks, args.lm_eval_limit)
            results[tag] = entry
            continue
        # Step A: low-rank + WO compensation (un-quantized) -> sets o_proj.
        layers = list(range(heads.L)); orig_snap = heads.snapshot(layers)
        fit_apply_mla_split(heads, G, layers, r_k, r_v, args.mla_ridge, dev)
        tied_snap = heads.snapshot(layers)
        if args.wo_rank>0:
            GYY,GYZ,n = _accumulate_streaming_stats(heads, orig_snap, tied_snap, calib_w, dev)
            pc = _rrr_precompute(heads, GYY, GYZ, n, args.wo_ridge); del GYY,GYZ
            _rrr_apply(heads, pc, args.wo_rank); del pc; torch.cuda.empty_cache()
        wo_saved=[l.self_attn.o_proj.weight.detach().clone() for l in heads.layers]
        heads.restore(orig_snap)
        for l in range(heads.L): heads.layers[l].self_attn.o_proj.weight.copy_(wo_saved[l])
        del orig_snap, tied_snap; torch.cuda.empty_cache()
        # Step B: build LatentProjV2 (center/heavy/hadamard) per layer
        for l in range(heads.L):
            Gl = G[l]; hm = hmean[l]
            WK = WKo[l].t().double(); WV = WVo[l].t().double()
            WDk,WUk,muk = build_factor(WK, Gl, hm, r_k, args.mla_ridge, args.hadamard, args.heavy_k, dtype, dev)
            WDv,WUv,muv = build_factor(WV, Gl, hm, r_v, args.mla_ridge, args.hadamard, args.heavy_k, dtype, dev)
            if not args.center:
                muk = torch.zeros_like(muk); muv = torch.zeros_like(muv)
            betak = (muk.double() @ WUk.double()).to(dtype)
            betav = (muv.double() @ WUv.double()).to(dtype)
            km = LatentProjV2(WDk,WUk,muk,betak,bits,args.group,args.heavy_k,args.heavy_bits, kb[l]).to(dev)
            vm = LatentProjV2(WDv,WUv,muv,betav,bits,args.group,args.heavy_k,args.heavy_bits, vb[l]).to(dev)
            km._WK_orig = WK.to(dtype); vm._WV_orig = WV.to(dtype)   # for qaware target
            heads.layers[l].self_attn.k_proj = km
            heads.layers[l].self_attn.v_proj = vm
        # Step C: quant-aware W_UP refit
        for _ in range(args.qaware_iters):
            qaware_refit(model, heads, dev, calib_w, args.qaware_ridge)
        ppl = perplexity(model, eval_w, dev)
        # cache accounting
        def bits_side(r):
            k = min(args.heavy_k, r)
            ng = (r//args.group if args.group>0 and r%args.group==0 else 1)
            return k*args.heavy_bits + (r-k)*bits + ng*2*16
        cache_bits = bits_side(r_k)+bits_side(r_v)
        cf = cache_bits/(kvb*16)
        tag=f"B{B}_b{bits}_c{int(args.center)}_hk{args.heavy_k}_hd{int(args.hadamard)}_qa{args.qaware_iters}"
        entry={"ppl":round(ppl,4),"cache_frac":round(cf,4),"total_x":round(1/cf,2),"r_k":r_k,"r_v":r_v}
        print(f"[ppl] {tag} cache={cf:.4f}x ({entry['total_x']}x) = {ppl:.4f} ({time.time()-tb:.1f}s)", flush=True)
        if args.lm_eval:
            entry["acc"]=run_lm_eval(model, tok, lm_tasks, args.lm_eval_limit)
            print(f"[lm-eval] {tag} {entry['acc']}", flush=True)
        results[tag]=entry

    restore()
    out=dict(model=args.model, opts=vars(args), results=results)
    with open(args.out,"w") as f: json.dump(out,f,indent=2)
    print("\n=== AKVC-v2 SUMMARY ===", flush=True)
    for k,v in results.items():
        accs=v.get("acc",{}); a=" ".join(f"{t.split('_')[0]}:{x}" for t,x in accs.items()) if accs else ""
        print(f"{k:40s} {v['cache_frac']:7.4f} {v.get('total_x',1):6.1f} {v['ppl']:9.3f}  {a}", flush=True)
    print(f"[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
