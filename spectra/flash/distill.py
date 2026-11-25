"""Block-wise local distillation of the record maps (base model frozen).

Long-key, short-query: each step takes one long calibration window and a random subset of query positions; every
query attends all keys of the window, so memory is heads x Q x S rather than heads x S x S and windows can be as
long as the serving context. The student mirrors the decode path exactly: query t attends
    [exact sink 0..sink) | record [sink, t - recent) | exact band [t - recent, t]
with one softmax. Loss: relative MSE of the attention output after o_proj, teacher-forced on the original model's
attention inputs (captured once per block).
"""
from __future__ import annotations

import math
import random
import time

import torch
import torch.nn.functional as F

from .closed_form import closed_form_block
from .ops import F8_MAX, LayerW, cos_sin_table, fp8_round, repeat_kv, rope_full, rope_pairs


def exact_kv(w: LayerW, x, cos, sin):
    """Exact RoPE'd keys and values for every position of x (S, d) fp32 -> (kvh, S, hd) each."""
    S = x.shape[0]
    k = rope_full((x @ w.wk).view(S, w.kvh, w.hd).transpose(0, 1), cos[:S], sin[:S])
    v = (x @ w.wv).view(S, w.kvh, w.hd).transpose(0, 1)
    return k, v


def teacher_out(w: LayerW, x, qpos, cos, sin, kv=None):
    """Exact attention output before o_proj for query positions qpos (Q,) over keys <= t -> (Q, nq hd)."""
    S = x.shape[0]; Q = qpos.numel()
    k, v = kv if kv is not None else exact_kv(w, x, cos, sin)
    q = rope_full((x[qpos] @ w.wq).view(Q, w.nq, w.hd).transpose(0, 1), cos[qpos], sin[qpos])
    mask = torch.arange(S, device=x.device)[None, :] <= qpos[:, None]
    o = F.scaled_dot_product_attention(q[None], repeat_kv(k, w.gq)[None], repeat_kv(v, w.gq)[None],
                                       attn_mask=mask[None, None], scale=w.scaling)[0]
    return o.transpose(0, 1).reshape(Q, -1)


def student_out(prm, ws, xs, qpos, cos, sin, sink, recent, fp8_scales=None, kvs=None):
    """Deployment-shaped attention for the block's n layers. prm: record maps; ws: LayerW per layer; xs: inputs
    (S, d) fp32 per layer; qpos (Q,) with every t >= sink + recent + 1. Returns n outputs (Q, nq hd) before o_proj."""
    S, d = xs[0].shape; Q = qpos.numel(); dev = xs[0].device
    H = torch.cat(xs, -1)
    c = H @ prm["Wd"]; s = H @ prm["Wr"]
    if fp8_scales is not None:
        c = fp8_round(c, fp8_scales[0]); s = fp8_round(s, fp8_scales[1])
    pos = torch.arange(S, device=dev)
    srot = rope_pairs(s, pos, prm["theta"])
    band = qpos[:, None] - torch.arange(recent, -1, -1, device=dev)[None, :]
    eidx = torch.cat([torch.arange(sink, device=dev)[None].expand(Q, -1), band], 1)
    E = eidx.shape[1]
    rec_bad = (pos[None, :] < sink) | (pos[None, :] >= (qpos[:, None] - recent))
    outs = []
    for i, w in enumerate(ws):
        R = c.shape[-1]; r = s.shape[-1]
        x = xs[i]; xq = x[qpos]
        k, v = kvs[i] if kvs is not None else exact_kv(w, x, cos, sin)
        q = (xq @ w.wq).view(Q, w.kvh, w.gq, w.hd)
        qf = rope_full(q.reshape(Q, w.nq, w.hd).transpose(0, 1), cos[qpos], sin[qpos]).reshape(w.kvh, w.gq, Q, w.hd)
        kg = k[:, eidx]; vg = v[:, eidx]
        sc_e = torch.einsum("gjqd,gqed->gjqe", qf, kg).reshape(w.nq, Q, E)
        qa = torch.einsum("qgjd,rgd->gjqr", q, prm["WuK"][i].view(R, w.kvh, w.hd)).reshape(w.nq, Q, R)
        qr = rope_pairs((xq @ prm["WQR"][i]).view(Q, w.nq, r).transpose(0, 1), qpos, prm["theta"])
        sc_r = (qa @ c.T + qr @ srot.T).masked_fill_(rec_bad[None], float("-inf"))
        P = torch.softmax(torch.cat([sc_e, sc_r], -1) * w.scaling, -1)
        o_e = torch.einsum("gjqe,gqed->gjqd", P[..., :E].reshape(w.kvh, w.gq, Q, E), vg).reshape(w.nq, Q, w.hd)
        ol = P[..., E:] @ c
        o_r = torch.einsum("gjqr,rgd->gjqd", ol.view(w.kvh, w.gq, Q, R),
                           prm["WuV"][i].view(R, w.kvh, w.hd)).reshape(w.nq, Q, w.hd)
        outs.append((o_e + o_r).transpose(0, 1).reshape(Q, w.nq * w.hd))
    return outs


def _buckets(S, lo):
    edges = [lo, 4096, 16384, S]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]


@torch.no_grad()
def heldout_error(prm, ws, X, held, cos, sin, cfg, fp8_scales=None, chunk=256, gen_start=None):
    """Post-o_proj relative MSE per layer on the held-out windows, overall and by query-position bucket. Windows with
    a generated span (gen_start >= 0) add 256 queries inside it and a "generated" bucket (queries in the span)."""
    S = X[0].shape[1]; lo = cfg.sink + cfg.recent + 1; dev = X[0].device
    B = _buckets(S, lo)
    names = [f"{a}-{z}" for a, z in B] + ["generated"]
    num = torch.zeros(len(ws), len(names) + 1); den = torch.zeros_like(num)
    for wi in held:
        gs = gen_start[wi] if gen_start else -1
        qe = torch.linspace(lo, S - 1, cfg.heldout_queries, device=dev).long()
        if gs >= 0:
            qe = torch.cat([qe, torch.linspace(gs, S - 1, 256, device=dev).long()]).unique()
        xs = [x[wi].float() for x in X]
        kvs = [exact_kv(w, xs[i], cos, sin) for i, w in enumerate(ws)]
        for j in range(0, qe.numel(), chunk):
            qp = qe[j:j + chunk]
            outs = student_out(prm, ws, xs, qp, cos, sin, cfg.sink, cfg.recent, fp8_scales, kvs)
            qc = qp.cpu()
            masks = [(qc >= a) & (qc < z) for a, z in B] + [(qc >= gs) if gs >= 0 else torch.zeros_like(qc, dtype=torch.bool)]
            for i, w in enumerate(ws):
                yr = teacher_out(w, xs[i], qp, cos, sin, kvs[i]) @ w.wo
                e = ((outs[i] @ w.wo - yr) ** 2).sum(-1).cpu(); y2 = (yr ** 2).sum(-1).cpu()
                num[i, -1] += e.sum(); den[i, -1] += y2.sum()
                for b, m in enumerate(masks):
                    num[i, b] += e[m].sum(); den[i, b] += y2[m].sum()
    rel = num / den.clamp_min(1e-12)
    return dict(all=rel[:, -1].tolist(),
                buckets={nm: rel[:, b].tolist() for b, nm in enumerate(names) if den[:, b].sum() > 0})


@torch.no_grad()
def record_scales(prm, X, train, cfg):
    """Static fp8 scales of the record: per latent channel, and per RoPE pair (both channels of a pair share one
    scale, so the scale commutes with the rotation). headroom x max over calibration windows, positions >= sink."""
    R = prm["Wd"].shape[1]; r = prm["Wr"].shape[1]; dev = X[0].device
    amc = torch.zeros(R, device=dev); ams = torch.zeros(r, device=dev)
    for wi in train[:2] + train[-2:]:
        H = torch.cat([x[wi] for x in X], -1)[cfg.sink:].float()
        amc = torch.maximum(amc, (H @ prm["Wd"]).abs().amax(0))
        ams = torch.maximum(ams, (H @ prm["Wr"]).abs().amax(0))
    cs = (amc * cfg.record_headroom / F8_MAX).clamp_min(1e-8)
    ss = (ams.view(-1, 2).amax(-1, keepdim=True).expand(-1, 2).reshape(-1) * cfg.record_headroom / F8_MAX).clamp_min(1e-8)
    return cs, ss


@torch.no_grad()
def query_scales(prm, ws, X, train, cs, ss, cfg, chunk=4096):
    """Static per-layer fp8 scale of the FlashMLA query [W_UK q * cs | RoPE(W_QR x) * ss] (the form the deployed
    kernel receives): headroom x max over two calibration windows (one text, one code), all positions."""
    R = prm["Wd"].shape[1]; r = prm["Wr"].shape[1]; out = []
    for i, w in enumerate(ws):
        amax = torch.zeros((), device=X[0].device)
        for wi in (train[0], train[-1]):
            x = X[i][wi].float()
            for a in range(0, x.shape[0], chunk):
                xc = x[a:a + chunk]; pos = torch.arange(a, a + xc.shape[0], device=x.device)
                q = (xc @ w.wq).view(-1, w.kvh, w.gq, w.hd)
                qa = torch.einsum("sgjd,rgd->sgjr", q, prm["WuK"][i].view(R, w.kvh, w.hd)) * cs
                qr = rope_pairs((xc @ prm["WQR"][i]).view(-1, w.nq, r).transpose(0, 1), pos, prm["theta"]) * ss
                amax = torch.maximum(amax, torch.maximum(qa.abs().max(), qr.abs().max()))
        out.append(float(amax) * cfg.query_headroom / F8_MAX)
    return out


def convert_block(model, grp, X, train, held, cfg, log=print, gen_start=None):
    """Closed-form init + distillation for one block. X: captured inputs per layer of the block, (N, S, d) bf16 on
    the GPU. Returns a dict of fp32 CPU tensors and the report."""
    t0 = time.time()
    layers = model.model.layers; d = model.config.hidden_size; n = len(grp); dev = X[0].device
    S = X[0].shape[1]; lo = cfg.sink + cfg.recent + 1
    seed = cfg.seed + grp[0]
    random.seed(seed); torch.manual_seed(seed)
    Gj = torch.zeros(n * d, n * d, device=dev)
    for wi in train:
        Hh = torch.cat([x[wi] for x in X], -1).float()
        Gj += Hh.T @ Hh
        del Hh
    Gj /= len(train) * S
    Gl = [Gj[i * d:(i + 1) * d, i * d:(i + 1) * d] for i in range(n)]
    init = closed_form_block(model, grp, Gj, Gl, cfg.r_latent, cfg.r_rope, cfg.query_weighted, cfg.rope_horizon)
    del Gj, Gl
    torch.cuda.empty_cache()
    ws = [LayerW(layers[l].self_attn) for l in grp]
    cos, sin = cos_sin_table(model.model.rotary_emb.inv_freq.float().to(dev), S, dev)
    he = lambda prm, sc=None: heldout_error(prm, ws, X, held, cos, sin, cfg, sc, gen_start=gen_start)
    err = {"init": he(init)}
    P = dict(theta=init["theta"],
             Wd=torch.nn.Parameter(init["Wd"].clone()), Wr=torch.nn.Parameter(init["Wr"].clone()),
             WuK=[torch.nn.Parameter(t.clone()) for t in init["WuK"]],
             WuV=[torch.nn.Parameter(t.clone()) for t in init["WuV"]],
             WQR=[torch.nn.Parameter(t.clone()) for t in init["WQR"]])
    params = [P["Wd"], P["Wr"]] + P["WuK"] + P["WuV"] + P["WQR"]
    opt = torch.optim.Adam(params, lr=cfg.lr)
    warm = max(1, cfg.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: min(1.0, (st + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, st / max(1, cfg.steps)))))
    curve = []
    for step in range(cfg.steps):
        wi = random.choice(train)
        xs = [x[wi].float() for x in X]
        gs = gen_start[wi] if gen_start else -1
        if gs >= 0 and cfg.decode_query_frac > 0:            # decode-phase data: part of the queries in the generated span
            k = int(cfg.queries * cfg.decode_query_frac)
            qp = torch.cat([torch.randint(max(lo, gs), S, (k,), device=dev),
                            torch.randint(lo, S, (cfg.queries - k,), device=dev)]).sort().values
        else:
            qp = torch.randint(lo, S, (cfg.queries,), device=dev).sort().values
        with torch.no_grad():
            kvs = [exact_kv(w, xs[i], cos, sin) for i, w in enumerate(ws)]
            Y = [teacher_out(w, xs[i], qp, cos, sin, kvs[i]) @ w.wo for i, w in enumerate(ws)]
        outs = student_out(P, ws, xs, qp, cos, sin, cfg.sink, cfg.recent, None, kvs)
        loss = sum(((outs[i] @ w.wo - Y[i]) ** 2).sum() / (Y[i] ** 2).sum() for i, w in enumerate(ws)) / n
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        curve.append(loss.item())
        del xs, kvs, Y, outs, loss
        if step % 250 == 0 or step == cfg.steps - 1:
            log(f"[spectra-flash] block {grp[0]} step {step}/{cfg.steps} loss {sum(curve[-50:]) / len(curve[-50:]):.4f}")
    trained = dict(theta=P["theta"], Wd=P["Wd"].detach(), Wr=P["Wr"].detach(),
                   WuK=[t.detach() for t in P["WuK"]], WuV=[t.detach() for t in P["WuV"]],
                   WQR=[t.detach() for t in P["WQR"]])
    del P, params, opt
    torch.cuda.empty_cache()
    err["trained"] = he(trained)
    cs, ss = record_scales(trained, X, train, cfg)
    err["trained_fp8"] = he(trained, (cs, ss))
    qs = query_scales(trained, ws, X, train, cs, ss, cfg)
    cpu = lambda v: v.cpu() if torch.is_tensor(v) else [t.cpu() for t in v]
    rec = dict(g0=grp[0], n=n, params={k: cpu(v) for k, v in trained.items()}, c_scale=cs.cpu(), s_scale=ss.cpu(),
               q_scale=qs, err=err, info=init["info"], curve=curve, seconds=time.time() - t0,
               gpu=torch.cuda.get_device_name(dev))
    log(f"[spectra-flash] block {grp[0]} done in {rec['seconds']:.0f}s | held-out relMSE init "
        f"{[round(x, 3) for x in err['init']['all']]} -> trained {[round(x, 3) for x in err['trained']['all']]} "
        f"(fp8 record {[round(x, 3) for x in err['trained_fp8']['all']]})")
    return rec
