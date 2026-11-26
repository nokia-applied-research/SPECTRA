"""Deployment tensors of a SPECTRA-Flash params dir, folded exactly as the reference engine folds them.

The params file holds fp32 masters. Folding (fp32, then one cast to bf16):
  Wdr16 = [W_d | W_r]^T                          (R + r, n d)   record entry of one token = H_block @ Wdr^T
  Wdr8  = ([W_d | W_r] / [c_scale | s_scale])^T  (R + r, n d)   the same in fp8 record units
  WuK16 = W_UK as (HKV, D, R),  WuK8 = WuK16 * c_scale          absorbed query  q_g @ WuK
  WuV16 = W_UV as (HKV, R, D),  WuV8 = WuV16 * c_scale          value up-projection after the record attention
  WQR   = W_QR^T                                 (H r, d)       rope query (adjacent RoPE pairs, angles theta)
Each layer's qkv GEMM is extended with its rope query rows and its half of Wdr, so one GEMM yields
q | k | v | rope query | partial record entry (the block's entry = sum of its layers' partials).
"""
from __future__ import annotations

import os

import torch

from . import params as P

BF = torch.bfloat16


def resolve(path_or_repo: str) -> str:
    """A local params dir, or a HF Hub repo id (downloads params.safetensors + flash_config.json)."""
    if os.path.isdir(path_or_repo):
        return path_or_repo
    if path_or_repo.startswith(("/", ".", "~")) or path_or_repo.count("/") > 1:
        raise FileNotFoundError(f"SPECTRA-Flash params dir not found: {path_or_repo!r} "
                                "(expected a local dir or a HF Hub repo id 'namespace/name')")
    from huggingface_hub import snapshot_download
    return snapshot_download(path_or_repo, allow_patterns=["params.safetensors", "flash_config.json", "README.md"])


def deploy_settings(meta: dict, recent: int | None = None) -> dict:
    """sink / recent / fp8 of the deployment. `deploy` in flash_config.json overrides the conversion config
    (Lite was trained with recent 64 and is deployed with 123); an explicit `recent` overrides both."""
    cfg, dep = meta["config"], meta.get("deploy", {})
    return dict(sink=dep.get("sink", cfg["sink"]), recent=recent or dep.get("recent", cfg["recent"]),
                fp8=dep.get("fp8", True))


def fold_block(bp: dict, n: int, hkv: int, hd: int, dev) -> dict:
    R, r = bp["Wd"].shape[1], bp["Wr"].shape[1]
    cs, ss = bp["cs"].to(dev).float(), bp["ss"].to(dev).float()
    Wdr = torch.cat([bp["Wd"], bp["Wr"]], 1).to(dev).float()                   # (n d, R + r)
    blk = dict(R=R, r=r, theta=bp["theta"].to(dev).float().contiguous(), cs=cs, ss=ss.contiguous(),
               Wdr16=Wdr.t().contiguous().to(BF),
               Wdr8=(Wdr / torch.cat([cs, ss])[None]).t().contiguous().to(BF),
               WuK16=[], WuK8=[], WuV16=[], WuV8=[], WQR=[])
    for li in range(n):
        wk = bp["WuK"][li].to(dev).float().view(R, hkv, hd).permute(1, 2, 0)       # (HKV, D, R)
        wv = bp["WuV"][li].to(dev).float().view(R, hkv, hd).permute(1, 0, 2)       # (HKV, R, D)
        blk["WuK16"].append(wk.contiguous().to(BF)); blk["WuK8"].append((wk * cs).contiguous().to(BF))
        blk["WuV16"].append(wv.contiguous().to(BF)); blk["WuV8"].append((wv * cs[None, :, None]).contiguous().to(BF))
        blk["WQR"].append(bp["WQR"][li].to(dev, BF).t().contiguous())             # (H r, d)
    return blk


def load(path_or_repo: str, dev, hkv: int, hd: int):
    """-> (n, {g0: folded block}, {layer: q_scale}, meta, local_dir)."""
    d = resolve(path_or_repo)
    p = P.load(d)
    blocks = {g0: fold_block(bp, p["n"], hkv, hd, dev) for g0, bp in p["blocks"].items()}
    qs = {l: bp["qscale"][li] for bp in p["blocks"].values() for li, l in enumerate(bp["layers"])}
    return p["n"], blocks, qs, p["meta"], d


def check_rope(blocks: dict, inv_freq: torch.Tensor):
    """The record's RoPE angles must be a subset of the model's frequencies (same convention as the conversion)."""
    th = torch.cat([b["theta"] for b in blocks.values()]).to(inv_freq.device)
    inv = inv_freq.float()
    rel = ((th[:, None] - inv[None]).abs() / inv[None]).min(1).values.max().item()
    if rel > 1e-5:
        raise ValueError(f"record RoPE angles do not match the model's inv_freq (rel {rel:.1e}): wrong base model?")
