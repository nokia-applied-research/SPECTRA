"""SPECTRA-Flash params format: <dir>/params.safetensors + <dir>/flash_config.json.

Tensors (fp32):
  block.{g0}.W_d (n d, R)   block.{g0}.W_r (n d, r)   block.{g0}.theta (r/2,)
  block.{g0}.c_scale (R,)   block.{g0}.s_scale (r,)   static fp8 record scales (per channel / per RoPE pair)
  layer.{l}.W_UK (R, kv)    layer.{l}.W_UV (R, kv)    layer.{l}.W_QR    layer.{l}.q_scale (1,)
g0 is the block's first layer. W_QR is (d, H r) on Llama (rope query from the hidden state) and (H, D, r) on Qwen3.5
(from the normalized query of each head; `arch` "qwen3_5" in the config).
flash_config.json: format, version, model, shapes, the conversion config, calibration corpus, cost, held-out errors;
`blocks` lists each block's layers when they are not consecutive (Qwen3.5: its full-attention layers, 4 apart).
"""
from __future__ import annotations

import json
import os

import torch

FORMAT, VERSION = "spectra-flash", 1


def save(out_dir, recs, meta):
    from safetensors.torch import save_file
    t = {}
    for rec in sorted(recs, key=lambda r: r["g0"]):
        g0, p = rec["g0"], rec["params"]
        t[f"block.{g0}.W_d"] = p["Wd"]; t[f"block.{g0}.W_r"] = p["Wr"]; t[f"block.{g0}.theta"] = p["theta"]
        t[f"block.{g0}.c_scale"] = rec["c_scale"]; t[f"block.{g0}.s_scale"] = rec["s_scale"]
        for li, l in enumerate(rec.get("layers") or range(g0, g0 + rec["n"])):
            t[f"layer.{l}.W_UK"] = p["WuK"][li]; t[f"layer.{l}.W_UV"] = p["WuV"][li]; t[f"layer.{l}.W_QR"] = p["WQR"][li]
            t[f"layer.{l}.q_scale"] = torch.tensor([rec["q_scale"][li]], dtype=torch.float32)
    save_file({k: v.float().contiguous() for k, v in t.items()}, os.path.join(out_dir, "params.safetensors"),
              metadata={"format": FORMAT, "version": str(VERSION)})
    with open(os.path.join(out_dir, "flash_config.json"), "w") as f:
        json.dump(dict(format=FORMAT, version=VERSION, **meta), f, indent=1)


def load(path, device="cpu"):
    """-> {"n", "meta", "blocks": {g0: {layers, Wd, Wr, theta, cs, ss, WuK: [..], WuV: [..], WQR: [..], qscale: [..]}}}"""
    from safetensors.torch import load_file
    d = path if os.path.isdir(path) else os.path.dirname(path)
    meta = json.load(open(os.path.join(d, "flash_config.json")))
    assert meta.get("format") == FORMAT, f"{d}: not a {FORMAT} params dir"
    t = load_file(os.path.join(d, "params.safetensors"), device=str(device))
    n = meta["config"]["group_size"]
    layers = {g[0]: list(g) for g in meta.get("blocks", [])}
    blocks = {}
    for g0 in sorted({int(k.split(".")[1]) for k in t if k.startswith("block.")}):
        ls = layers.get(g0) or list(range(g0, g0 + n))
        blocks[g0] = dict(layers=ls, Wd=t[f"block.{g0}.W_d"], Wr=t[f"block.{g0}.W_r"], theta=t[f"block.{g0}.theta"],
                          cs=t[f"block.{g0}.c_scale"], ss=t[f"block.{g0}.s_scale"],
                          WuK=[t[f"layer.{l}.W_UK"] for l in ls], WuV=[t[f"layer.{l}.W_UV"] for l in ls],
                          WQR=[t[f"layer.{l}.W_QR"] for l in ls], qscale=[float(t[f"layer.{l}.q_scale"]) for l in ls])
    return dict(n=n, meta=meta, blocks=blocks)
