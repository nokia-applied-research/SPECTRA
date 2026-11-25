"""Convert a HuggingFace Llama model to SPECTRA-Flash (post-training; the base model stays frozen).

    python -m spectra.flash.convert --model meta-llama/Llama-3.1-8B-Instruct --out flash-llama31-8b --gpus 0,1,2,3

One worker per GPU converts its share of the blocks (closed-form init + block-wise distillation on calibration
windows, see distill.py); the main process merges the blocks into <out>/params.safetensors + flash_config.json and
prints the report (held-out error per block, GPU time). Finished blocks are kept in <out>/blocks/ and skipped on a
rerun. Reference cost (Llama-3.1-8B, defaults): 2.6 GPU hours, 51 min on 4 x H100 NVL.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import subprocess
import sys
import time

import torch

from .calibration import calibration_windows
from .config import DEPLOY_RECENT, FLASH_CODE_SOURCES, FlashConfig
from . import params as P

SUPPORTED = {"llama"}


class _Stop(Exception):
    pass


@torch.no_grad()
def capture_block(model, wins, grp, log=print):
    """Attention inputs (post input-norm, the k_proj input) of the block's layers for every window, (N, S, d) bf16
    per layer on the model's GPU. The forward stops after the block's last layer."""
    layers = model.model.layers; dev = next(model.parameters()).device
    N, S, d = len(wins), wins[0].numel(), model.config.hidden_size
    X = [torch.empty(N, S, d, dtype=torch.bfloat16, device=dev) for _ in grp]
    cur = {}

    def mk(i, last):
        def hook(mod, inp):
            X[i][cur["wi"]].copy_(inp[0][0])
            if last:
                raise _Stop
        return hook
    hooks = [layers[l].self_attn.k_proj.register_forward_pre_hook(mk(i, i == len(grp) - 1)) for i, l in enumerate(grp)]
    try:
        for wi, w in enumerate(wins):
            cur["wi"] = wi
            try:
                model.model(input_ids=w[None].to(dev), use_cache=False)
            except _Stop:
                pass
    finally:
        for h in hooks:
            h.remove()
    return X


def _check(model, cfg):
    mt = getattr(model.config, "model_type", "")
    if mt not in SUPPORTED:
        raise NotImplementedError(f"SPECTRA-Flash supports {sorted(SUPPORTED)} models, got '{mt}'")
    a = model.model.layers[0].self_attn
    if a.k_proj.bias is not None or getattr(a, "k_norm", None) is not None:
        raise NotImplementedError("k/v biases and QK-norm are not supported")
    if model.config.num_hidden_layers % cfg.group_size:
        raise ValueError(f"{model.config.num_hidden_layers} layers do not split into blocks of {cfg.group_size}")


def _load_model(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to("cuda").eval()   # no accelerate needed
    model.requires_grad_(False)
    return model, tok


def prepare(model_id, out, cfg, log=print):
    """Calibration windows, written once to <out>/calib_windows.pt and shared by the workers. With decode_frac > 0
    a share of the training windows, and the held-out windows, end in decode_tokens tokens the model generated
    greedily from the window's own prefix (window length unchanged)."""
    path = os.path.join(out, "calib_windows.pt")
    if os.path.exists(path):
        return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    N = cfg.calib_windows + cfg.calib_heldout
    wins, corpus = calibration_windows(tok, cfg.calib_seq, N, 0.5, log=log, sources=FLASH_CODE_SOURCES)
    assert len(wins) == N, f"calibration corpus gave {len(wins)} windows of {cfg.calib_seq} tokens, need {N}"
    wins = [w.clone() for w in wins]                       # views of the whole corpus otherwise
    nn_ = N - int(N * 0.5)
    held = [nn_ - 1, N - 1][: cfg.calib_heldout]          # last text window, last code window
    train = [i for i in range(N) if i not in held]
    gen_start = [-1] * N
    if cfg.decode_frac > 0:
        G = cfg.decode_tokens
        text = [i for i in train if i < nn_]; code = [i for i in train if i >= nn_]
        pick = text[:round(cfg.decode_frac * len(text))] + code[:round(cfg.decode_frac * len(code))] + held
        model, _ = _load_model(model_id)
        dev = next(model.parameters()).device
        t0 = time.time()
        with torch.no_grad():
            for a in range(0, len(pick), cfg.decode_batch):
                idx = pick[a:a + cfg.decode_batch]
                ids = torch.stack([wins[i][:-G] for i in idx]).to(dev)
                o = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=G,
                                   min_new_tokens=G, do_sample=False, pad_token_id=tok.eos_token_id)
                for j, i in enumerate(idx):
                    wins[i] = torch.cat([wins[i][:-G], o[j, ids.shape[1]:].cpu().to(wins[i].dtype)])
                    gen_start[i] = cfg.calib_seq - G
        corpus = dict(corpus, decode=dict(windows=pick, tokens=G, seconds=round(time.time() - t0, 1)))
        log(f"[spectra-flash] generated {G} tokens at the end of {len(pick)} windows in {time.time() - t0:.0f}s")
        del model
    torch.save(dict(windows=wins, gen_start=gen_start, train=train, held=held, corpus=corpus), path + ".tmp")
    os.replace(path + ".tmp", path)


def worker(model_id, out, blocks, cfg, log=print):
    model, tok = _load_model(model_id)
    _check(model, cfg)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    cw = torch.load(os.path.join(out, "calib_windows.pt"), weights_only=False)
    wins, gen_start, train, held, corpus = cw["windows"], cw["gen_start"], cw["train"], cw["held"], cw["corpus"]
    N = len(wins)
    os.makedirs(os.path.join(out, "blocks"), exist_ok=True)
    for g0 in blocks:
        path = os.path.join(out, "blocks", f"block_{g0}.pt")
        if os.path.exists(path):
            log(f"[spectra-flash] block {g0}: done already, skipped"); continue
        grp = list(range(g0, g0 + cfg.group_size))
        t0 = time.time()
        X = capture_block(model, wins, grp, log)
        log(f"[spectra-flash] block {g0}: captured {len(grp)} layers x {N} windows in {time.time() - t0:.0f}s")
        from .distill import convert_block
        rec = convert_block(model, grp, X, train, held, cfg, log, gen_start=gen_start)
        rec["capture_seconds"] = time.time() - t0 - rec["seconds"]; rec["calibration"] = corpus
        torch.save(rec, path + ".tmp"); os.replace(path + ".tmp", path)
        del X
        torch.cuda.empty_cache()


def merge(model_id, out, cfg, wall, log=print):
    recs = [torch.load(f, map_location="cpu", weights_only=False) for f in glob.glob(os.path.join(out, "blocks", "block_*.pt"))]
    from transformers import AutoConfig
    L = AutoConfig.from_pretrained(model_id).num_hidden_layers
    assert len(recs) == L // cfg.group_size, f"{len(recs)} of {L // cfg.group_size} blocks converted"
    gpu_s = sum(r["seconds"] + r.get("capture_seconds", 0.0) for r in recs)
    gpu_s += recs[0]["calibration"].get("decode", {}).get("seconds", 0.0)      # decode-phase data: the generation pass
    meta = dict(model=model_id, config=cfg.to_dict(), calibration=recs[0]["calibration"],
                deploy=dict(sink=cfg.sink, recent=max(cfg.recent, DEPLOY_RECENT), fp8=True),
                cost=dict(gpu_hours=round(gpu_s / 3600, 2), wall_minutes=round(wall / 60, 1), gpu=recs[0]["gpu"],
                          calibration_tokens=cfg.calib_windows * cfg.calib_seq),
                heldout={str(r["g0"]): dict(init=r["err"]["init"]["all"], trained=r["err"]["trained"]["all"],
                                            trained_fp8=r["err"]["trained_fp8"]["all"],
                                            trained_fp8_buckets=r["err"]["trained_fp8"]["buckets"]) for r in recs},
                torch=torch.__version__)
    try:
        import transformers
        meta["transformers"] = transformers.__version__
    except Exception:
        pass
    P.save(out, recs, meta)
    log(f"[spectra-flash] wrote {out}/params.safetensors: {len(recs)} blocks, {meta['cost']['gpu_hours']} GPU hours "
        f"({meta['cost']['gpu']}), wall {meta['cost']['wall_minutes']} min")
    for r in sorted(recs, key=lambda r: r["g0"]):
        log(f"  block {r['g0']:2d}: held-out relMSE fp8 {[round(x, 3) for x in r['err']['trained_fp8']['all']]}")


def main(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--preset", default="lite", choices=["lite"])
    base = FlashConfig.preset(pre.parse_known_args(argv)[0].preset)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default="0", help="comma-separated GPU ids, one worker each")
    ap.add_argument("--preset", default="lite", choices=["lite"],
                    help="lite: the reference conversion (explicit flags override)")
    for f in dataclasses.fields(FlashConfig):
        typ = (lambda v: str(v).lower() in ("1", "true", "yes")) if isinstance(f.default, bool) else type(f.default)
        ap.add_argument("--" + f.name.replace("_", "-"), type=typ, default=getattr(base, f.name))
    ap.add_argument("--worker-blocks", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--prepare-only", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    cfg = FlashConfig(**{f.name: getattr(a, f.name) for f in dataclasses.fields(FlashConfig)})
    os.makedirs(a.out, exist_ok=True)
    log = lambda m: print(m, flush=True)
    if a.prepare_only:
        prepare(a.model, a.out, cfg, log=log)
        return
    if a.worker_blocks is not None:
        worker(a.model, a.out, [int(x) for x in a.worker_blocks.split(",") if x], cfg, log=log)
        return
    from transformers import AutoConfig
    L = AutoConfig.from_pretrained(a.model).num_hidden_layers
    starts = list(range(0, L, cfg.group_size))
    gpus = [g for g in a.gpus.split(",") if g]
    share = {g: starts[i::len(gpus)] for i, g in enumerate(gpus)}          # interleaved: balances early/late blocks
    t0 = time.time()
    flags = sum([["--" + f.name.replace("_", "-"), str(getattr(cfg, f.name))] for f in dataclasses.fields(FlashConfig)], [])
    if not os.path.exists(os.path.join(a.out, "calib_windows.pt")):       # windows (+ generated tails) on one GPU
        r = subprocess.run([sys.executable, "-m", "spectra.flash.convert", "--model", a.model, "--out", a.out,
                            "--prepare-only"] + flags, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpus[0]})
        if r.returncode != 0:
            sys.exit("[spectra-flash] preparing the calibration windows failed")
    procs = []
    for g, bl in share.items():
        cmd = [sys.executable, "-m", "spectra.flash.convert", "--model", a.model, "--out", a.out,
               "--worker-blocks", ",".join(map(str, bl))] + flags
        logf = open(os.path.join(a.out, f"worker_gpu{g}.log"), "w")
        procs.append((g, subprocess.Popen(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": g}, stdout=logf,
                                          stderr=subprocess.STDOUT), logf))
        print(f"[spectra-flash] GPU {g}: blocks {bl} (log {logf.name})", flush=True)
    bad = []
    for g, p, logf in procs:
        if p.wait() != 0:
            bad.append(g)
        logf.close()
    if bad:
        sys.exit(f"[spectra-flash] worker(s) on GPU {bad} failed; see {a.out}/worker_gpu*.log (finished blocks are kept)")
    merge(a.model, a.out, cfg, time.time() - t0, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
