"""Needle in a haystack, the grid behind the paper's retrieval results.

WikiText-2 text as the haystack; one magic access code (or k codes, RULER-style) planted at each depth; the model's
chat template with a system prompt; greedy decoding of 24 new tokens (32 for several needles). A cell is 1 when the
code appears in the output (the fraction of codes found with several needles). Defaults: contexts 1K-32K, depths
0-100% in 7 steps.

    python -m spectra.eval.niah --model NousResearch/Meta-Llama-3.1-8B-Instruct --preset 11x --out runs/niah.json
    python -m spectra.eval.niah --model NousResearch/Meta-Llama-3.1-8B-Instruct --preset 11x --needles 4 --out runs/niah4.json
"""
from __future__ import annotations
import argparse, json, os, time

import torch


def run_niah(gen_fn, tok, dev, contexts, depths, magic="7429183", gen_tokens=24, needles=1, log=print):
    """gen_fn(ids, max_new) -> generated text. Returns {grid, overall, contexts, depths}."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    hay = "\n\n".join(t for t in ds["text"] if t.strip())
    hay_ids = tok(hay, add_special_tokens=False)["input_ids"]
    n = max(1, int(needles))
    codes = [str(int(magic) + 111111 * i) for i in range(n)]
    ndl_ids = [tok(f" One special detail to remember: magic access code #{i+1} for this document is {codes[i]}. ",
                   add_special_tokens=False)["input_ids"] for i in range(n)]
    ndl_len = sum(len(x) for x in ndl_ids)
    sysp = ("You are a helpful assistant. Read the document carefully and answer the question using "
            "only information from the document.")
    if n == 1:
        q = "\n\nWhat is the magic access code for this document? Answer with only the number."
        gtok = gen_tokens
    else:
        q = f"\n\nThere are {n} magic access codes hidden in the document. List ALL {n} of them, separated by commas."
        gtok = max(gen_tokens, 32)
    grid = {}
    for ctx in contexts:
        grid[ctx] = {}
        budget = ctx - ndl_len - 96
        if budget < 64:
            for d in depths:
                grid[ctx][d] = None
            continue
        base_hay = hay_ids[:budget]
        for d in depths:
            offs = sorted(set(int(round(((d + i * (90.0 / n)) % 100) / 100.0 * len(base_hay))) for i in range(n)))
            doc_ids = []; prev = 0
            for k, off in enumerate(offs):
                doc_ids += base_hay[prev:off] + ndl_ids[k % n]; prev = off
            doc_ids += base_hay[prev:]
            doc = tok.decode(doc_ids)
            msgs = [{"role": "system", "content": sysp}, {"role": "user", "content": doc + q}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
            if not torch.is_tensor(ids):
                ids = ids["input_ids"] if hasattr(ids, "__getitem__") and "input_ids" in ids else torch.as_tensor(ids)
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            g = gen_fn(ids.to(dev), gtok).replace(",", "")
            found = sum(1 for c in codes if c in g)
            grid[ctx][d] = round(found / n, 3)
            log(f"[niah] ctx={ctx} depth={d}% -> {found}/{n}")
    allv = [grid[c][d] for c in contexts for d in depths if grid[c].get(d) is not None]
    overall = round(100 * sum(allv) / max(1, len(allv)), 2)
    log(f"[niah] overall {overall}%")
    return {"grid": {str(c): grid[c] for c in contexts}, "overall": overall, "contexts": contexts, "depths": depths}


def main():
    ap = argparse.ArgumentParser(description="Needle-in-a-haystack grid for the dense model or a SPECTRA preset.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--preset", default="", help="SPECTRA preset (e.g. 11x, see docs/presets.md); empty = dense")
    ap.add_argument("--calib", default="wikitext+code", choices=["wikitext+code", "wikitext"])
    ap.add_argument("--contexts", default="1024,2048,4096,8192,16384,32768")
    ap.add_argument("--depths", default="0,17,33,50,67,83,100")
    ap.add_argument("--needles", type=int, default=1)
    ap.add_argument("--gen-tokens", type=int, default=24)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev, dtype = args.device, getattr(torch, args.dtype)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, trust_remote_code=True,
                                                 low_cpu_mem_usage=True, device_map=dev).eval()
    rec = {"model": args.model, "preset": args.preset or "dense", "needles": args.needles}
    if args.preset:
        from ..api import compress
        model, tok = compress(model, tok, preset=args.preset, calib=args.calib, device=dev, dtype=dtype)
        rec.update(calib=args.calib, latent_ratio=model._spectra["compression"], window=model._spectra["window"])

    @torch.no_grad()
    def gen_fn(ids, max_new):
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    ctxs = [int(x) for x in args.contexts.split(",") if x.strip()]
    deps = [int(x) for x in args.depths.split(",") if x.strip()]
    rec.update(run_niah(gen_fn, tok, dev, ctxs, deps, gen_tokens=args.gen_tokens, needles=args.needles))
    rec["secs"] = round(time.time() - t0, 1)
    print("depth\\ctx " + " ".join(f"{c:>6}" for c in ctxs))
    for d in deps:
        print(f"{d:>8}% " + " ".join(f"{rec['grid'][str(c)][d] if rec['grid'][str(c)][d] is not None else '-':>6}"
                                     for c in ctxs))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"[out] {args.out}  overall={rec['overall']}%", flush=True)


if __name__ == "__main__":
    main()
