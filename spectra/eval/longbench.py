"""LongBench (8 English tasks), the protocol behind the paper's tables.

Prompting and scoring follow the official THUDM/LongBench harness:
  * prompts and max new tokens: config/dataset2prompt.json and config/dataset2maxlen.json (verbatim);
  * prompts longer than --max-length are truncated in the middle, before the chat template;
  * the model's chat template is applied except on trec, triviaqa, samsum, lcc and repobench-p;
  * greedy decoding; samsum also stops at a newline;
  * scores: metrics.py (verbatim) with the official first-line post-processing for trec / triviaqa / samsum.
The full test split of every task is used by default. SPECTRA (and its dense reference): 🤗 Transformers, fp16,
batched generation (left-padded, sorted by length, at most 16 prompts or 64K tokens per batch), as in the paper's
runs. SPECTRA-Flash (and its dense reference): the SPECTRA-Flash decode engine, bf16, one prompt at a time.

    python -m spectra.eval.longbench --model NousResearch/Meta-Llama-3.1-8B-Instruct --out runs/dense.json
    python -m spectra.eval.longbench --model NousResearch/Meta-Llama-3.1-8B-Instruct --preset 8.8x --out runs/8.8x.json
    python -m spectra.eval.longbench --model meta-llama/Llama-3.1-8B-Instruct --flash <params> --out runs/flash.json
    python -m spectra.eval.longbench --model meta-llama/Llama-3.1-8B-Instruct --flash-dense --out runs/flash_dense.json

The output records the score and the number of samples per task, the context length of every sample, and for
SPECTRA the compression ratio over those samples (window counted, see spectra.accounting).
"""
from __future__ import annotations
import argparse, json, os, time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
DATASET2PROMPT = json.load(open(os.path.join(_HERE, "config", "dataset2prompt.json")))
DATASET2MAXLEN = json.load(open(os.path.join(_HERE, "config", "dataset2maxlen.json")))
DATASET2METRIC_NAME = {
    "narrativeqa": "qa_f1_score", "qasper": "qa_f1_score", "multifieldqa_en": "qa_f1_score",
    "hotpotqa": "qa_f1_score", "2wikimqa": "qa_f1_score", "musique": "qa_f1_score",
    "gov_report": "rouge_score", "qmsum": "rouge_score", "multi_news": "rouge_score",
    "trec": "classification_score", "triviaqa": "qa_f1_score", "samsum": "rouge_score",
    "passage_retrieval_en": "retrieval_score", "passage_count": "count_score",
    "lcc": "code_sim_score", "repobench-p": "code_sim_score",
}
NO_CHAT_TASKS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
TASKS = "qasper,qmsum,multi_news,trec,triviaqa,samsum,lcc,repobench-p"


def build_chat(tok, prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)


def scorer(dataset, predictions, answers, all_classes):
    """Official LongBench eval.py scorer."""
    from . import metrics                       # needs rouge, fuzzywuzzy, jieba (pip install -e '.[eval]')
    metric = getattr(metrics, DATASET2METRIC_NAME[dataset])
    total = 0.
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, metric(prediction, ground_truth, all_classes=all_classes))
        total += score
    return round(100 * total / len(predictions), 2)


def encode(tok, task, ex, max_len, use_chat=True):
    prompt = DATASET2PROMPT[task].format(**ex)
    ids = tok(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(ids) > max_len:
        half = max_len // 2
        prompt = (tok.decode(ids[:half], skip_special_tokens=True)
                  + tok.decode(ids[-half:], skip_special_tokens=True))
    if use_chat and task not in NO_CHAT_TASKS:
        return tok(build_chat(tok, prompt), truncation=False, return_tensors="pt",
                   add_special_tokens=False).input_ids[0]
    return tok(prompt, truncation=False, return_tensors="pt", add_special_tokens=True).input_ids[0]


@torch.no_grad()
def generate_batched(model, tok, prompts, max_new, dev, samsum=False, budget=64000, max_bs=16):
    """Greedy generation in left-padded batches of similar length. Returns the decoded outputs in input order."""
    order = sorted(range(len(prompts)), key=lambda i: prompts[i].shape[0])
    out = [None] * len(prompts)
    eos = tok.eos_token_id
    eos_ids = [eos]
    if samsum:
        nl = tok.encode("\n", add_special_tokens=False)
        if nl:
            eos_ids = [eos, nl[-1]]
    i = 0
    while i < len(order):
        maxlen = prompts[order[i]].shape[0]; j = i
        while j < len(order):
            cand = max(maxlen, prompts[order[j]].shape[0])
            if (j - i + 1) > max_bs or (j - i + 1) * cand > budget:
                break
            maxlen = cand; j += 1
        idxs = order[i:j]; chunk = [prompts[k] for k in idxs]
        ml = max(x.shape[0] for x in chunk)
        ids = torch.full((len(chunk), ml), tok.pad_token_id, dtype=torch.long)
        am = torch.zeros((len(chunk), ml), dtype=torch.long)
        for t, x in enumerate(chunk):
            ids[t, ml - x.shape[0]:] = x; am[t, ml - x.shape[0]:] = 1
        o = model.generate(input_ids=ids.to(dev), attention_mask=am.to(dev), max_new_tokens=max_new,
                           do_sample=False, num_beams=1, pad_token_id=eos, eos_token_id=eos_ids)
        for t, k in enumerate(idxs):
            out[k] = tok.decode(o[t, ml:], skip_special_tokens=True)
        i = j
    return out


def load_task(task):
    """The task's test records in file order. Uses the LongBench dataset script (datasets < 4); with datasets >= 4,
    which dropped script loaders, reads the same jsonl files from the dataset's data.zip."""
    try:
        from datasets import load_dataset
        return list(load_dataset("THUDM/LongBench", task, split="test", trust_remote_code=True))
    except Exception:  # noqa: BLE001
        import zipfile
        from huggingface_hub import hf_hub_download
        z = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
        with zipfile.ZipFile(z) as zf, zf.open(f"data/{task}.jsonl") as f:
            return [json.loads(line) for line in f if line.strip()]


def flash_generator(model_id, params, prompts_by_task, log=print):
    """SPECTRA-Flash decode engine at batch 1. params=None: the same engine with the plain bf16 cache (the dense
    reference). Returns gen(task, prompts) -> outputs."""
    from ..flash import engine as EN
    tcap = max(p.numel() for ps in prompts_by_task.values() for p in ps) + max(
        DATASET2MAXLEN[t] for t in prompts_by_task) + 64
    M = EN.Model(model_id, params, "cuda")
    E = EN.Engine(M, EN.Cache(M, "flash8" if params else "dense", 1, tcap))

    def gen(task, prompts, tok_):
        eos_ids = {tok_.eos_token_id}
        if task == "samsum":
            nl = tok_.encode("\n", add_special_tokens=False)
            if nl:
                eos_ids.add(nl[-1])
        outs = []
        for ids in prompts:
            with torch.no_grad():
                toks = E.generate(ids.cuda(), DATASET2MAXLEN[task], eos=eos_ids)
            outs.append(tok_.decode(toks, skip_special_tokens=True))
        return outs
    return gen


def run_longbench(model, tok, dev, tasks, max_samples=None, max_len=31500, use_chat=True, log=print, gen=None):
    """Returns {task: score, 'overall': mean, '_n': {task: n}, '_len': {task: [context lengths]}}.
    gen(task, prompts, tok) replaces the batched 🤗 Transformers generation (SPECTRA-Flash engine)."""
    scores, n, lens = {}, {}, {}
    for task in tasks:
        data = load_task(task)
        data = data if max_samples is None else data[:max_samples]
        prompts = [encode(tok, task, ex, max_len, use_chat).to(dev) for ex in data]
        if gen is None:
            outs = generate_batched(model, tok, prompts, DATASET2MAXLEN[task], dev, samsum=(task == "samsum"))
        else:
            outs = gen(task, prompts, tok)
        all_classes = data[0].get("all_classes", []) if data else []
        scores[task] = scorer(task, outs, [ex["answers"] for ex in data], all_classes)
        n[task] = len(outs)
        lens[task] = [int(p.shape[-1]) for p in prompts]
        log(f"[longbench] {task}: {scores[task]}  (n={n[task]})")
    scores["overall"] = round(float(np.mean([scores[t] for t in tasks])), 2)
    log(f"[longbench] overall: {scores['overall']}")
    scores["_n"], scores["_len"] = n, lens
    return scores


def main():
    ap = argparse.ArgumentParser(description="LongBench-8 with the official prompts and scorer, full test split.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--preset", default="", help="SPECTRA preset (e.g. 8.8x, see docs/presets.md); empty = dense")
    ap.add_argument("--calib", default="wikitext+code", choices=["wikitext+code", "wikitext"])
    ap.add_argument("--flash", default="", help="SPECTRA-Flash params dir or HF Hub repo (decode engine, batch 1)")
    ap.add_argument("--flash-dense", action="store_true", help="the SPECTRA-Flash engine with the plain cache (dense)")
    ap.add_argument("--tasks", default=TASKS)
    ap.add_argument("--max-samples", type=int, default=None, help="cap per task (default: full test split)")
    ap.add_argument("--max-length", type=int, default=31500)
    ap.add_argument("--no-chat", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ..accounting import accounting_block
    dev, dtype = args.device, getattr(torch, args.dtype)
    tasks = [t for t in args.tasks.split(",") if t.strip()]
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    rec = {"model": args.model, "preset": args.preset or "dense", "max_samples": args.max_samples,
           "max_length": args.max_length}
    if args.flash or args.flash_dense:
        from ..accounting import flash_accounting
        data = {t: load_task(t) for t in tasks}
        data = {t: (d if args.max_samples is None else d[:args.max_samples]) for t, d in data.items()}
        prompts = {t: [encode(tok, t, ex, args.max_length, not args.no_chat) for ex in d] for t, d in data.items()}
        gen = flash_generator(args.model, args.flash or None, prompts)
        rec.update(preset="spectra-flash" if args.flash else "dense (flash engine)", params=args.flash or None,
                   backend="spectra.flash engine, bf16, batch 1")
        scores = run_longbench(None, tok, "cpu", tasks, args.max_samples, args.max_length, not args.no_chat, gen=gen)
        rec.update(scores={t: scores[t] for t in tasks}, overall=scores["overall"], n=scores["_n"],
                   context_lengths=scores["_len"], secs=round(time.time() - t0, 1))
        if args.flash:
            rec["accounting"] = flash_accounting(scores["_len"])
            print(f"[longbench] compression over the evaluated samples: {rec['accounting']['ratio']}x", flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(rec, open(args.out, "w"), indent=1)
        print(f"[out] {args.out}  overall={rec['overall']}  ({rec['secs']}s)", flush=True)
        return
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, trust_remote_code=True,
                                                 low_cpu_mem_usage=True, device_map=dev).eval()
    if args.preset:
        from ..api import compress
        model, tok = compress(model, tok, preset=args.preset, calib=args.calib, device=dev, dtype=dtype)
        rec.update(calib=args.calib, latent_ratio=model._spectra["compression"], window=model._spectra["window"])

    scores = run_longbench(model, tok, dev, tasks, args.max_samples, args.max_length, not args.no_chat)
    rec.update(scores={t: scores[t] for t in tasks}, overall=scores["overall"], n=scores["_n"],
               context_lengths=scores["_len"], secs=round(time.time() - t0, 1))
    if args.preset:
        w = sum(model._spectra["window"]) if model._spectra["window"] else 0
        rec["accounting"] = accounting_block(rec["latent_ratio"], scores["_len"], window=w)
        print(f"[longbench] compression over the evaluated samples: {rec['accounting']['ratio']}x "
              f"(latent {rec['latent_ratio']}x, exact window {w} tokens)", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"[out] {args.out}  overall={rec['overall']}  ({rec['secs']}s)", flush=True)


if __name__ == "__main__":
    main()
