"""longbench.py — LongBench evaluation harness (the field-standard long-context benchmark).

Implements the KIVI/GEAR subset (8 English tasks spanning single-doc QA, summarization,
few-shot, code) with the official LongBench prompt templates, generation lengths, and metrics
(token-F1 / ROUGE-L / classification / code edit-similarity). Method application (base / SPECTRA
configs / quant+evict baselines) reuses the exact same plumbing as ruler.py.

Tasks: qasper, qmsum, multi_news, trec, triviaqa, samsum, lcc, repobench-p.
Dataset: THUDM/LongBench (one config per task). Middle-truncate prompts to --max-len.
"""
from __future__ import annotations
import argparse, json, re, string, time
from collections import Counter
import difflib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..core import AttnHeads, collect_grams_and_mean, make_windows
from .. import baselines as kvb
from ..method import build_spectra as build_akvc

QUANT = {"kivi", "turboquant", "polarquant", "quarot"}

# ---- official LongBench prompt templates (English subset) ----
PROMPT = {
    "qasper": ("You are given a scientific article and a question. Answer the question as concisely "
               "as you can, using a single phrase or sentence if possible. If the question cannot be "
               "answered based on the information in the article, write \"unanswerable\".\n\n"
               "Article: {context}\n\n Answer the question based on the above article as concisely as "
               "you can, using a single phrase or sentence if possible.\n\nQuestion: {input}\n\nAnswer:"),
    "qmsum": ("You are given a meeting transcript and a query. Answer the query in one or more "
              "sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above "
              "meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:"),
    "multi_news": ("You are given several news passages. Write a one-page summary of all news.\n\n"
                   "News:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:"),
    "trec": ("Please determine the type of the question below. Here are some examples of questions.\n\n"
             "{context}\n{input}"),
    "triviaqa": ("Answer the question based on the given passage. Only give me the answer and do not "
                 "output any other words. The following are some examples.\n\n{context}\n\n{input}"),
    "samsum": ("Summarize the dialogue into a few short sentences. The following are some examples.\n\n"
               "{context}\n\n{input}"),
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}
MAXGEN = {"qasper": 128, "qmsum": 512, "multi_news": 512, "trec": 64,
          "triviaqa": 32, "samsum": 128, "lcc": 64, "repobench-p": 64}
METRIC = {"qasper": "qa_f1", "qmsum": "rouge", "multi_news": "rouge", "trec": "class",
          "triviaqa": "qa_f1", "samsum": "rouge", "lcc": "code", "repobench-p": "code"}


# ----------------------------- metrics (official-style) -----------------------------
def _norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def qa_f1(pred, refs):
    best = 0.0
    pt = _norm(pred).split()
    for r in refs:
        gt = _norm(r).split()
        common = Counter(pt) & Counter(gt)
        ns = sum(common.values())
        if ns == 0 or not pt or not gt:
            best = max(best, 0.0); continue
        prec = ns / len(pt); rec = ns / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def _lcs(a, b):
    n, m = len(a), len(b)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            dp[j] = prev + 1 if a[i-1] == b[j-1] else max(dp[j], dp[j-1])
            prev = tmp
    return dp[m]


def rouge_l(pred, refs):
    best = 0.0
    p = _norm(pred).split()
    for r in refs:
        g = _norm(r).split()
        if not p or not g:
            continue
        l = _lcs(p, g)
        if l == 0:
            continue
        prec = l / len(p); rec = l / len(g)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def class_score(pred, refs, all_classes):
    p = pred.strip().lower()
    em = 0.0
    for r in refs:
        rl = r.strip().lower()
        # the predicted class is the ground-truth iff it appears and no other class "covers" it more
        if rl in p:
            em = 1.0
    return em


def code_sim(pred, refs):
    pl = pred.lstrip("\n").split("\n")
    pred1 = pl[0] if pl else ""
    best = 0.0
    for r in refs:
        best = max(best, difflib.SequenceMatcher(None, pred1, r).ratio())
    return best


def score_one(task, pred, refs, all_classes):
    m = METRIC[task]
    if m == "qa_f1":
        return qa_f1(pred, refs)
    if m == "rouge":
        return rouge_l(pred, refs)
    if m == "class":
        return class_score(pred, refs, all_classes)
    if m == "code":
        return code_sim(pred, refs)
    return 0.0


# ----------------------------- prompt build + truncation -----------------------------
def make_input(tok, task, ex, max_len, dev):
    text = PROMPT[task].format(context=ex.get("context", ""), input=ex.get("input", ""))
    if getattr(tok, "chat_template", None):
        ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                      add_generation_prompt=True, return_tensors="pt",
                                      return_dict=False)
    else:
        ids = tok(text, return_tensors="pt", add_special_tokens=True).input_ids
    ids = ids[0]
    if ids.shape[0] > max_len:                       # middle truncation (LongBench convention)
        h = max_len // 2
        ids = torch.cat([ids[:h], ids[-(max_len - h):]])
    return ids.unsqueeze(0).to(dev)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--dtype", default="float16")
    ap.add_argument("--tasks", default="qasper,qmsum,multi_news,trec,triviaqa,samsum,lcc,repobench-p")
    ap.add_argument("--max-samples", type=int, default=150)
    ap.add_argument("--max-len", type=int, default=31500)
    # methods (same surface as ruler)
    ap.add_argument("--configs", default=""); ap.add_argument("--baselines", default="")
    ap.add_argument("--kv-ratio", type=float, default=0.25); ap.add_argument("--wo-rank", type=int, default=4)
    ap.add_argument("--mla-ridge", type=float, default=1e-3); ap.add_argument("--wo-ridge", type=float, default=1e-2)
    ap.add_argument("--group", type=int, default=64); ap.add_argument("--center", action="store_true")
    ap.add_argument("--heavy-k", type=int, default=0); ap.add_argument("--heavy-bits", type=int, default=8)
    ap.add_argument("--hadamard", action="store_true"); ap.add_argument("--plain-svd", action="store_true")
    ap.add_argument("--water-filling", action="store_true")
    ap.add_argument("--wf-bmax", type=int, default=8); ap.add_argument("--wf-clip", type=float, default=4.0)
    ap.add_argument("--wf-residual", type=int, default=0)
    ap.add_argument("--gk", type=int, default=128); ap.add_argument("--gv", type=int, default=32)
    ap.add_argument("--residual", type=int, default=128)
    ap.add_argument("--calib-windows", type=int, default=32); ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--skip-base", action="store_true", help="skip base (fp16) pass; use when base is already computed")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    dev = args.device; dtype = getattr(torch, args.dtype)
    tasks = [t for t in args.tasks.split(",") if t.strip()]
    print(f"[load] {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(dev).eval()
    heads = AttnHeads(model)
    print(f"[cfg] arch={heads.arch} L={heads.L} kv_dim={heads.kv_dim}", flush=True)

    from datasets import load_dataset
    data = {}
    for t in tasks:
        ds = load_dataset("THUDM/LongBench", t, split="test", trust_remote_code=True)
        data[t] = list(ds.select(range(min(args.max_samples, len(ds)))))
        print(f"[data] {t}: {len(data[t])}", flush=True)

    def run_method(gen_fn, label):
        res = {}
        for t in tasks:
            sc = []
            for ex in data[t]:
                ids = make_input(tok, t, ex, args.max_len, dev)
                txt = gen_fn(ids, MAXGEN[t])
                sc.append(score_one(t, txt, ex.get("answers", []), ex.get("all_classes", [])))
            res[t] = round(100 * sum(sc) / max(1, len(sc)), 2)
            print(f"[{label}] {t}: {res[t]}", flush=True)
        res["overall"] = round(sum(res[t] for t in tasks) / len(tasks), 2)
        return res

    results = {}

    def base_gen(ids, mg):
        o = model.generate(ids, max_new_tokens=mg, do_sample=False, pad_token_id=tok.eos_token_id, use_cache=True)
        return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True)
    if not args.skip_base:
        results["base"] = {"acc": run_method(base_gen, "base"), "total_x": 1.0}

    # SPECTRA configs
    cfgs = [s for s in args.configs.split(",") if s.strip()]
    if cfgs:
        hay = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"] if t.strip())
        WKo = [l.self_attn.k_proj.weight.detach().clone() for l in heads.layers]
        WVo = [l.self_attn.v_proj.weight.detach().clone() for l in heads.layers]
        WOo = [l.self_attn.o_proj.weight.detach().clone() for l in heads.layers]
        kb = [(l.self_attn.k_proj.bias.detach().clone() if l.self_attn.k_proj.bias is not None else None) for l in heads.layers]
        vb = [(l.self_attn.v_proj.bias.detach().clone() if l.self_attn.v_proj.bias is not None else None) for l in heads.layers]
        ok_ = [l.self_attn.k_proj for l in heads.layers]; ov_ = [l.self_attn.v_proj for l in heads.layers]

        def restore():
            for l in range(heads.L):
                heads.layers[l].self_attn.k_proj = ok_[l]; heads.layers[l].self_attn.v_proj = ov_[l]
                ok_[l].weight.copy_(WKo[l]); ov_[l].weight.copy_(WVo[l])
                heads.layers[l].self_attn.o_proj.weight.copy_(WOo[l])

        G, hmean = collect_grams_and_mean(model, heads, make_windows(tok, hay, args.seq_len, args.calib_windows), dev)
        if args.plain_svd:
            G = [torch.eye(heads.d, device=dev, dtype=G[l].dtype) for l in range(heads.L)]
        for spec in cfgs:
            B, bits = (int(x) for x in spec.split(":")); restore()
            cw = make_windows(tok, hay, args.seq_len, args.calib_windows)
            r_k, r_v, cf = build_akvc(model, heads, G, hmean, WKo, WVo, WOo, kb, vb, cw, dev, dtype, B, bits, args)
            results[f"akvc_B{B}_b{bits}"] = {"acc": run_method(base_gen, f"akvc_B{B}_b{bits}"),
                                             "total_x": round(1 / cf, 2)}
        restore()

    # quant / evict baselines
    for spec in [s for s in args.baselines.split(",") if s.strip()]:
        method, param = spec.split(":"); param = int(param)
        if method in QUANT:
            _, eb = kvb.make_quant_cache(method, heads.hd, param, dev, gk=args.gk, gv=args.gv, residual=args.residual)

            def gen_fn(ids, mg, _m=method, _p=param):
                c, _ = kvb.make_quant_cache(_m, heads.hd, _p, dev, gk=args.gk, gv=args.gv, residual=args.residual)
                o = model.generate(ids, max_new_tokens=mg, do_sample=False, pad_token_id=tok.eos_token_id,
                                   use_cache=True, past_key_values=c)
                return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True)
            results[f"{method}_b{param}"] = {"acc": run_method(gen_fn, f"{method}_b{param}"),
                                             "total_x": round(16 / eb, 2)}
        else:
            # token-eviction: effective compression = sum(seq_len) / sum(min(budget, seq_len)),
            # since kept tokens stay fp16 (bits/elem ratio vs the full fp16 cache).
            tok_acc = {"full": 0, "kept": 0}
            def gen_fn(ids, mg, _m=method, _p=param, _a=tok_acc):
                n = ids.shape[1]
                _a["full"] += n; _a["kept"] += min(_p, n)
                txt, _ = kvb.evict_generate(model, tok, heads, ids, mg, _m, _p)
                return txt
            acc = run_method(gen_fn, f"{method}_bud{param}")
            tx = round(tok_acc["full"] / max(1, tok_acc["kept"]), 2)
            results[f"{method}_bud{param}"] = {"acc": acc, "total_x": tx, "budget": param}

    out = dict(model=args.model, opts=vars(args), tasks=tasks, results=results)
    with open(args.out, "w") as f: json.dump(out, f, indent=2)
    print("\n=== LONGBENCH SUMMARY (score %) ===", flush=True)
    for k, v in results.items():
        print(f"{k:24s} x={v.get('total_x', v.get('budget','-')):>6}  overall={v['acc']['overall']}", flush=True)
    print(f"[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
