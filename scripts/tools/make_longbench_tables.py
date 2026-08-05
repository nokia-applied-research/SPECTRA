"""Per-model LongBench per-task LaTeX tables (columns = tasks, rows = method x compression).

Rows are grouped into compression bands separated by \\midrule so methods are comparable within a band.
Usage:
  python scripts/tools/make_longbench_tables.py [--root results/arxiv_v1] [--only MODELKEY] [--label-suffix S] > out.tex
"""
import json, os, glob, argparse

MODELS = [("llama3.1-8b", "Llama-3.1-8B-Instruct"),
          ("mistral-7b",  "Mistral-7B-Instruct-v0.3"),
          ("qwen2.5-7b",  "Qwen2.5-7B-Instruct"),
          ("qwen2.5-14b", "Qwen2.5-14B-Instruct"),
          ("llama2-7b",   "Llama-2-7B")]
TASKS = ["qasper","qmsum","multi_news","trec","triviaqa","samsum","lcc","repobench-p"]
HDR   = ["Qasper","QMSum","MNews","TREC","TQA","SAMSum","LCC","RB-P"]

PRETTY = {
    "base_base": "base (fp16)",
    "kivi_bits_4": "KIVI b4", "kivi_bits_3": "KIVI b3", "kivi_bits_2": "KIVI b2",
    "turboquant_bits_3": "TurboQuant b3", "turboquant_bits_2": "TurboQuant b2",
    "polarquant_bits_3": "PolarQuant b3", "polarquant_bits_2": "PolarQuant b2",
    "quarot_bits_2": "QuaRot b2",
}
def pretty(stem):
    if stem in PRETTY: return PRETTY[stem]
    if stem.startswith("spectra_wf_B_"):
        return "\\spectra-WF " + stem.replace("spectra_wf_B_","").replace("_bits_","/b")
    if stem.startswith("spectra_B_"):
        return "\\spectra-LR " + stem.replace("spectra_B_","").replace("_bits_","/b")
    if stem.startswith("palu_B_"):
        return "Palu " + stem.replace("palu_B_","").replace("_bits_","/b")
    return stem

def band(tx):
    if not isinstance(tx,(int,float)): return 9
    if tx < 1.5:  return 0      # base
    if tx < 5.0:  return 1      # near-lossless 3-4.5x
    if tx < 8.5:  return 2      # cliff zone 5-7.6x
    if tx < 11.0: return 3      # ~9x
    return 4                    # ~12-13x

def rows_for(root, model):
    rows = []
    for fp in glob.glob(f"{root}/{model}/longbench/*.json"):
        if fp.endswith(".meta.json"): continue
        stem = os.path.basename(fp)[:-5]
        if stem.startswith(("snapkv","pyramidkv")): continue    # off-axis (token budget)
        try: d = json.load(open(fp))
        except: continue
        res = d.get("results", {})
        key = "base" if stem == "base_base" else next((k for k in res if k != "base"), None)
        if key is None or "acc" not in res.get(key, {}): continue
        acc = res[key]["acc"]; tx = 1.0 if stem == "base_base" else res[key].get("total_x")
        rows.append((tx if isinstance(tx,(int,float)) else 1e9, pretty(stem), tx, acc))
    return sorted(rows, key=lambda r: r[0])

def fmt(x): return "--" if x is None else f"{x:.2f}"

def emit(root, model, nice, label_suffix=""):
    rows = rows_for(root, model)
    if not rows: return None
    lab = f"tab:lb-{model}" + (f"-{label_suffix}" if label_suffix else "")
    L = ["\\begin{table}[t]", "\\centering\\small\\setlength{\\tabcolsep}{3.5pt}",
         f"\\caption{{\\textbf{{LongBench per task --- {nice}.}} Higher is better; Avg is the mean over "
         f"the 8 tasks. Comp.\\ is effective bits/elem vs fp16 KV. Rows are grouped by compression band.}}",
         f"\\label{{{lab}}}", "\\resizebox{\\textwidth}{!}{%",
         "\\begin{tabular}{l c " + "c"*len(TASKS) + " c}", "\\toprule",
         "Method & Comp. & " + " & ".join(HDR) + " & \\textbf{Avg} \\\\", "\\midrule"]
    prev = None
    for tx, name, txr, acc in rows:
        b = band(tx)
        if prev is not None and b != prev: L.append("\\midrule")
        prev = b
        comp = "$1.0\\times$" if b == 0 else (f"${txr:.2f}\\times$" if isinstance(txr,(int,float)) else "--")
        cells = " & ".join(fmt(acc.get(t)) for t in TASKS)
        L.append(f"{name} & {comp} & {cells} & \\textbf{{{fmt(acc.get('overall'))}}} \\\\")
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}\n"]
    return "\n".join(L)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/arxiv_v1")
    ap.add_argument("--only", default=None, help="emit only this model key")
    ap.add_argument("--label-suffix", default="", help="suffix appended to \\label (avoid clashes)")
    a = ap.parse_args()
    for m, nice in MODELS:
        if a.only and m != a.only: continue
        t = emit(a.root, m, nice, a.label_suffix)
        if t: print(t)

if __name__ == "__main__":
    main()
