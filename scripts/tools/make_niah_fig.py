"""Multi-panel needle-in-a-haystack heatmap from a run_niah.py JSON.

Usage: python make_niah_fig.py l31_niah.json out.png
One panel per method (base + baselines + SPECTRA), depth (y) x context (x), RdYlGn (green=hit).
"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def pretty(tag, tx):
    if tag == "base": return "base (fp16)"
    if tag.startswith("kivi"):
        b = "".join(ch for ch in tag if ch.isdigit()) or "2"
        return f"KIVI {b}-bit ({tx:.1f}x)"
    if tag.startswith("B") and "_b" in tag:  # SPECTRA latent config Bxxxx_bx
        return f"SPECTRA-LR ({tx:.1f}x)"
    return f"{tag} ({tx:.1f}x)"

def main():
    src, out = sys.argv[1], sys.argv[2]
    d = json.load(open(src))
    res = d["results"] if "results" in d else d
    # order: base first, then by compression
    items = sorted(res.items(), key=lambda kv: (kv[0] != "base", kv[1].get("total_x", 1.0)))
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.0), squeeze=False)
    for ax, (tag, v) in zip(axes[0], items):
        grid = v["grid"]
        ctxs = sorted((int(c) for c in grid), key=int)
        depths = sorted((int(x) for x in next(iter(grid.values()))), key=int)
        Z = np.array([[grid[str(c)][str(dp)] for c in ctxs] for dp in depths], dtype=float)
        ax.imshow(Z, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(ctxs))); ax.set_xticklabels([f"{c//1024}k" for c in ctxs], fontsize=8)
        ax.set_yticks(range(len(depths))); ax.set_yticklabels([f"{dp}%" for dp in depths], fontsize=8)
        ax.set_title(pretty(tag, v.get("total_x", 1.0)), fontsize=10)
        ax.set_xlabel("context", fontsize=9)
    axes[0][0].set_ylabel("needle depth", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)

if __name__ == "__main__":
    main()
