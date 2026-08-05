"""Multi-needle (4-needle) NIAH heatmap: SPECTRA {dense,3.6,6.6,9.5,12.2}x + official KIVI/Turbo/Polar 2-bit.
Cell = fraction of 4 needles retrieved."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

wf = json.load(open("/tmp/mn/mn_wf.json"))
lrhi = json.load(open("/tmp/mn/mn_lrhi.json"))
lrmid = json.load(open("/tmp/mn/mn_lrmid.json"))
contexts = wf["contexts"]; depths = wf["depths"]
off = {t: json.load(open(f"/tmp/mn_off/{t}.json"))["acc"]
       for t in ["kivi_k2v2_g128_mn4", "turbo_k2v2_mn4", "polar_r2t2v4_mn4"]}

panels = [
    ("Dense (fp16)", wf["results"]["base"]["grid"], False),
    ("SPECTRA  3.6x", wf["results"]["B2048_b4"]["grid"], False),
    ("SPECTRA  6.6x", lrmid["results"]["B1536_b3"]["grid"], False),
    ("SPECTRA  9.5x", lrhi["results"]["B1024_b3"]["grid"], False),
    ("SPECTRA  12.2x", lrhi["results"]["B768_b3"]["grid"], False),
    ("KIVI 2-bit  7.1x", off["kivi_k2v2_g128_mn4"]["grid"], True),
    ("TurboQuant 2-bit  ~8x", off["turbo_k2v2_mn4"]["grid"], True),
    ("PolarQuant 2-bit  4.7x", off["polar_r2t2v4_mn4"]["grid"], True),
]


def grid_to_Z(g):
    Z = np.full((len(depths), len(contexts)), np.nan)
    for i, d in enumerate(depths):
        for j, c in enumerate(contexts):
            cell = g.get(str(c), {})
            v = cell.get(str(d), cell.get(d))
            if v is not None:
                Z[i, j] = v
    return Z


plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#888"})
fig, axes = plt.subplots(2, 4, figsize=(19, 8.4), facecolor="white", constrained_layout=True)
im = None
for ax, (title, g, is_bl) in zip(axes.flat, panels):
    Z = grid_to_Z(g)
    acc = np.nanmean(Z) * 100
    im = ax.imshow(Z, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(np.arange(-.5, len(contexts), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(depths), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.2)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(len(contexts)))
    ax.set_xticklabels([f"{c//1024}k" if c >= 1024 else str(c) for c in contexts], fontsize=8)
    ax.set_yticks(range(len(depths))); ax.set_yticklabels([f"{d}" for d in depths], fontsize=8)
    ax.set_xlabel("context length", fontsize=8, color="#444")
    ax.set_ylabel("first-needle depth (%)", fontsize=8, color="#444")
    tcol = "#8a1c2b" if is_bl else "#1a1a1a"
    ax.set_title(title, fontsize=12, fontweight="bold", color=tcol, pad=8)
    bcol = "#3a9d5d" if acc >= 97 else ("#e8a33d" if acc >= 80 else "#d1495b")
    ax.text(0.98, 1.02, f"{acc:.0f}%", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=11.5, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.3", fc=bcol, ec="none"))
    for s in ax.spines.values():
        s.set_visible(is_bl); s.set_color("#8a1c2b"); s.set_linewidth(2.2)

cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, pad=0.015)
cbar.set_label("fraction of 4 needles retrieved", fontsize=9)
fig.suptitle("Multi-needle (4) retrieval — Llama-3.1-8B-Instruct\n"
             "SPECTRA stays ~100% to 9.5x (reaches 12x); TurboQuant drops to 78% and PolarQuant to 47% "
             "under multi-needle pressure", fontsize=14, fontweight="bold")
out = "docs/story_telling/exp/niah_multi.png"
fig.savefig(out, dpi=140, facecolor="white", bbox_inches="tight")
print("[saved]", out)
