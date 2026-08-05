"""Pareto frontier: LongBench overall vs compression (Llama-3.1-8B). SPECTRA = full-sample (final);
baselines = 150-sample preview (will refresh when full-sample sweep completes)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = 30.63

# SPECTRA (full-sample, best variant per point): (comp, score, label)
spectra = [
    (3.56, 31.06, "WF b4"), (4.06, 30.41, "WF+B"), (4.57, 29.68, "WF b3"),
    (5.02, 29.85, "LR+B"), (5.69, 30.13, "LR+B"), (6.56, 30.29, "LR+B"),
    (7.11, 30.37, "LR+B"), (8.13, 29.98, "LR+B"), (9.48, 29.61, "LR+B"),
    (12.19, 27.11, "LR+B"),
]
# baselines: KIVI/Polar/Turbo/OTT = full-sample; RotateKV = 150-sample preview (full pending)
baselines = {
    "KIVI":       [(3.76, 30.77), (7.11, 27.52)],
    "PolarQuant": [(3.66, 31.07), (4.13, 29.99), (4.74, 28.95), (6.74, 26.85)],
    "TurboQuant": [(4.00, 29.99), (5.33, 27.99), (8.00, 23.60)],
    "OTT":        [(3.76, 30.64), (7.11, 28.42)],
    "RotateKV":   [(4.00, 29.17), (8.00, 24.46)],
}
markers = {"KIVI": "s", "PolarQuant": "^", "TurboQuant": "D", "OTT": "v", "RotateKV": "P"}
bcolors = {"KIVI": "tab:blue", "PolarQuant": "tab:green", "TurboQuant": "tab:orange",
           "OTT": "tab:purple", "RotateKV": "tab:brown"}

fig, ax = plt.subplots(figsize=(9, 6))
ax.axhline(BASE, ls="--", c="gray", lw=1, alpha=0.7, label=f"fp16 base ({BASE})")

# all points for a global Pareto frontier (higher comp AND higher score dominates)
allpts = [(c, s, "SPECTRA") for c, s, _ in spectra]
for m, pts in baselines.items():
    allpts += [(c, s, m) for c, s in pts]
front = []
for c, s, m in sorted(allpts, key=lambda p: (-p[0], -p[1])):
    if all(not (fc >= c and fs >= s and (fc, fs) != (c, s)) for fc, fs, _ in allpts):
        front.append((c, s))
front = sorted(set(front))
ax.plot([c for c, s in front], [s for c, s in front], c="crimson", lw=2.2, alpha=0.35,
        zorder=1, label="Pareto frontier")

sc = sorted(spectra)
ax.plot([c for c, s, _ in sc], [s for c, s, _ in sc], "-o", c="crimson", lw=2.4, ms=8,
        zorder=3, label="SPECTRA (ours, full-sample)")
for c, s, lab in spectra:
    ax.annotate(f"{c}x", (c, s), textcoords="offset points", xytext=(4, 6), fontsize=7, c="crimson")

for m, pts in baselines.items():
    p = sorted(pts)
    ax.plot([c for c, s in p], [s for c, s in p], "--", marker=markers[m], ms=8, lw=1.6,
            c=bcolors[m], alpha=0.85, zorder=2, label=m)

ax.set_xlabel("KV compression ratio (x, higher = smaller cache)")
ax.set_ylabel("LongBench overall (higher better)")
ax.set_title("KV compression — Llama-3.1-8B-Instruct (full LongBench), per-method curves\n"
             "(SPECTRA + KIVI/Polar/Turbo/OTT full-sample; RotateKV 150-sample preview)")
ax.grid(True, alpha=0.25)
ax.legend(loc="lower left", fontsize=8, ncol=2)
ax.set_xlim(3, 13); ax.set_ylim(23, 32)
fig.tight_layout()
out = "docs/story_telling/exp/pareto_full.png"
fig.savefig(out, dpi=140)
print("[saved]", out)
