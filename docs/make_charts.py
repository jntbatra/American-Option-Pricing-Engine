#!/usr/bin/env python3
"""Regenerate the README charts from the measured benchmark numbers.

Emits a light and a dark PNG for each figure so the README can serve the right
one via <picture> and prefers-color-scheme -- a single static image always looks
broken in one of GitHub's two themes.

Colours are the categorical slots validated for this project (see
docs/benchmark-results.html); each mode uses its own steps rather than a naive
inversion of the other.

    python3 docs/make_charts.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent / "img"
OUT.mkdir(exist_ok=True)

EXACT = 10.450584

N = [1_000, 10_000, 100_000, 1_000_000, 4_000_000]
ERR_LCG = [0.241814, 0.092363, 0.011733, 0.000757, 0.001834]
ERR_QMC = [-0.012560, -0.003931, -0.000280, 0.000014, -0.000017]

TIMES = {                      # milliseconds, median of 5 after a warmup
    "Serial":      [1.225, 5.567, 61.327, 798.255, 3199.784],
    "OpenMP":      [1.105, 2.187, 12.033, 143.755, 596.664],
    "OpenMP QMC":  [1.166, 3.015, 29.312, 293.253, 1163.046],
    "CUDA fp32":   [0.617, 0.632, 2.074, 15.398, 59.881],
    "CUDA QMC":    [0.777, 1.066, 5.205, 45.126, 178.905],
}

BIAS_M   = [1, 5, 9, 13, 17, 21]
BIAS_NAI = [10.4518, 13.3630, 14.7195, 15.4959, 15.9839, 16.3388]
BIAS_LSM = [10.4518, 10.4426, 10.4342, 10.4525, 10.4594, 10.4573]

SPEEDUP = [("Serial", 1.00), ("OpenMP QMC", 2.75), ("OpenMP", 5.36),
           ("CUDA QMC", 17.9), ("CUDA fp64", 19.9), ("CUDA fp32", 53.4)]

THEMES = {
    "light": dict(bg="#ffffff", fg="#14171c", muted="#5b6472", grid="#dde1e7",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]),
    "dark":  dict(bg="#0d1117", fg="#eaeef3", muted="#949daa", grid="#2b3038",
                  series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]),
}


def style(ax, t, xlabel, ylabel, title):
    ax.set_facecolor(t["bg"])
    ax.figure.patch.set_facecolor(t["bg"])
    ax.grid(True, color=t["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], labelsize=9)
    ax.set_xlabel(xlabel, color=t["muted"], fontsize=9.5)
    ax.set_ylabel(ylabel, color=t["muted"], fontsize=9.5)
    ax.set_title(title, color=t["fg"], fontsize=12, fontweight="bold",
                 loc="left", pad=12)


def save(fig, name, mode):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}-{mode}.png", dpi=170,
                facecolor=fig.get_facecolor())
    plt.close(fig)


def chart_bias(mode):
    t = THEMES[mode]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.axhline(0, color=t["fg"], lw=1.3, ls="--", alpha=0.55, zorder=1)
    ax.annotate("exact Black-Scholes price", (13.2, 0.30), color=t["muted"],
                fontsize=8.5)
    ax.plot(BIAS_M, [v - EXACT for v in BIAS_NAI], "-o", color=t["series"][1],
            lw=2, ms=5.5, mec=t["bg"], mew=1.6, label="Paper's recursion", zorder=3)
    ax.plot(BIAS_M, [v - EXACT for v in BIAS_LSM], "-o", color=t["series"][0],
            lw=2, ms=5.5, mec=t["bg"], mew=1.6, label="Longstaff-Schwartz", zorder=3)
    ax.set_ylim(-1, 6.6)
    ax.set_xticks(BIAS_M)               # m is a count; halves are meaningless
    style(ax, t, "Exercise dates  m", "Price error",
          "Perfect-foresight bias grows without bound; LSM does not")
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    for txt in leg.get_texts():
        txt.set_color(t["fg"])
    save(fig, "bias", mode)


def chart_convergence(mode):
    t = THEMES[mode]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ref = [abs(ERR_LCG[0]) * (N[0] / n) ** 0.5 for n in N]
    ax.loglog(N, ref, ls="--", lw=1.4, color=t["muted"], alpha=0.8,
              label=r"$N^{-1/2}$ reference", zorder=2)
    ax.loglog(N, [abs(e) for e in ERR_LCG], "-o", color=t["series"][0], lw=2,
              ms=5.5, mec=t["bg"], mew=1.6, label="LCG pseudo-random", zorder=3)
    ax.loglog(N, [abs(e) for e in ERR_QMC], "-o", color=t["series"][1], lw=2,
              ms=5.5, mec=t["bg"], mew=1.6, label="Sobol QMC + bridge", zorder=3)
    style(ax, t, "Paths  N", "|error| vs exact price",
          "QMC reaches 3-decimal accuracy 10x cheaper")
    leg = ax.legend(frameon=False, fontsize=9.5)
    for txt in leg.get_texts():
        txt.set_color(t["fg"])
    save(fig, "convergence", mode)


def chart_runtime(mode):
    t = THEMES[mode]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for i, (name, ys) in enumerate(TIMES.items()):
        ax.loglog(N, ys, "-o", color=t["series"][i], lw=2, ms=5,
                  mec=t["bg"], mew=1.5, label=name, zorder=3)
    style(ax, t, "Paths  N", "Wall-clock  ms",
          "Runtime, median of 5 runs after warmup")
    leg = ax.legend(frameon=False, fontsize=9, ncol=2)
    for txt in leg.get_texts():
        txt.set_color(t["fg"])
    save(fig, "runtime", mode)


def chart_speedup(mode):
    t = THEMES[mode]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    names = [n for n, _ in SPEEDUP]
    vals = [v for _, v in SPEEDUP]
    # One hue: this is magnitude of a single measure, not six categories.
    colors = [t["series"][0]] * len(vals)
    colors[-1] = t["series"][2]                    # the configuration shipped
    bars = ax.barh(names, vals, color=colors, height=0.62, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(v + 0.9, b.get_y() + b.get_height() / 2, f"{v:.2f}x",
                va="center", color=t["fg"], fontsize=9.5, fontweight="bold")
    ax.set_xlim(0, 62)
    style(ax, t, "Speedup vs serial", "",
          "Speedup at N = 4,000,000")
    ax.grid(axis="y", visible=False)
    save(fig, "speedup", mode)


if __name__ == "__main__":
    for mode in THEMES:
        chart_bias(mode)
        chart_convergence(mode)
        chart_runtime(mode)
        chart_speedup(mode)
    print("wrote", len(list(OUT.glob("*.png"))), "charts to", OUT)
