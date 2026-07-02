#!/usr/bin/env python3
"""
Generate the three README figures for the TOTO anomaly-detection pipeline.

  1. readme_step1_nll.png         - error scoring (NLL construction)
  2. readme_step2_aggregation.png - anomaly scoring (R^N -> R collapse)
  3. readme_step3_threshold.png   - unsupervised thresholding

Light / print styling. Notation:
  y_t          observed vector at step t        (subscript = time)
  y_t^{(j)}    sensor j at step t               (parenthesised superscript = dimension)
  e_t^{(j)} = -log p(y_t^{(j)} | y_<t)          per-sensor error (NLL)
  s_t = agg_j e_t^{(j)}                         scalar anomaly score
  a_t = 1[s_t > tau]                            binary verdict
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon, FancyBboxPatch, ConnectionPatch
from scipy.stats import t as student_t, norm, gaussian_kde
from pathlib import Path

OUT = Path("toto/anomaly_detection/figures")
OUT.mkdir(parents=True, exist_ok=True)

INK, BLUE, GREEN, CRIM, MUT, PAPER = "#1B2A4A", "#2F6DB5", "#2E8B6F", "#D1495B", "#7C8AA5", "#FFFFFF"
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "cm", "font.size": 11,
    "axes.edgecolor": "#33405A", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": "#33405A", "ytick.color": "#33405A", "axes.linewidth": 0.9,
})


def figure1():
    comps = [(0.70, 0.0, 0.95, 3), (0.30, 2.7, 0.72, 4)]   # weight, loc, scale, df
    xs = np.linspace(-4, 8, 900)
    mix = lambda x: sum(w * student_t.pdf(x, df, loc=l, scale=s) for (w, l, s, df) in comps)
    fx = mix(xs)
    mode = xs[np.argmax(fx)]; fmax = fx.max()
    obs = 5.1; p_obs = float(mix(obs)); nll = -np.log(p_obs)

    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    ax.fill_between(xs, fx, color=BLUE, alpha=0.12)
    ax.plot(xs, fx, color=BLUE, lw=2.3)
    ax.annotate("predicted distribution (Student-$t$ mixture)\npretrained TOTO · conditioned on $y_{<t}$",
                xy=(2.7, float(mix(2.7))), xytext=(1.7, 0.21), color=BLUE, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1))

    ax.axvline(mode, color=GREEN, lw=1.5, ls=(0, (4, 3)))
    ax.plot([mode], [fmax], "o", color=GREEN, ms=6)
    ax.annotate(r"forecast  $\hat{y}_t^{(j)}$" + "\nmax-probability sample",
                xy=(mode, fmax), xytext=(mode - 3.7, fmax * 0.80), color=GREEN, fontsize=10,
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=1))

    ax.plot([obs], [0], "o", color=CRIM, ms=7, clip_on=False, zorder=6)
    ax.plot([obs, obs], [0, p_obs], color=CRIM, lw=1.5, ls=":")
    ax.plot([obs], [p_obs], "o", color=CRIM, ms=5)
    ax.annotate(r"observed  $y_t^{(j)}$", xy=(obs, 0), xytext=(obs + 0.15, fmax * 0.28),
                color=CRIM, fontsize=10)
    ax.annotate(r"$p\,(y_t^{(j)})$", xy=(obs, p_obs), xytext=(obs + 0.8, p_obs + 0.035),
                color=CRIM, fontsize=10, arrowprops=dict(arrowstyle="->", color=CRIM, lw=1))

    ax.text(0.985, 0.96,
            r"$e_t^{(j)} = -\log\, p\,(y_t^{(j)}\mid y_{<t}) = %.2f$" % nll,
            transform=ax.transAxes, ha="right", va="top", fontsize=13,
            bbox=dict(boxstyle="round,pad=0.5", fc="#F3F6FB", ec="#C7D2E4"))

    ax.set_xlabel(r"value of sensor $j$ at step $t$"); ax.set_ylabel("probability density")
    ax.set_xlim(-4, 8); ax.set_ylim(0, fmax * 1.18)
    ax.set_title("Step 1 · Error scoring — the surprise (NLL) of the observed value under the predicted distribution",
                 fontsize=11.5, fontweight="bold", pad=12, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "readme_step1_nll.png", facecolor=PAPER, bbox_inches="tight")
    plt.close(fig)


def figure2():
    """Error matrix E = [e_t^(j)] -> per-column funnel collapse R^N -> R -> score row."""
    cols = [r"$t$", r"$t{+}1$", r"$t{+}2$", r"$t{+}3$"]
    vals = [["1.2", "0.4", "3.1", "0.7"],      # j = 1   (rows across the four timesteps)
            ["0.6", "5.6", "2.8", "0.5"],      # j = 2
            ["⋮",   "⋮",   "⋮",   "⋮"],        # elided rows
            ["0.9", "0.3", "4.0", "0.6"]]      # j = N
    cx = [5.0, 6.6, 8.2, 9.8]                  # column x-centres
    ry = [8.3, 7.6, 6.9, 6.2]                  # row y-centres
    rlab = [r"$j=1$", r"$j=2$", "⋮", r"$j=N$"]
    yt, yb, xl, xr = 8.62, 5.92, 4.5, 10.32    # bracket extent
    sft, hw = 0.13, 0.55                        # bracket serif, funnel half-width
    y_funnel_top, y_apex, y_score = yb, 4.05, 3.45

    fig, ax = plt.subplots(figsize=(9.4, 6.3))
    ax.set_xlim(0, 12.4); ax.set_ylim(2.0, 9.5); ax.axis("off")

    def line(p, q, **kw):
        ax.plot([p[0], q[0]], [p[1], q[1]], solid_capstyle="round", **kw)

    # matrix name + N brace
    ax.text(0.45, 7.35, "error matrix\n" + r"$E=[\,e_t^{(j)}\,]$", fontsize=13, color=INK,
            ha="left", va="center")
    xn = 3.15
    ax.annotate("", xy=(xn, yt + 0.05), xytext=(xn, yb - 0.05),
                arrowprops=dict(arrowstyle="<->", color=MUT, lw=1.3))
    ax.text(xn - 0.26, (yt + yb) / 2, r"$N$ sensors", fontsize=10.5, color=MUT,
            rotation=90, ha="center", va="center")

    # row labels
    for lab, y in zip(rlab, ry):
        ax.text(3.98, y, lab, fontsize=12, color=INK, ha="right", va="center")

    # brackets
    for x, s in ((xl, sft), (xr, -sft)):
        line((x, yb), (x, yt), color=INK, lw=1.8)
        line((x, yt), (x + s, yt), color=INK, lw=1.8)
        line((x, yb), (x + s, yb), color=INK, lw=1.8)

    # column headers + cells
    for k, x in enumerate(cx):
        ax.text(x, 8.98, cols[k], fontsize=12.5, color=BLUE, ha="center", va="center", fontweight="bold")
        for r, y in enumerate(ry):
            ax.text(x, y, vals[r][k], fontsize=12.5, color=INK, ha="center", va="center")

    # per-column funnel: diagonal bars converging R^N -> R
    for x in cx:
        ax.add_patch(Polygon([(x - hw, y_funnel_top), (x + hw, y_funnel_top), (x, y_apex)],
                             closed=True, facecolor=BLUE, alpha=0.08, edgecolor="none"))
        for f in np.linspace(-hw, hw, 5):
            line((x + f, y_funnel_top), (x, y_apex), color=BLUE, lw=0.8, alpha=0.55)
        # score box
        ax.add_patch(FancyBboxPatch((x - 0.42, y_score - 0.30), 0.84, 0.60,
                     boxstyle="round,pad=0.02,rounding_size=0.12",
                     fc="#EAF1FB", ec=BLUE, lw=1.2))
    for k, x in enumerate(cx):
        sub = ["t", "t{+}1", "t{+}2", "t{+}3"][k]
        ax.text(x, y_score, r"$s_{%s}$" % sub, fontsize=12.5, color=INK, ha="center", va="center")

    # compression caption (technique-agnostic) in the open lower-left
    ax.text(0.45, 4.75,
            "compress  " + r"$\mathbb{R}^{N}\!\rightarrow\mathbb{R}$" + "\n"
            r"$s_t=\mathrm{agg}\,(e_t^{(1)},\,\ldots,\,e_t^{(N)})$" + "\n\n"
            r"agg $\in$ {  mean,  max,  L2  }",
            fontsize=11, color=INK, ha="left", va="top")

    ax.text((cx[0] + cx[-1]) / 2, 2.62, r"one anomaly score per timestep   $s_t\in\mathbb{R}$",
            fontsize=11, color=MUT, ha="center", va="center")

    ax.set_title("Step 2 · Anomaly scoring — collapse $N$ sensor errors into one score per step   "
                 r"($\mathbb{R}^{N}\!\rightarrow\mathbb{R}$)",
                 fontsize=12.5, fontweight="bold", pad=10, loc="left")
    fig.savefig(OUT / "readme_step2_aggregation.png", facecolor=PAPER, bbox_inches="tight")
    plt.close(fig)


def figure3():
    rng = np.random.default_rng(3)
    T = 60
    anom = np.zeros(T, bool)
    for a, b in ((17, 24), (40, 45), (52, 53)):
        anom[a:b + 1] = True
    s = 2.5 + 1.0 * rng.random(T)
    s[anom] += 2.8 + 1.6 * rng.random(anom.sum())
    cal = np.exp(np.log(3.15) + 0.30 * rng.standard_normal(5000))   # calibration: normal only
    tau = float(np.percentile(cal, 95))

    fig = plt.figure(figsize=(9.9, 5.2))
    gs = GridSpec(1, 2, width_ratios=[1, 3.3], wspace=0.05)
    axc = fig.add_subplot(gs[0]); axt = fig.add_subplot(gs[1], sharey=axc)

    # ── ① FIT: calibration density (normal only) sets tau ──
    yy = np.linspace(0, 10, 400)
    dens = gaussian_kde(cal)(yy)
    axc.fill_betweenx(yy, 0, dens, color=BLUE, alpha=0.15)
    axc.plot(dens, yy, color=BLUE, lw=2)
    m = yy >= tau
    axc.fill_betweenx(yy[m], 0, dens[m], color=CRIM, alpha=0.22)
    axc.axhline(tau, color=CRIM, lw=1.6, ls=(0, (6, 4)))
    axc.annotate("top 5%", xy=(dens[m].max() * 0.95, tau + 0.85), fontsize=8.5,
                 color=CRIM, ha="center")
    axc.set_ylabel(r"anomaly score  $s_t$"); axc.set_xlabel("density")
    axc.set_xticks([]); axc.set_ylim(0, 10); axc.invert_xaxis()

    # ── ② TEST: apply tau to the evaluation stream ──
    tt = np.arange(T)
    axt.plot(tt, s, color="#5A6B82", lw=1.6, zorder=2)
    axt.axhline(tau, color=CRIM, lw=1.6, ls=(0, (6, 4)), zorder=1)
    flag = s > tau
    axt.fill_between(tt, tau, s, where=flag, color=CRIM, alpha=0.12)
    axt.scatter(tt[flag], s[flag], s=46, color=CRIM, zorder=3,
                label=r"flagged anomaly  $s_t>\tau$")
    axt.text(T - 0.6, tau + 0.24, r"$\tau$ = 95th percentile of calibration", ha="right",
             color=CRIM, fontsize=9.5)
    axt.text(0.015, 0.94, r"verdict  $a_t = \mathbb{1}[\,s_t > \tau\,]$",
             transform=axt.transAxes, va="top", fontsize=11)
    axt.set_xlabel(r"time  $t$"); axt.set_xlim(0, T - 1)
    axt.legend(loc="upper right", frameon=False, fontsize=9.5)

    # τ is ONE boundary: fit on the left, carried across and applied on the right
    bridge = ConnectionPatch(xyA=(0, tau), coordsA=axc.transData,
                             xyB=(0, tau), coordsB=axt.transData,
                             color=CRIM, lw=1.6, ls=(0, (6, 4)))
    fig.add_artist(bridge)

    for ax in (axc, axt):
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Step 3 · Thresholding — fit $\\tau$ on normal data, then flag $s_t>\\tau$ on unseen data",
                 fontsize=12.0, fontweight="bold", x=0.065, ha="left", y=1.10)
    cxl = sum(axc.get_position().intervalx) / 2
    cxr = sum(axt.get_position().intervalx) / 2
    fig.text(cxl, 0.92, "①  FIT\ncalibration (normal only)", ha="center", va="bottom",
             fontsize=10.5, color=INK, fontweight="bold", linespacing=1.5)
    fig.text(cxr, 0.92, "②  TEST\nevaluation (unseen data)", ha="center", va="bottom",
             fontsize=10.5, color=INK, fontweight="bold", linespacing=1.5)
    fig.savefig(OUT / "readme_step3_threshold.png", facecolor=PAPER, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    figure1(); print("wrote readme_step1_nll.png")
    figure2(); print("wrote readme_step2_aggregation.png")
    figure3(); print("wrote readme_step3_threshold.png")
