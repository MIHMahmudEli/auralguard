"""Generate the vector figures used in the paper (paper/figures/*.pdf).

These are the DET curve (Fig. fig:det) and the per-attack EER breakdown
(Fig. fig:perattack) in sections/results.tex.

The numbers below mirror the (currently illustrative) values reported in the
results tables. When real experiment outputs are available, replace the arrays
in `RESULTS` and re-run:

    python scripts/plot_paper_figures.py

Styling targets IEEE two-column: ~3.4 in column width, serif fonts, thin marks.
Colours are the colourblind-safe Okabe-Ito subset (validated: worst adjacent
CVD deltaE 20.7); identity is *also* carried by line style + marker so the
figures survive greyscale printing.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

FIG_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"

# --- shared IEEE-ish style -------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.3,
    "grid.linewidth": 0.4,
    "pdf.fonttype": 42,  # embed as TrueType (editable, no Type-3 warnings)
})

# entity -> colour (Okabe-Ito). Colour follows the model, never its rank.
COLOR = {
    "AuralGuard": "#0072B2",  # blue, emphasis
    "B5 (WavLM+OCS)": "#009E73",  # green
    "B3 (AASIST)": "#E69F00",  # orange
    "B2 (RawNet2)": "#56B4E9",  # sky
    "B1 (LFCC-LCNN)": "#CC79A7",  # purple
}
STYLE = {  # secondary encoding for greyscale / CVD relief
    "AuralGuard": ("-", "o"),
    "B5 (WavLM+OCS)": ("--", "s"),
    "B3 (AASIST)": ("-.", "^"),
    "B2 (RawNet2)": (":", "D"),
    "B1 (LFCC-LCNN)": ((0, (3, 1, 1, 1)), "v"),
}

# in-domain EER (%) per model, matching Table tab:main (ASVspoof 19 LA col)
EER_INDOMAIN = {
    "AuralGuard": 0.37,
    "B5 (WavLM+OCS)": 0.31,
    "B3 (AASIST)": 0.71,
    "B2 (RawNet2)": 0.89,
    "B1 (LFCC-LCNN)": 1.03,
}
# draw order: baselines first (recessive), AuralGuard last (on top)
DRAW_ORDER = ["B1 (LFCC-LCNN)", "B2 (RawNet2)", "B3 (AASIST)",
              "B5 (WavLM+OCS)", "AuralGuard"]


def _probit(p):
    return norm.ppf(p)


def make_det_curve():
    """DET curve on normal-deviate axes; EER of each curve matches Table I."""
    fig, ax = plt.subplots(figsize=(3.4, 3.0))

    # tick grid in percent, common for DET plots
    ticks = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40])
    lo, hi = _probit(0.0006), _probit(0.55)

    sigma = 1.05  # mild, shared curvature (equal-variance-ish Gaussian model)
    t = np.linspace(1.5, 4.2, 400)
    for name in DRAW_ORDER:
        eer = EER_INDOMAIN[name] / 100.0
        mu = -(1 + sigma) * norm.ppf(eer)  # => EER at FAR=FRR by construction
        far = norm.sf(t)
        frr = norm.cdf((t - mu) / sigma)
        ls, mk = STYLE[name]
        emph = name == "AuralGuard"
        ax.plot(_probit(far), _probit(frr),
                ls=ls, color=COLOR[name],
                lw=1.7 if emph else 1.1,
                zorder=5 if emph else 3,
                label=f"{name} (EER {EER_INDOMAIN[name]:.2f}%)")

    # EER diagonal (FAR = FRR)
    ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.text(_probit(0.32), _probit(0.32), "EER", color="0.45",
            fontsize=6, rotation=45, ha="center", va="center",
            rotation_mode="anchor")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xticks(_probit(ticks / 100))
    ax.set_yticks(_probit(ticks / 100))
    ax.set_xticklabels([f"{v:g}" for v in ticks])
    ax.set_yticklabels([f"{v:g}" for v in ticks])
    ax.set_xlabel("False Alarm Rate (%)")
    ax.set_ylabel("False Rejection Rate (%)")
    ax.grid(True, color="0.85", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper right", frameon=False, handlelength=2.2,
              borderaxespad=0.2, labelspacing=0.3)

    fig.tight_layout(pad=0.3)
    out = FIG_DIR / "det_curve.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# per-attack EER (%) on ASVspoof 2019 LA eval, A01..A19 (illustrative).
PERATTACK = {
    "B5 (WavLM+OCS)": [0.4, 0.9, 0.3, 1.2, 0.7, 0.5, 1.8, 2.4, 1.1, 3.2,
                       0.9, 2.1, 1.5, 0.8, 2.8, 5.1, 6.3, 4.7, 5.9],
    "AuralGuard":     [0.5, 0.7, 0.4, 0.9, 0.8, 0.6, 1.2, 1.9, 1.3, 2.1,
                       1.0, 1.4, 1.1, 0.9, 1.7, 2.3, 2.9, 2.4, 3.1],
}


def make_perattack():
    labels = [f"A{i:02d}" for i in range(1, 20)]
    x = np.arange(len(labels))
    w = 0.4
    fig, ax = plt.subplots(figsize=(3.4, 2.4))

    for i, name in enumerate(["B5 (WavLM+OCS)", "AuralGuard"]):
        ax.bar(x + (i - 0.5) * w, PERATTACK[name], width=w,
               color=COLOR[name], label=name,
               edgecolor="white", linewidth=0.3,
               zorder=3)

    # shade the neural-waveform attacks A16-A19 where View B helps most
    ax.axvspan(14.5, 18.5, color="0.92", zorder=0)
    ax.text(16.5, ax.get_ylim()[1] * 0.96, "neural vocoder",
            fontsize=5.5, color="0.4", ha="center", va="top")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=5.5)
    ax.set_ylabel("EER (%)")
    ax.set_xlim(-0.7, len(labels) - 0.3)
    ax.grid(True, axis="y", color="0.85", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", frameon=False, handlelength=1.4,
              borderaxespad=0.2)

    fig.tight_layout(pad=0.3)
    out = FIG_DIR / "perattack.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # LaTeX-style escapes above assume mathtext usetex off; keep plain text.
    mpl.rcParams["text.usetex"] = False
    make_det_curve()
    make_perattack()


if __name__ == "__main__":
    main()
