"""
plot_lognormal.py

Generate PDF and CDF plots for a lognormal distribution.

Usage:
    python plot_lognormal.py --mu 0.0 --sigma 1.0
    python plot_lognormal.py --mu 1.5 --sigma 0.5 --percentile 99 --out my_plot.png
    python plot_lognormal.py --mu 1.0 --sigma 0.6 --no-reflines
    python plot_lognormal.py --mu 1.0 --sigma 0.6 --no-annotation
    python plot_lognormal.py --mu 1.0 --sigma 0.6 --no-reflines --no-annotation

Parameters (lognormal parameterization):
    mu    : mean of the underlying normal (log-scale mean)
    sigma : std dev of the underlying normal (log-scale std dev, must be > 0)

Overlay controls:
    --no-reflines   : suppress mean/median/mode vertical reference lines on both panels
    --no-annotation : suppress the stats call-out box on the PDF panel

The script displays the plot interactively and optionally saves it to a file.
"""

import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import lognorm


def build_x_range(mu: float, sigma: float, percentile: float) -> np.ndarray:
    """Return an x array from near-zero to the given upper percentile of the distribution."""
    dist = lognorm(s=sigma, scale=np.exp(mu))
    x_max = dist.ppf(percentile / 100.0)
    # Start just above 0; lognormal is undefined at 0
    x_start = dist.ppf(0.001)
    return np.linspace(x_start, x_max, 2000)


def compute_stats(mu: float, sigma: float) -> dict:
    """Return key descriptive statistics for the lognormal(mu, sigma) distribution."""
    dist = lognorm(s=sigma, scale=np.exp(mu))
    mean = dist.mean()
    median = dist.median()
    mode = np.exp(mu - sigma**2)         # mode = exp(mu - sigma^2)
    std = dist.std()
    skewness = dist.stats(moments="s")
    kurt = dist.stats(moments="k")       # excess kurtosis
    return dict(mean=mean, median=median, mode=mode, std=std,
                skewness=float(skewness), excess_kurtosis=float(kurt))


def make_annotation(stats: dict) -> str:
    lines = [
        f"Mean          = {stats['mean']:.4g}",
        f"Median        = {stats['median']:.4g}",
        f"Mode          = {stats['mode']:.4g}",
        f"Std Dev       = {stats['std']:.4g}",
        f"Skewness      = {stats['skewness']:.4g}",
        f"Excess Kurt.  = {stats['excess_kurtosis']:.4g}",
    ]
    return "\n".join(lines)


def plot_lognormal(
    mu: float,
    sigma: float,
    percentile: float,
    out: str | None,
    show_reflines: bool = True,
    show_annotation: bool = True,
) -> None:
    dist = lognorm(s=sigma, scale=np.exp(mu))
    x = build_x_range(mu, sigma, percentile)
    pdf = dist.pdf(x)
    cdf = dist.cdf(x)
    stats = compute_stats(mu, sigma)

    fig, (ax_pdf, ax_cdf) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True,
        constrained_layout=True,
    )

    title = f"Lognormal Distribution  (mu={mu}, sigma={sigma})"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # --- PDF panel ---
    ax_pdf.plot(x, pdf, color="steelblue", linewidth=2, label="PDF")
    ax_pdf.fill_between(x, pdf, alpha=0.15, color="steelblue")

    # Vertical reference lines
    ref_lines = [
        ("Mean",   stats["mean"],   "firebrick",   "--"),
        ("Median", stats["median"], "darkorange",  "-."),
        ("Mode",   stats["mode"],   "seagreen",    ":"),
    ]
    if show_reflines:
        for label, val, color, ls in ref_lines:
            ax_pdf.axvline(val, color=color, linestyle=ls, linewidth=1.4,
                           label=f"{label} = {val:.4g}")

    ax_pdf.set_ylabel("Probability Density", fontsize=10)
    ax_pdf.legend(fontsize=8, loc="upper right")
    ax_pdf.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4g"))
    ax_pdf.grid(True, linestyle=":", alpha=0.5)

    # Stats annotation box (PDF panel)
    if show_annotation:
        ann_text = make_annotation(stats)
        ax_pdf.text(
            0.02, 0.97, ann_text,
            transform=ax_pdf.transAxes,
            fontsize=7.5, verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8, edgecolor="gray"),
        )

    # --- CDF panel ---
    ax_cdf.plot(x, cdf, color="darkorchid", linewidth=2, label="CDF")

    if show_reflines:
        for label, val, color, ls in ref_lines:
            ax_cdf.axvline(val, color=color, linestyle=ls, linewidth=1.4)

    ax_cdf.axhline(0.5, color="gray", linestyle=":", linewidth=1.0, label="CDF = 0.50")
    ax_cdf.set_ylabel("Cumulative Probability", fontsize=10)
    ax_cdf.set_xlabel("x", fontsize=10)
    ax_cdf.set_ylim(0, 1.05)
    ax_cdf.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax_cdf.legend(fontsize=8, loc="lower right")
    ax_cdf.grid(True, linestyle=":", alpha=0.5)

    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {out}")

    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PDF and CDF of a lognormal distribution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mu",    type=float, default=0.0,
                        help="Log-scale mean (mu) of the lognormal distribution.")
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="Log-scale std dev (sigma); must be > 0.")
    parser.add_argument("--percentile", type=float, default=99.5,
                        help="Upper percentile cutoff for the x-axis (e.g. 99.5).")
    parser.add_argument("--out",   type=str,   default=None,
                        help="Optional output file path (e.g. plot.png, plot.pdf).")
    parser.add_argument("--no-reflines", action="store_true",
                        help="Suppress mean/median/mode vertical reference lines on both panels.")
    parser.add_argument("--no-annotation", action="store_true",
                        help="Suppress the stats call-out box on the PDF panel.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.sigma <= 0:
        print("ERROR: sigma must be > 0.", file=sys.stderr)
        sys.exit(1)
    if not (50.0 < args.percentile < 100.0):
        print("ERROR: --percentile must be in (50, 100).", file=sys.stderr)
        sys.exit(1)

    plot_lognormal(
        mu=args.mu,
        sigma=args.sigma,
        percentile=args.percentile,
        out=args.out,
        show_reflines=not args.no_reflines,
        show_annotation=not args.no_annotation,
    )


if __name__ == "__main__":
    main()
