"""
chisq_gof_demo.py

Generates a random sample from a specified distribution, plots a histogram
overlaid with the fitted PDF, and performs a chi-square goodness-of-fit test
to illustrate how the test works.

Default distribution: Lognormal(mu=2.884, sigma=0.472)
  - mu    : mean of log(X)  [log-scale mean, i.e. scipy 'scale' = exp(mu)]
  - sigma : std  of log(X)  [log-scale std,  i.e. scipy 's' parameter]

Usage examples:
  python chisq_gof_demo.py
  python chisq_gof_demo.py --n 1000
  python chisq_gof_demo.py --n 200 --bins 15 --seed 99
  python chisq_gof_demo.py --dist exponential --rate 0.5
  python chisq_gof_demo.py --dist normal --mean 10 --std 2
  python chisq_gof_demo.py --out gof_plot.png
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from scipy.stats import chi2


# ---------------------------------------------------------------------------
# Distribution registry
# ---------------------------------------------------------------------------

def build_lognormal(mu: float, sigma: float) -> stats.rv_continuous:
    """Return a frozen lognormal distribution.  scipy convention:
       lognorm(s=sigma, scale=exp(mu))"""
    return stats.lognorm(s=sigma, scale=np.exp(mu))


def build_normal(mean: float, std: float) -> stats.rv_continuous:
    return stats.norm(loc=mean, scale=std)


def build_exponential(rate: float) -> stats.rv_continuous:
    """scipy uses scale = 1/rate."""
    return stats.expon(scale=1.0 / rate)


# ---------------------------------------------------------------------------
# Chi-square goodness-of-fit
# ---------------------------------------------------------------------------

def chisq_gof(sample: np.ndarray,
              dist: stats.rv_continuous,
              n_params_estimated: int,
              n_bins: int) -> dict:
    """
    Compute a chi-square GoF statistic against a *fully specified* frozen
    distribution (parameters are treated as known, not estimated from the
    same sample).

    Bins are chosen so that each bin has equal expected probability under the
    null distribution (equiprobable binning), which is the standard approach
    for continuous data.

    Parameters
    ----------
    sample              : observed data
    dist                : frozen scipy distribution (the null hypothesis)
    n_params_estimated  : number of distribution parameters estimated from
                          data (used to adjust degrees of freedom)
    n_bins              : number of bins (expected count per bin = n / n_bins)

    Returns
    -------
    dict with keys: bins, edges, observed, expected, chi2_stat, df, p_value
    """
    n = len(sample)

    # Equal-probability bin edges via the quantile function
    probs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = dist.ppf(probs)

    # Clip to finite range (handles theoretical -inf / +inf tails)
    edges[0]  = min(edges[0],  sample.min())
    edges[-1] = max(edges[-1], sample.max())

    observed, _ = np.histogram(sample, bins=edges)
    expected = np.full(n_bins, n / n_bins, dtype=float)

    # Degrees of freedom: (bins - 1) - params_estimated
    df = (n_bins - 1) - n_params_estimated
    if df < 1:
        df = 1  # safeguard

    chi2_stat = np.sum((observed - expected) ** 2 / expected)
    p_value   = 1.0 - chi2.cdf(chi2_stat, df)

    return dict(
        n_bins=n_bins,
        edges=edges,
        observed=observed,
        expected=expected,
        chi2_stat=chi2_stat,
        df=df,
        p_value=p_value,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_gof(sample: np.ndarray,
             dist: stats.rv_continuous,
             gof: dict,
             dist_label: str,
             n_hist_bins: int,
             out_path: str | None) -> None:
    """
    Two-panel figure:
      Top   : histogram of sample + PDF overlay
      Bottom: observed vs expected counts per GoF bin
    """
    fig, axes = plt.subplots(
        2, 1,
        figsize=(9, 8),
        gridspec_kw={"height_ratios": [3, 2]},
    )
    fig.subplots_adjust(hspace=0.45)

    # ---- Panel 1: histogram + PDF ----------------------------------------
    ax1 = axes[0]

    # Use more bins for the visual histogram (smoother look)
    counts, bin_edges, patches = ax1.hist(
        sample,
        bins=n_hist_bins,
        density=True,
        color="#4C72B0",
        alpha=0.55,
        edgecolor="white",
        linewidth=0.6,
        label=f"Sample histogram  (n={len(sample):,})",
        zorder=2,
    )

    # PDF curve
    x_lo = dist.ppf(0.001)
    x_hi = dist.ppf(0.999)
    x_lo = min(x_lo, sample.min())
    x_hi = max(x_hi, sample.max())
    xs = np.linspace(x_lo, x_hi, 500)
    ax1.plot(xs, dist.pdf(xs), color="#C44E52", linewidth=2.2,
             label=f"Theoretical PDF\n{dist_label}", zorder=3)

    # Mark GoF bin edges as vertical dashed lines
    for edge in gof["edges"][1:-1]:
        ax1.axvline(edge, color="#555555", linewidth=0.7,
                    linestyle="--", alpha=0.6, zorder=1)

    ax1.set_xlabel("Value", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title("Sample Histogram with Theoretical PDF Overlay", fontsize=13, pad=10)
    ax1.legend(fontsize=9, framealpha=0.85)

    # Annotation: chi-square result
    result_text = (
        f"Chi-square GoF test\n"
        f"  Bins  : {gof['n_bins']}\n"
        f"  Stat  : {gof['chi2_stat']:.3f}\n"
        f"  df    : {gof['df']}\n"
        f"  p     : {gof['p_value']:.4f}"
    )
    ax1.text(
        0.98, 0.97, result_text,
        transform=ax1.transAxes,
        va="top", ha="right",
        fontsize=8.5,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                  edgecolor="#aaaaaa", alpha=0.9),
    )

    # ---- Panel 2: observed vs expected per GoF bin -----------------------
    ax2 = axes[1]

    # Replace any remaining inf edges with sample-based finite values for display
    plot_edges = gof["edges"].copy()
    plot_edges[0]  = max(plot_edges[0],  sample.min())
    plot_edges[-1] = min(plot_edges[-1], sample.max())

    bin_centers = 0.5 * (plot_edges[:-1] + plot_edges[1:])
    bin_widths  = np.diff(plot_edges) * 0.38   # narrow bars so both sets show

    ax2.bar(bin_centers - bin_widths / 2, gof["observed"],
            width=bin_widths, color="#4C72B0", alpha=0.75,
            label="Observed", edgecolor="white", linewidth=0.5)
    ax2.bar(bin_centers + bin_widths / 2, gof["expected"],
            width=bin_widths, color="#C44E52", alpha=0.75,
            label="Expected", edgecolor="white", linewidth=0.5)

    ax2.set_xlabel("Bin midpoint (equiprobable bins)", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Chi-Square GoF: Observed vs Expected Counts per Bin", fontsize=12)
    ax2.legend(fontsize=9, framealpha=0.85)

    # Draw chi-square contribution as text above each bar pair
    for i, (obs, exp) in enumerate(zip(gof["observed"], gof["expected"])):
        contrib = (obs - exp) ** 2 / exp
        ax2.text(bin_centers[i], max(obs, exp) + 1, f"{contrib:.2f}",
                 ha="center", va="bottom", fontsize=6.5, color="#333333")

    plt.suptitle("Chi-Square Goodness-of-Fit Demonstration", fontsize=14,
                 fontweight="bold", y=1.01)

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {out_path}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

SEPARATOR = "-" * 60

def print_report(sample: np.ndarray,
                 dist: stats.rv_continuous,
                 gof: dict,
                 dist_label: str,
                 alpha: float = 0.05) -> None:
    n = len(sample)
    print(SEPARATOR)
    print("CHI-SQUARE GOODNESS-OF-FIT REPORT")
    print(SEPARATOR)
    print(f"Distribution   : {dist_label}")
    print(f"Sample size    : {n:,}")
    print()
    print("Descriptive Statistics (sample)")
    print(f"  Mean         : {sample.mean():.4f}")
    print(f"  Std dev      : {sample.std(ddof=1):.4f}")
    print(f"  Min          : {sample.min():.4f}")
    print(f"  Median       : {np.median(sample):.4f}")
    print(f"  Max          : {sample.max():.4f}")
    print()
    print("GoF Test (equiprobable bins)")
    print(f"  Number of bins  : {gof['n_bins']}")
    print(f"  Expected / bin  : {gof['expected'][0]:.1f}")
    print(f"  Chi-square stat : {gof['chi2_stat']:.4f}")
    print(f"  Degrees freedom : {gof['df']}")
    print(f"  p-value         : {gof['p_value']:.4f}")
    print()

    # Per-bin table
    print(f"  {'Bin':>4}  {'Lower':>10}  {'Upper':>10}  "
          f"{'Obs':>6}  {'Exp':>6}  {'(O-E)^2/E':>10}")
    print("  " + "-" * 56)
    for i in range(gof["n_bins"]):
        lo  = gof["edges"][i]
        hi  = gof["edges"][i + 1]
        obs = gof["observed"][i]
        exp = gof["expected"][i]
        con = (obs - exp) ** 2 / exp
        print(f"  {i+1:>4}  {lo:>10.4f}  {hi:>10.4f}  "
              f"{obs:>6d}  {exp:>6.1f}  {con:>10.4f}")

    print(SEPARATOR)
    decision = "FAIL TO REJECT" if gof["p_value"] >= alpha else "REJECT"
    print(f"At alpha={alpha}: {decision} the null hypothesis")
    print(f"  (H0: data follow {dist_label})")
    print(SEPARATOR)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Chi-square GoF demo: sample from a distribution and test fit.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Sample
    p.add_argument("--n",    type=int,   default=500,
                   help="Sample size (default: 500)")
    p.add_argument("--seed", type=int,   default=None,
                   help="Random seed for reproducibility")
    p.add_argument("--bins", type=int,   default=20,
                   help="Histogram display bins (default: 20)")
    p.add_argument("--gof-bins", type=int, default=10,
                   help="Number of equiprobable bins for chi-square test (default: 10)")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Significance level for hypothesis test (default: 0.05)")
    p.add_argument("--out",  type=str,   default=None,
                   help="Output file path for the plot (e.g. plot.png). "
                        "If omitted, the plot is displayed interactively.")

    # Distribution selection
    p.add_argument("--dist", choices=["lognormal", "normal", "exponential"],
                   default="lognormal",
                   help="Distribution to sample from (default: lognormal)")

    # Lognormal params
    p.add_argument("--mu",    type=float, default=2.884,
                   help="Lognormal: log-scale mean mu (default: 2.884)")
    p.add_argument("--sigma", type=float, default=0.472,
                   help="Lognormal: log-scale std sigma (default: 0.472)")

    # Normal params
    p.add_argument("--mean",  type=float, default=0.0,
                   help="Normal: mean (default: 0.0)")
    p.add_argument("--std",   type=float, default=1.0,
                   help="Normal: standard deviation (default: 1.0)")

    # Exponential params
    p.add_argument("--rate",  type=float, default=1.0,
                   help="Exponential: rate lambda = 1/mean (default: 1.0)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Validate GoF bins: expected count should be >= 5
    expected_per_bin = args.n / args.gof_bins
    if expected_per_bin < 5:
        print(f"WARNING: expected count per bin ({expected_per_bin:.1f}) is below 5. "
              f"Consider reducing --gof-bins or increasing --n.", file=sys.stderr)

    # Build frozen distribution
    if args.dist == "lognormal":
        if args.sigma <= 0:
            sys.exit("ERROR: --sigma must be > 0 for lognormal distribution.")
        dist = build_lognormal(args.mu, args.sigma)
        dist_label = f"Lognormal(mu={args.mu}, sigma={args.sigma})"
        n_params = 2
    elif args.dist == "normal":
        if args.std <= 0:
            sys.exit("ERROR: --std must be > 0 for normal distribution.")
        dist = build_normal(args.mean, args.std)
        dist_label = f"Normal(mean={args.mean}, std={args.std})"
        n_params = 2
    elif args.dist == "exponential":
        if args.rate <= 0:
            sys.exit("ERROR: --rate must be > 0 for exponential distribution.")
        dist = build_exponential(args.rate)
        dist_label = f"Exponential(rate={args.rate})"
        n_params = 1
    else:
        sys.exit(f"Unknown distribution: {args.dist}")

    # Generate sample
    rng = np.random.default_rng(args.seed)
    sample = dist.rvs(size=args.n, random_state=rng)

    # Chi-square GoF
    # n_params_estimated=0 because the distribution is fully specified
    # (we are testing the exact parameterized distribution, not one fit to data)
    gof = chisq_gof(sample, dist,
                    n_params_estimated=0,
                    n_bins=args.gof_bins)

    # Report to console
    print_report(sample, dist, gof, dist_label, alpha=args.alpha)

    # Plot
    plot_gof(sample, dist, gof, dist_label,
             n_hist_bins=args.bins, out_path=args.out)


if __name__ == "__main__":
    main()
