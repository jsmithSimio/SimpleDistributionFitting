"""
ks_gof_demo.py

Generates a random sample from a specified distribution and performs a
Kolmogorov-Smirnov (KS) goodness-of-fit test, illustrating how the test
statistic D is derived from the maximum vertical gap between the empirical
CDF (ECDF) and the theoretical CDF.

Default distribution: Lognormal(mu=2.884, sigma=0.472)

The KS test statistic is:
    D = max over all x of |F_n(x) - F(x)|

where F_n(x) is the empirical CDF (a step function) and F(x) is the
theoretical CDF being tested.

Usage examples:
  python ks_gof_demo.py
  python ks_gof_demo.py --n 1000 --seed 7
  python ks_gof_demo.py --dist normal --mean 10 --std 2
  python ks_gof_demo.py --dist exponential --rate 0.5
  python ks_gof_demo.py --n 200 --out ks_plot.png
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy import stats


# ---------------------------------------------------------------------------
# Distribution registry  (same as chisq_gof_demo.py)
# ---------------------------------------------------------------------------

def build_lognormal(mu: float, sigma: float) -> stats.rv_continuous:
    return stats.lognorm(s=sigma, scale=np.exp(mu))

def build_normal(mean: float, std: float) -> stats.rv_continuous:
    return stats.norm(loc=mean, scale=std)

def build_exponential(rate: float) -> stats.rv_continuous:
    return stats.expon(scale=1.0 / rate)


# ---------------------------------------------------------------------------
# KS test internals
# ---------------------------------------------------------------------------

def ks_test(sample: np.ndarray,
            dist: stats.rv_continuous) -> dict:
    """
    Compute the one-sample KS statistic manually so we can expose the full
    geometry (ECDF values, theoretical CDF values, and the location / size
    of D) for plotting.

    The KS statistic is:
        D = max_i { |F_n(x_i) - F(x_i)|,  |F_n(x_{i-1}) - F(x_i)| }

    evaluated at every order statistic x_(i).  The second term captures the
    *pre-step* gap (just before the ECDF jumps).

    Returns
    -------
    dict with keys:
      x_sorted      : sorted sample
      ecdf_post     : ECDF value just AFTER each jump  (i/n)
      ecdf_pre      : ECDF value just BEFORE each jump ((i-1)/n)
      cdf_vals      : theoretical CDF at each order statistic
      d_plus        : max( F_n(x_i) - F(x_i) )   upper gaps
      d_minus       : max( F(x_i) - F_n(x_{i-1}) ) lower gaps
      d_stat        : max(d_plus, d_minus) = KS statistic
      d_loc         : x-value where D is attained
      d_ecdf        : ECDF value at D location
      d_cdf         : CDF  value at D location
      p_value       : two-sided p-value (scipy)
    """
    n = len(sample)
    x = np.sort(sample)

    cdf_vals  = dist.cdf(x)
    ecdf_post = np.arange(1, n + 1) / n        # F_n just after the jump
    ecdf_pre  = np.arange(0, n)     / n        # F_n just before the jump

    # Upper one-sided: ECDF above CDF
    upper_gaps = ecdf_post - cdf_vals
    # Lower one-sided: CDF above ECDF (use pre-jump ECDF value)
    lower_gaps = cdf_vals  - ecdf_pre

    d_plus  = upper_gaps.max()
    d_minus = lower_gaps.max()
    d_stat  = max(d_plus, d_minus)

    # Locate where D is attained
    if d_plus >= d_minus:
        idx      = upper_gaps.argmax()
        d_ecdf   = ecdf_post[idx]
        d_cdf    = cdf_vals[idx]
    else:
        idx      = lower_gaps.argmax()
        d_ecdf   = ecdf_pre[idx]
        d_cdf    = cdf_vals[idx]

    d_loc = x[idx]

    # scipy KS test for the p-value
    ks_result = stats.kstest(sample, dist.cdf)
    p_value   = ks_result.pvalue

    return dict(
        x_sorted  = x,
        ecdf_post = ecdf_post,
        ecdf_pre  = ecdf_pre,
        cdf_vals  = cdf_vals,
        d_plus    = d_plus,
        d_minus   = d_minus,
        d_stat    = d_stat,
        d_loc     = d_loc,
        d_ecdf    = d_ecdf,
        d_cdf     = d_cdf,
        p_value   = p_value,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ks(sample: np.ndarray,
            dist: stats.rv_continuous,
            ks: dict,
            dist_label: str,
            out_path: str | None) -> None:
    """
    Three-panel figure:
      Top    : ECDF vs theoretical CDF, with D annotated
      Middle : Gap function  delta(x) = F_n(x) - F(x)  showing where D lands
      Bottom : PDF with sample rug for context
    """
    fig, axes = plt.subplots(
        3, 1,
        figsize=(9, 10),
        gridspec_kw={"height_ratios": [3, 2, 2]},
    )
    fig.subplots_adjust(hspace=0.50)

    x_lo = dist.ppf(0.001)
    x_hi = dist.ppf(0.999)
    x_lo = min(x_lo, sample.min())
    x_hi = max(x_hi, sample.max())
    xs   = np.linspace(x_lo, x_hi, 800)

    # ---- Panel 1: ECDF vs CDF --------------------------------------------
    ax1 = axes[0]

    # Theoretical CDF (smooth)
    ax1.plot(xs, dist.cdf(xs), color="#C44E52", linewidth=2.2,
             label=f"Theoretical CDF\n{dist_label}", zorder=3)

    # ECDF as a step function
    # Build explicit step-function coordinates for a clean plot
    n = len(sample)
    x_step = np.repeat(ks["x_sorted"], 2)
    y_step = np.zeros(2 * n)
    y_step[0::2] = ks["ecdf_pre"]
    y_step[1::2] = ks["ecdf_post"]

    # Prepend the leading flat segment at y=0
    x_step = np.concatenate([[x_lo], x_step])
    y_step = np.concatenate([[0.0],  y_step])
    # Append trailing flat segment at y=1
    x_step = np.concatenate([x_step, [x_hi]])
    y_step = np.concatenate([y_step, [1.0]])

    ax1.plot(x_step, y_step, color="#4C72B0", linewidth=1.6,
             label=f"ECDF  (n={n:,})", zorder=2)

    # Mark D on the plot
    d_x    = ks["d_loc"]
    d_lo   = min(ks["d_ecdf"], ks["d_cdf"])
    d_hi   = max(ks["d_ecdf"], ks["d_cdf"])

    ax1.annotate(
        "",
        xy=(d_x, d_hi), xytext=(d_x, d_lo),
        arrowprops=dict(
            arrowstyle="<->",
            color="#2CA02C",
            lw=2.0,
        ),
        zorder=5,
    )
    ax1.text(
        d_x + (x_hi - x_lo) * 0.015,
        0.5 * (d_lo + d_hi),
        f"D = {ks['d_stat']:.4f}",
        color="#2CA02C", fontsize=9, va="center", fontweight="bold",
    )
    ax1.axvline(d_x, color="#2CA02C", linewidth=0.8, linestyle=":", alpha=0.7)

    # Result box
    result_text = (
        f"KS test\n"
        f"  D     : {ks['d_stat']:.4f}\n"
        f"  D+    : {ks['d_plus']:.4f}\n"
        f"  D-    : {ks['d_minus']:.4f}\n"
        f"  p     : {ks['p_value']:.4f}"
    )
    ax1.text(
        0.02, 0.97, result_text,
        transform=ax1.transAxes, va="top", ha="left",
        fontsize=8.5, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                  edgecolor="#aaaaaa", alpha=0.9),
    )

    ax1.set_xlabel("Value", fontsize=11)
    ax1.set_ylabel("Cumulative Probability", fontsize=11)
    ax1.set_title("ECDF vs Theoretical CDF  (KS test)", fontsize=13, pad=8)
    ax1.legend(fontsize=9, loc="lower right", framealpha=0.85)
    ax1.set_ylim(-0.03, 1.08)

    # ---- Panel 2: Gap function  delta(x) = F_n(x) - F(x) ---------------
    ax2 = axes[1]

    # At each order statistic, the gap function has two values:
    # just before the jump (ecdf_pre - cdf) and just after (ecdf_post - cdf)
    gap_pre  = ks["ecdf_pre"]  - ks["cdf_vals"]   # lower one-sided gaps (negated)
    gap_post = ks["ecdf_post"] - ks["cdf_vals"]   # upper one-sided gaps

    # Build the same step-function structure for the gap
    gap_x = np.repeat(ks["x_sorted"], 2)
    gap_y = np.zeros(2 * n)
    gap_y[0::2] = gap_pre
    gap_y[1::2] = gap_post
    gap_x = np.concatenate([[x_lo], gap_x, [x_hi]])
    gap_y = np.concatenate([[0.0 - dist.cdf(x_lo)], gap_y,
                             [1.0 - dist.cdf(x_hi)]])

    ax2.plot(gap_x, gap_y, color="#8172B2", linewidth=1.4,
             label=r"$\Delta(x) = F_n(x) - F(x)$")
    ax2.axhline(0, color="#888888", linewidth=0.8, linestyle="--")

    # Mark D+ and D-
    ax2.axhline( ks["d_plus"],  color="#C44E52", linewidth=1.2,
                linestyle="--", alpha=0.8, label=f"D+ = {ks['d_plus']:.4f}")
    ax2.axhline(-ks["d_minus"], color="#4C72B0", linewidth=1.2,
                linestyle="--", alpha=0.8, label=f"-D- = -{ks['d_minus']:.4f}")

    # Mark D location
    ax2.axvline(d_x, color="#2CA02C", linewidth=0.8, linestyle=":", alpha=0.7)
    ax2.scatter([d_x], [ks["d_ecdf"] - ks["d_cdf"]], color="#2CA02C",
                zorder=5, s=50, label=f"D attained at x={d_x:.3f}")

    ax2.set_xlabel("Value", fontsize=11)
    ax2.set_ylabel(r"$F_n(x) - F(x)$", fontsize=11)
    ax2.set_title("Gap Function: ECDF minus Theoretical CDF", fontsize=12)
    ax2.legend(fontsize=8.5, framealpha=0.85)

    # ---- Panel 3: PDF + rug for context ----------------------------------
    ax3 = axes[2]

    ax3.plot(xs, dist.pdf(xs), color="#C44E52", linewidth=2.0,
             label=f"PDF  {dist_label}")
    ax3.fill_between(xs, dist.pdf(xs), alpha=0.12, color="#C44E52")

    # Rug plot
    ax3.plot(sample, np.full_like(sample, -0.002 * dist.pdf(xs).max()),
             "|", color="#4C72B0", alpha=0.4, markersize=4, label="Sample rug")

    ax3.axvline(d_x, color="#2CA02C", linewidth=0.8, linestyle=":",
                alpha=0.7, label=f"D location  x={d_x:.3f}")

    ax3.set_xlabel("Value", fontsize=11)
    ax3.set_ylabel("Density", fontsize=11)
    ax3.set_title("Theoretical PDF with Sample Rug", fontsize=12)
    ax3.legend(fontsize=9, framealpha=0.85)

    plt.suptitle("Kolmogorov-Smirnov Goodness-of-Fit Demonstration",
                 fontsize=14, fontweight="bold", y=1.01)

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {out_path}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

SEPARATOR = "-" * 62

def print_report(sample: np.ndarray,
                 ks: dict,
                 dist_label: str,
                 alpha: float = 0.05) -> None:
    n = len(sample)
    print(SEPARATOR)
    print("KOLMOGOROV-SMIRNOV GOODNESS-OF-FIT REPORT")
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
    print("KS Test Results")
    print(f"  D  (two-sided)  : {ks['d_stat']:.6f}")
    print(f"  D+ (upper)      : {ks['d_plus']:.6f}  "
          f"  max of ECDF - CDF")
    print(f"  D- (lower)      : {ks['d_minus']:.6f}  "
          f"  max of CDF - ECDF (pre-jump)")
    print(f"  D attained at x : {ks['d_loc']:.4f}")
    print(f"  p-value         : {ks['p_value']:.6f}")
    print()

    # Approximate critical value (Kolmogorov distribution, large n)
    # c(alpha) / sqrt(n) where c(0.05) ~ 1.3581
    c = {0.10: 1.2238, 0.05: 1.3581, 0.01: 1.6276}
    print("  Approximate critical values  (large-n Kolmogorov distribution)")
    for a, cv in c.items():
        crit = cv / np.sqrt(n)
        reject = "REJECT" if ks["d_stat"] > crit else "fail to reject"
        print(f"    alpha={a:.2f}  D_crit={crit:.6f}  -> {reject}")

    print()
    decision = "FAIL TO REJECT" if ks["p_value"] >= alpha else "REJECT"
    print(SEPARATOR)
    print(f"At alpha={alpha}: {decision} the null hypothesis")
    print(f"  (H0: data follow {dist_label})")
    print(SEPARATOR)

    # Notes on KS vs chi-square
    print()
    print("Notes")
    print("  - KS uses the full CDF; no binning is required.")
    print("  - KS is sensitive to deviations anywhere in the distribution,")
    print("    but is most powerful near the median (where the CDF is steep).")
    print("  - The test assumes parameters are KNOWN (not estimated from data).")
    print("    If parameters were estimated via MLE, use the Lilliefors")
    print("    correction or an Anderson-Darling test instead.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="KS GoF demo: sample from a distribution and test fit via CDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument("--n",     type=int,   default=500,
                   help="Sample size (default: 500)")
    p.add_argument("--seed",  type=int,   default=None,
                   help="Random seed for reproducibility")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Significance level (default: 0.05)")
    p.add_argument("--out",   type=str,   default=None,
                   help="Output file path for the plot. "
                        "If omitted, the plot is displayed interactively.")

    p.add_argument("--dist", choices=["lognormal", "normal", "exponential"],
                   default="lognormal",
                   help="Distribution to sample from (default: lognormal)")

    # Lognormal
    p.add_argument("--mu",    type=float, default=2.884)
    p.add_argument("--sigma", type=float, default=0.472)
    # Normal
    p.add_argument("--mean",  type=float, default=0.0)
    p.add_argument("--std",   type=float, default=1.0)
    # Exponential
    p.add_argument("--rate",  type=float, default=1.0)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.dist == "lognormal":
        if args.sigma <= 0:
            sys.exit("ERROR: --sigma must be > 0.")
        dist = build_lognormal(args.mu, args.sigma)
        dist_label = f"Lognormal(mu={args.mu}, sigma={args.sigma})"
    elif args.dist == "normal":
        if args.std <= 0:
            sys.exit("ERROR: --std must be > 0.")
        dist = build_normal(args.mean, args.std)
        dist_label = f"Normal(mean={args.mean}, std={args.std})"
    elif args.dist == "exponential":
        if args.rate <= 0:
            sys.exit("ERROR: --rate must be > 0.")
        dist = build_exponential(args.rate)
        dist_label = f"Exponential(rate={args.rate})"
    else:
        sys.exit(f"Unknown distribution: {args.dist}")

    rng    = np.random.default_rng(args.seed)
    sample = dist.rvs(size=args.n, random_state=rng)

    ks = ks_test(sample, dist)

    print_report(sample, ks, dist_label, alpha=args.alpha)
    plot_ks(sample, dist, ks, dist_label, out_path=args.out)


if __name__ == "__main__":
    main()
