"""
plot_dist_discrete.py

Generate PMF and CDF plots for a chosen discrete distribution.

Supported distributions (select with --dist):
    binomial     Binomial(n, p)
    negbinom     Negative Binomial(r, p)   -- r = number of successes, p = success prob
    geometric    Geometric(p)              -- number of trials until first success; support {1, 2, ...}
    poisson      Poisson(mu)               -- mu = mean rate (lambda)
    duniform     Discrete Uniform(low, high) -- equal probability over {low, low+1, ..., high}
    hypergeom    Hypergeometric(M, n, N)   -- M = population, n = successes in pop, N = sample size

Usage examples:
    python plot_dist_discrete.py --dist binomial
    python plot_dist_discrete.py --dist binomial   --n 30 --p 0.4
    python plot_dist_discrete.py --dist negbinom   --r 10 --p 0.6
    python plot_dist_discrete.py --dist geometric  --p 0.25
    python plot_dist_discrete.py --dist poisson    --mu 8.0
    python plot_dist_discrete.py --dist duniform   --du-low 1 --du-high 6
    python plot_dist_discrete.py --dist hypergeom  --hg-M 100 --hg-n 20 --hg-N 10
    python plot_dist_discrete.py --dist poisson    --mu 3.0 --out my_plot.png --width 1600 --height 900
    python plot_dist_discrete.py --dist binomial   --no-reflines --no-annotation
    python plot_dist_discrete.py --dist poisson    --pmf-only
    python plot_dist_discrete.py --dist geometric  --cdf-only

Overlay controls (apply to all distributions):
    --percentile    : upper x-axis cutoff as a CDF percentile (default 99.5)
    --no-reflines   : suppress mean/mode vertical reference lines on both panels
    --no-annotation : suppress the stats call-out box on the PMF panel
    --pmf-only      : show only the PMF panel
    --cdf-only      : show only the CDF panel
    --out           : save plot to file
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Interactive backend helper
# ---------------------------------------------------------------------------

def _ensure_interactive_backend() -> bool:
    """Switch to an interactive backend if the current one is non-interactive (Agg)."""
    if matplotlib.get_backend().lower() != "agg":
        return True
    for backend in ("TkAgg", "QtAgg", "Qt5Agg"):
        try:
            matplotlib.use(backend)
            plt.switch_backend(backend)
            return True
        except Exception:
            continue
    return False


# ---------------------------------------------------------------------------
# Distribution registry
# ---------------------------------------------------------------------------

@dataclass
class DiscreteDistSpec:
    """Wraps a frozen scipy.stats discrete distribution and metadata for plotting."""
    name: str           # display name
    frozen: object      # frozen scipy.stats discrete distribution
    simio_expr: str     # informational Simio expression (not used in plot)
    k_lo: int | None = None  # hard lower support bound; None = use ppf(0.001)
    k_hi: int | None = None  # hard upper support bound; None = use ppf(percentile)

    def pmf(self, k: np.ndarray) -> np.ndarray:
        return self.frozen.pmf(k)

    def cdf(self, k: np.ndarray) -> np.ndarray:
        return self.frozen.cdf(k)

    def ppf(self, q: float) -> int:
        return int(self.frozen.ppf(q))

    def mean(self) -> float:
        return float(self.frozen.mean())

    def std(self) -> float:
        return float(self.frozen.std())

    def var(self) -> float:
        return float(self.frozen.var())

    def stats_skew_kurt(self):
        s, k = self.frozen.stats(moments="sk")
        return float(s), float(k)


def _build_binomial(args: argparse.Namespace) -> DiscreteDistSpec:
    if args.n < 1:
        _die("--n must be >= 1 for binomial distribution.")
    if not (0 < args.p < 1):
        _die("--p must be in (0, 1) for binomial distribution.")
    frozen = scipy_stats.binom(n=args.n, p=args.p)
    return DiscreteDistSpec(
        name=f"Binomial(n={args.n}, p={args.p})",
        frozen=frozen,
        simio_expr=f"Random.Binomial({args.n}, {args.p})",
    )


def _build_negbinom(args: argparse.Namespace) -> DiscreteDistSpec:
    if args.r < 1:
        _die("--r must be >= 1 for negative binomial distribution.")
    if not (0 < args.p < 1):
        _die("--p must be in (0, 1) for negative binomial distribution.")
    # scipy nbinom: n=number of successes (r), p=probability of success
    frozen = scipy_stats.nbinom(n=args.r, p=args.p)
    return DiscreteDistSpec(
        name=f"Negative Binomial(r={args.r}, p={args.p})",
        frozen=frozen,
        simio_expr=f"Random.NegativeBinomial({args.r}, {args.p})",
    )


def _build_geometric(args: argparse.Namespace) -> DiscreteDistSpec:
    if not (0 < args.p < 1):
        _die("--p must be in (0, 1) for geometric distribution.")
    # scipy geom: support {1, 2, 3, ...} -- number of trials until first success
    frozen = scipy_stats.geom(p=args.p)
    return DiscreteDistSpec(
        name=f"Geometric(p={args.p})",
        frozen=frozen,
        simio_expr=f"Random.Geometric({args.p})",
    )


def _build_poisson(args: argparse.Namespace) -> DiscreteDistSpec:
    if args.mu <= 0:
        _die("--mu must be > 0 for Poisson distribution.")
    frozen = scipy_stats.poisson(mu=args.mu)
    return DiscreteDistSpec(
        name=f"Poisson(mu={args.mu})",
        frozen=frozen,
        simio_expr=f"Random.Poisson({args.mu})",
    )


def _build_duniform(args: argparse.Namespace) -> DiscreteDistSpec:
    if args.du_low >= args.du_high:
        _die("--du-low must be < --du-high for discrete uniform distribution.")
    # scipy randint(low, high) has support {low, ..., high-1}; pass high+1 so
    # the support is {du_low, ..., du_high} inclusive.
    frozen = scipy_stats.randint(low=args.du_low, high=args.du_high + 1)
    return DiscreteDistSpec(
        name=f"Discrete Uniform(low={args.du_low}, high={args.du_high})",
        frozen=frozen,
        simio_expr=f"Random.DiscreteUniform({args.du_low}, {args.du_high})",
        k_lo=args.du_low,
        k_hi=args.du_high,
    )


def _build_hypergeometric(args: argparse.Namespace) -> DiscreteDistSpec:
    M, n, N = args.hg_M, args.hg_n, args.hg_N
    if M < 1:
        _die("--hg-M (population size) must be >= 1.")
    if n < 1 or n > M:
        _die("--hg-n (successes in population) must be in [1, M].")
    if N < 1 or N > M:
        _die("--hg-N (sample size) must be in [1, M].")
    # scipy hypergeom(M, n, N): M=population, n=successes in pop, N=sample size
    frozen = scipy_stats.hypergeom(M=M, n=n, N=N)
    k_lo = max(0, N + n - M)   # max(0, N+n-M)
    k_hi = min(N, n)            # min(N, n)
    return DiscreteDistSpec(
        name=f"Hypergeometric(M={M}, n={n}, N={N})",
        frozen=frozen,
        simio_expr=f"Random.Hypergeometric({M}, {n}, {N})",
        k_lo=k_lo,
        k_hi=k_hi,
    )


BUILDERS: dict[str, Callable[[argparse.Namespace], DiscreteDistSpec]] = {
    "binomial":    _build_binomial,
    "negbinom":    _build_negbinom,
    "geometric":   _build_geometric,
    "poisson":     _build_poisson,
    "duniform":    _build_duniform,
    "hypergeom":   _build_hypergeometric,
}


# ---------------------------------------------------------------------------
# Mode (analytical)
# ---------------------------------------------------------------------------

def _compute_mode(dist_key: str, args: argparse.Namespace) -> int:
    """Return the mode of the distribution as an integer."""
    if dist_key == "binomial":
        # Mode is floor((n+1)*p); if (n+1)*p is an integer there are two modes --
        # return the lower one (consistent with floor convention).
        return int((args.n + 1) * args.p)
    if dist_key == "negbinom":
        # Mode is floor((r-1)*(1-p)/p) for r >= 1; equals 0 when r == 1.
        return max(0, int((args.r - 1) * (1 - args.p) / args.p))
    if dist_key == "geometric":
        return 1    # always 1 for the trials-until-success parameterization
    if dist_key == "poisson":
        # Mode is floor(mu); when mu is a positive integer, floor(mu)-1 is also a
        # mode -- return floor(mu) as the canonical choice.
        return int(args.mu)
    if dist_key == "duniform":
        # Every value is equally likely; return the midpoint (rounded down).
        return (args.du_low + args.du_high) // 2
    if dist_key == "hypergeom":
        # Mode ~ floor((N+1)(n+1) / (M+2))
        return int((args.hg_N + 1) * (args.hg_n + 1) / (args.hg_M + 2))
    raise ValueError(f"Unknown dist_key: {dist_key}")


# ---------------------------------------------------------------------------
# Stats computation
# ---------------------------------------------------------------------------

def compute_stats(dist_key: str, args: argparse.Namespace,
                  spec: DiscreteDistSpec) -> dict:
    mean  = spec.mean()
    std   = spec.std()
    var   = spec.var()
    skew, kurt = spec.stats_skew_kurt()
    mode  = _compute_mode(dist_key, args)
    return dict(mean=mean, std=std, var=var, mode=mode,
                skewness=skew, excess_kurtosis=kurt)


def make_annotation(st: dict) -> str:
    lines = [
        f"Mean          = {st['mean']:.4g}",
        f"Mode          = {st['mode']}",
        f"Std Dev       = {st['std']:.4g}",
        f"Variance      = {st['var']:.4g}",
        f"Skewness      = {st['skewness']:.4g}",
        f"Excess Kurt.  = {st['excess_kurtosis']:.4g}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Integer x-range
# ---------------------------------------------------------------------------

def build_k_range(spec: DiscreteDistSpec, percentile: float) -> np.ndarray:
    """Return an integer array covering the distribution's plotted support."""
    k_lo = spec.k_lo if spec.k_lo is not None else spec.ppf(0.001)
    k_hi = spec.k_hi if spec.k_hi is not None else spec.ppf(percentile / 100.0)
    return np.arange(k_lo, k_hi + 1, dtype=int)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_dist(
    dist_key: str,
    spec: DiscreteDistSpec,
    args: argparse.Namespace,
    show_reflines: bool,
    show_annotation: bool,
) -> None:
    k   = build_k_range(spec, args.percentile)
    pmf = spec.pmf(k)
    cdf = spec.cdf(k)
    st  = compute_stats(dist_key, args, spec)

    dpi  = args.dpi
    w_px = args.width  if args.width  is not None else 1200
    h_px = args.height if args.height is not None else 1050
    figsize = (w_px / dpi, h_px / dpi)

    # Scale fonts and line widths relative to the default figure size (1200x1050).
    _DEFAULT_W, _DEFAULT_H = 1200, 1050
    scale     = (w_px / _DEFAULT_W * h_px / _DEFAULT_H) ** 0.5
    fs_title  = round(13  * scale, 1)
    fs_label  = round(10  * scale, 1)
    fs_legend = round(8   * scale, 1)
    fs_annot  = round(7.5 * scale, 1)
    lw_ref    = round(1.4 * scale, 2)
    lw_half   = round(1.0 * scale, 2)
    # Stem marker size and line width scale with figure
    ms_stem   = round(4   * scale, 1)
    lw_stem   = round(1.2 * scale, 2)

    show_pmf  = not args.cdf_only
    show_cdf  = not args.pmf_only
    n_panels  = show_pmf + show_cdf

    fig, axes = plt.subplots(
        n_panels, 1, figsize=figsize,
        sharex=(n_panels > 1),
        constrained_layout=True,
    )
    axes   = [axes] if n_panels == 1 else list(axes)
    ax_pmf = axes[0]  if show_pmf else None
    ax_cdf = axes[-1] if show_cdf else None

    fig.suptitle(spec.name, fontsize=fs_title, fontweight="bold")

    # Reference lines: mean and mode
    ref_lines = [
        ("Mean", st["mean"], "firebrick",  "--"),
        ("Mode", st["mode"], "seagreen",   ":"),
    ]

    # --- PMF panel ---
    if show_pmf:
        # Stem plot: vertical lines from baseline to marker
        markerline, stemlines, baseline = ax_pmf.stem(
            k, pmf, linefmt="steelblue", markerfmt="o", basefmt=" ",
        )
        markerline.set(markersize=ms_stem, color="steelblue")
        plt.setp(stemlines, linewidth=lw_stem)

        if show_reflines:
            for label, val, color, ls in ref_lines:
                ax_pmf.axvline(val, color=color, linestyle=ls, linewidth=lw_ref,
                               label=f"{label} = {val:.4g}")

        ax_pmf.set_ylabel("Probability", fontsize=fs_label)
        ax_pmf.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax_pmf.tick_params(axis="both", labelsize=fs_legend)
        ax_pmf.grid(True, linestyle=":", alpha=0.5, axis="y")

        # Build legend: PMF entry + ref lines if shown
        pmf_handle = matplotlib.lines.Line2D(
            [], [], color="steelblue", marker="o", linestyle="-",
            linewidth=lw_stem, markersize=ms_stem, label="PMF",
        )
        handles = [pmf_handle]
        if show_reflines:
            for label, val, color, ls in ref_lines:
                handles.append(matplotlib.lines.Line2D(
                    [], [], color=color, linestyle=ls, linewidth=lw_ref,
                    label=f"{label} = {val:.4g}",
                ))
        ax_pmf.legend(handles=handles, fontsize=fs_legend, loc="best")

        if not show_cdf:
            ax_pmf.set_xlabel("k", fontsize=fs_label)

        if show_annotation:
            ann_text = make_annotation(st)
            # Place annotation on whichever horizontal side has less PMF mass,
            # so it doesn't compete with the bulk of the stems or the legend.
            k_mid = (k[0] + k[-1]) / 2.0
            ann_on_right = st["mode"] < k_mid
            ann_x     = 0.98 if ann_on_right else 0.02
            ann_ha    = "right" if ann_on_right else "left"
            ax_pmf.text(
                ann_x, 0.97, ann_text,
                transform=ax_pmf.transAxes,
                fontsize=fs_annot, verticalalignment="top",
                horizontalalignment=ann_ha,
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                          alpha=0.8, edgecolor="gray"),
            )

    # --- CDF panel ---
    if show_cdf:
        # Step plot: right-continuous CDF for discrete distributions
        ax_cdf.step(k, cdf, where="post", color="darkorchid",
                    linewidth=lw_ref * 1.4, label="CDF")
        # Dots at each step to mark the defined values
        ax_cdf.plot(k, cdf, "o", color="darkorchid",
                    markersize=ms_stem * 0.85, zorder=3)

        if show_reflines:
            for label, val, color, ls in ref_lines:
                ax_cdf.axvline(val, color=color, linestyle=ls, linewidth=lw_ref)

        ax_cdf.axhline(0.5, color="gray", linestyle=":", linewidth=lw_half,
                       label="CDF = 0.50")
        ax_cdf.set_ylabel("Cumulative Probability", fontsize=fs_label)
        ax_cdf.set_xlabel("k", fontsize=fs_label)
        ax_cdf.set_ylim(0, 1.05)
        ax_cdf.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax_cdf.tick_params(axis="both", labelsize=fs_legend)
        ax_cdf.legend(fontsize=fs_legend, loc="lower right")
        ax_cdf.grid(True, linestyle=":", alpha=0.5)

    # Force integer x-axis ticks
    bottom_ax = ax_cdf if show_cdf else ax_pmf
    bottom_ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    if args.out:
        fig.savefig(args.out, dpi=dpi, bbox_inches="tight")
        print(f"Plot saved to: {args.out}  ({w_px}x{h_px} px at {dpi} dpi)")

    if _ensure_interactive_backend():
        plt.show()
    elif not args.out:
        print(
            "WARNING: No interactive display available and no --out file specified.\n"
            "         Use --out <file> to save the plot (e.g. --out plot.png).",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PMF and CDF of a discrete distribution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dist",
        type=str.lower,
        choices=list(BUILDERS),
        default="poisson",
        metavar="DIST",
        help=(
            "Distribution to plot. Choices: "
            + ", ".join(BUILDERS)
            + ". (default: poisson)"
        ),
    )

    # --- Binomial ---
    g_binom = parser.add_argument_group("Binomial parameters")
    g_binom.add_argument("--n", type=int, default=20,
                         help="Number of trials (must be >= 1).")
    g_binom.add_argument("--p", type=float, default=0.5,
                         help="Probability of success per trial; must be in (0, 1). "
                              "Used by binomial, negative binomial, and geometric.")

    # --- Negative Binomial ---
    g_nb = parser.add_argument_group("Negative Binomial parameters")
    g_nb.add_argument("--r", type=int, default=5,
                      help="Number of successes to achieve (must be >= 1). "
                           "--p (above) is also used.")

    # --- Poisson ---
    g_pois = parser.add_argument_group("Poisson parameters")
    g_pois.add_argument("--mu", type=float, default=5.0,
                        help="Mean rate (lambda) of the Poisson distribution (must be > 0).")

    # --- Discrete Uniform ---
    g_du = parser.add_argument_group("Discrete Uniform parameters")
    g_du.add_argument("--du-low",  type=int, default=1,
                      help="Lower bound of the discrete uniform distribution (inclusive).")
    g_du.add_argument("--du-high", type=int, default=10,
                      help="Upper bound of the discrete uniform distribution (inclusive, must be > --du-low).")

    # --- Hypergeometric ---
    g_hg = parser.add_argument_group("Hypergeometric parameters")
    g_hg.add_argument("--hg-M", type=int, default=50,
                      help="Population size (must be >= 1).")
    g_hg.add_argument("--hg-n", type=int, default=10,
                      help="Number of successes in the population (must be in [1, M]).")
    g_hg.add_argument("--hg-N", type=int, default=15,
                      help="Sample size drawn from the population (must be in [1, M]).")

    # --- Plot controls (shared) ---
    g_plot = parser.add_argument_group("Plot controls")
    g_plot.add_argument("--percentile", type=float, default=99.5,
                        help="Upper x-axis cutoff as a CDF percentile; must be in (50, 100).")
    g_plot.add_argument("--out",    type=str, default=None,
                        help="Optional output file path (e.g. plot.png, plot.pdf).")
    g_plot.add_argument("--width",  type=int, default=None,
                        help="Output image width in pixels. Default: 1200.")
    g_plot.add_argument("--height", type=int, default=None,
                        help="Output image height in pixels. Default: 1050.")
    g_plot.add_argument("--dpi",    type=int, default=150,
                        help="Resolution in dots per inch for the saved file.")
    g_plot.add_argument("--no-reflines",   action="store_true",
                        help="Suppress mean/mode vertical reference lines on both panels.")
    g_plot.add_argument("--no-annotation", action="store_true",
                        help="Suppress the stats call-out box on the PMF panel.")
    g_plot.add_argument("--pmf-only", action="store_true",
                        help="Show only the PMF panel (mutually exclusive with --cdf-only).")
    g_plot.add_argument("--cdf-only", action="store_true",
                        help="Show only the CDF panel (mutually exclusive with --pmf-only).")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.pmf_only and args.cdf_only:
        _die("--pmf-only and --cdf-only are mutually exclusive.")
    if not (50.0 < args.percentile < 100.0):
        _die("--percentile must be in (50, 100).")

    builder = BUILDERS[args.dist]
    spec = builder(args)

    plot_dist(
        dist_key=args.dist,
        spec=spec,
        args=args,
        show_reflines=not args.no_reflines,
        show_annotation=not args.no_annotation,
    )


if __name__ == "__main__":
    main()
