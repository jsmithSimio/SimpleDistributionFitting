"""
plot_dist.py

Generate PDF and CDF plots for a chosen parametric distribution.

Supported distributions (select with --dist):
    normal       Normal(mean, std)
    uniform      Uniform(low, high)
    exponential  Exponential(rate)          -- parameterized by rate = 1/mean
    triangular   Triangular(low, mode, high)
    lognormal    Lognormal(mu, sigma)       -- log-scale parameterization
    weibull      Weibull(shape, scale)      -- 2-parameter Weibull

Usage examples:
    python plot_dist.py --dist normal
    python plot_dist.py --dist normal      --mean 5.0 --std 1.5
    python plot_dist.py --dist uniform     --low 2.0 --high 8.0
    python plot_dist.py --dist exponential --rate 0.5
    python plot_dist.py --dist triangular  --low 1.0 --tri-mode 3.0 --high 7.0
    python plot_dist.py --dist lognormal   --mu 1.5 --sigma 0.4
    python plot_dist.py --dist weibull     --shape 2.0 --scale 5.0
    python plot_dist.py --dist lognormal   --mu 1.0 --sigma 0.6 --percentile 99 --out my_plot.png
    python plot_dist.py --dist normal      --no-reflines --no-annotation

Overlay controls (apply to all distributions):
    --percentile    : upper x-axis cutoff as a CDF percentile (default 99.5)
    --no-reflines   : suppress mean/median/mode vertical reference lines
    --no-annotation : suppress the stats call-out box on the PDF panel
    --out           : save plot to file instead of (or in addition to) displaying
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
# Interactive backend helper (unchanged from plot_lognormal.py)
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
class DistSpec:
    """Wraps a frozen scipy.stats distribution and metadata for plotting."""
    name: str               # display name
    frozen: object          # frozen scipy.stats distribution
    simio_expr: str         # informational Simio expression (not used in plot)
    x_lower_bound: float | None = None  # hard lower bound for x-axis (e.g. 0.0 for
                                        # non-negative distributions); None = use ppf(0.001)
    x_upper_bound: float | None = None  # hard upper bound for x-axis (e.g. high for
                                        # bounded distributions); None = use ppf(percentile)

    def ppf(self, q: float) -> float:
        return self.frozen.ppf(q)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self.frozen.pdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return self.frozen.cdf(x)

    def mean(self) -> float:
        return float(self.frozen.mean())

    def median(self) -> float:
        return float(self.frozen.median())

    def std(self) -> float:
        return float(self.frozen.std())

    def stats_skew_kurt(self):
        s, k = self.frozen.stats(moments="sk")
        return float(s), float(k)


def _build_normal(args: argparse.Namespace) -> DistSpec:
    frozen = scipy_stats.norm(loc=args.mean, scale=args.std)
    return DistSpec(
        name=f"Normal(mean={args.mean}, std={args.std})",
        frozen=frozen,
        simio_expr=f"Random.Normal({args.mean}, {args.std})",
    )


def _build_uniform(args: argparse.Namespace) -> DistSpec:
    if args.high <= args.low:
        _die("--high must be greater than --low for uniform distribution.")
    frozen = scipy_stats.uniform(loc=args.low, scale=args.high - args.low)
    return DistSpec(
        name=f"Uniform(low={args.low}, high={args.high})",
        frozen=frozen,
        simio_expr=f"Random.Uniform({args.low}, {args.high})",
        x_lower_bound=args.low,
        x_upper_bound=args.high,
    )


def _build_exponential(args: argparse.Namespace) -> DistSpec:
    if args.rate <= 0:
        _die("--rate must be > 0 for exponential distribution.")
    mean = 1.0 / args.rate
    frozen = scipy_stats.expon(scale=mean)
    return DistSpec(
        name=f"Exponential(rate={args.rate})",
        frozen=frozen,
        simio_expr=f"Random.Exponential({mean})",
        x_lower_bound=0.0,
    )


def _build_triangular(args: argparse.Namespace) -> DistSpec:
    lo, mo, hi = args.low, args.tri_mode, args.high
    if not (lo < hi):
        _die("--low must be < --high for triangular distribution.")
    if not (lo <= mo <= hi):
        _die("--tri-mode must be between --low and --high.")
    # scipy triangular: c = (mode - low) / (high - low), loc=low, scale=high-low
    c = (mo - lo) / (hi - lo)
    frozen = scipy_stats.triang(c=c, loc=lo, scale=hi - lo)
    return DistSpec(
        name=f"Triangular(low={lo}, mode={mo}, high={hi})",
        frozen=frozen,
        simio_expr=f"Random.Triangular({lo}, {mo}, {hi})",
        x_lower_bound=lo,
        x_upper_bound=hi,
    )


def _build_lognormal(args: argparse.Namespace) -> DistSpec:
    if args.sigma <= 0:
        _die("--sigma must be > 0 for lognormal distribution.")
    frozen = scipy_stats.lognorm(s=args.sigma, scale=np.exp(args.mu))
    return DistSpec(
        name=f"Lognormal(mu={args.mu}, sigma={args.sigma})",
        frozen=frozen,
        simio_expr=f"Random.Lognormal({args.mu}, {args.sigma})",
        x_lower_bound=0.0,
    )


def _build_weibull(args: argparse.Namespace) -> DistSpec:
    if args.shape <= 0:
        _die("--shape must be > 0 for Weibull distribution.")
    if args.scale <= 0:
        _die("--scale must be > 0 for Weibull distribution.")
    # scipy weibull_min: shape=c (shape), scale=scale, loc=0
    frozen = scipy_stats.weibull_min(c=args.shape, scale=args.scale)
    return DistSpec(
        name=f"Weibull(shape={args.shape}, scale={args.scale})",
        frozen=frozen,
        simio_expr=f"Random.Weibull({args.shape}, {args.scale})",
        x_lower_bound=0.0,
    )


BUILDERS: dict[str, Callable[[argparse.Namespace], DistSpec]] = {
    "normal":       _build_normal,
    "uniform":      _build_uniform,
    "exponential":  _build_exponential,
    "triangular":   _build_triangular,
    "lognormal":    _build_lognormal,
    "weibull":      _build_weibull,
}


# ---------------------------------------------------------------------------
# Mode helpers (analytical where closed-form exists, else numerical)
# ---------------------------------------------------------------------------

def _compute_mode(dist_key: str, args: argparse.Namespace, spec: DistSpec) -> float | None:
    """
    Return the mode for the distribution, or None if it is not well-defined
    (e.g. exponential with rate > 0 has mode = 0, which may be outside the plot range).
    """
    if dist_key == "normal":
        return args.mean
    if dist_key == "uniform":
        return None                          # every point is the mode; skip ref line
    if dist_key == "exponential":
        return 0.0                           # mode is at the boundary
    if dist_key == "triangular":
        return args.tri_mode
    if dist_key == "lognormal":
        return float(np.exp(args.mu - args.sigma ** 2))
    if dist_key == "weibull":
        k, lam = args.shape, args.scale
        if k > 1.0:
            return float(lam * ((k - 1.0) / k) ** (1.0 / k))
        else:
            return 0.0                       # mode at boundary for k <= 1
    return None


# ---------------------------------------------------------------------------
# Stats computation
# ---------------------------------------------------------------------------

def compute_stats(dist_key: str, args: argparse.Namespace, spec: DistSpec) -> dict:
    mean   = spec.mean()
    median = spec.median()
    std    = spec.std()
    skew, kurt = spec.stats_skew_kurt()
    mode   = _compute_mode(dist_key, args, spec)
    return dict(mean=mean, median=median, mode=mode, std=std,
                skewness=skew, excess_kurtosis=kurt)


def make_annotation(stats: dict) -> str:
    mode_str = f"{stats['mode']:.4g}" if stats["mode"] is not None else "N/A"
    lines = [
        f"Mean          = {stats['mean']:.4g}",
        f"Median        = {stats['median']:.4g}",
        f"Mode          = {mode_str}",
        f"Std Dev       = {stats['std']:.4g}",
        f"Skewness      = {stats['skewness']:.4g}",
        f"Excess Kurt.  = {stats['excess_kurtosis']:.4g}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# X-range
# ---------------------------------------------------------------------------

def _right_edge(spec: DistSpec, percentile: float) -> float:
    """
    Return a right x-axis bound for unbounded distributions that is the larger of:
      - ppf(percentile/100), and
      - the point past the mode where the PDF decays to 1% of its peak value.

    The second criterion prevents visually abrupt cutoffs for tight distributions
    (e.g. high-shape Weibull) where ppf(0.995) still has meaningful density.
    """
    from scipy.optimize import brentq

    x_pct = spec.ppf(percentile / 100.0)

    # Find the mode (peak of PDF) via the median as a search starting point
    x_mode = spec.ppf(0.50)
    # Refine: scan a grid from x_lo to x_pct to locate the approximate peak
    x_scan = np.linspace(spec.ppf(0.01), x_pct, 500)
    pdf_scan = spec.pdf(x_scan)
    x_mode = x_scan[np.argmax(pdf_scan)]
    peak = pdf_scan.max()

    threshold = peak * 0.01  # 1% of peak density

    # If PDF at x_pct is already below threshold, no extension needed
    if spec.pdf(x_pct) <= threshold:
        return x_pct

    # Search for the right x where PDF drops to threshold, beyond x_pct
    # Upper search bound: ppf(0.9999) is safely in the far tail
    x_far = spec.ppf(0.9999)
    try:
        x_decay = brentq(lambda x: spec.pdf(x) - threshold, x_pct, x_far)
    except ValueError:
        # brentq failed (PDF never drops below threshold in range); fall back
        return x_far

    return max(x_pct, x_decay)


def build_x_range(spec: DistSpec, percentile: float) -> np.ndarray:
    """Return x from the hard lower bound (or ppf(0.001)) to the hard upper bound (or ppf(percentile))."""
    if spec.x_upper_bound is not None:
        x_hi = spec.x_upper_bound
    else:
        x_hi = _right_edge(spec, percentile)

    if spec.x_lower_bound is not None:
        x_lo = spec.x_lower_bound
    else:
        x_lo = spec.ppf(0.001)
        if not np.isfinite(x_lo):
            x_lo = spec.ppf(0.0)
    return np.linspace(x_lo, x_hi, 2000)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_dist(
    dist_key: str,
    spec: DistSpec,
    args: argparse.Namespace,
    show_reflines: bool,
    show_annotation: bool,
) -> None:
    x    = build_x_range(spec, args.percentile)
    pdf  = spec.pdf(x)
    cdf  = spec.cdf(x)
    st   = compute_stats(dist_key, args, spec)

    dpi = args.dpi
    w_px = args.width  if args.width  is not None else 1200
    h_px = args.height if args.height is not None else 1050
    figsize = (w_px / dpi, h_px / dpi)

    # Scale fonts and line widths relative to the default figure size (1200x1050).
    # Use the geometric mean of the width and height scale factors so that
    # non-proportional resizes (e.g. wider but not taller) split the difference.
    _DEFAULT_W, _DEFAULT_H = 1200, 1050
    scale = (w_px / _DEFAULT_W * h_px / _DEFAULT_H) ** 0.5
    fs_title  = round(13   * scale, 1)
    fs_label  = round(10   * scale, 1)
    fs_legend = round(8    * scale, 1)
    fs_annot  = round(7.5  * scale, 1)
    lw_curve  = round(2    * scale, 2)
    lw_ref    = round(1.4  * scale, 2)
    lw_half   = round(1.0  * scale, 2)

    fig, (ax_pdf, ax_cdf) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        constrained_layout=True,
    )
    fig.suptitle(spec.name, fontsize=fs_title, fontweight="bold")

    # Reference lines: mean (always), median (always), mode (if defined)
    ref_lines = [
        ("Mean",   st["mean"],   "firebrick",  "--"),
        ("Median", st["median"], "darkorange", "-."),
    ]
    if st["mode"] is not None:
        ref_lines.append(("Mode", st["mode"], "seagreen", ":"))

    # --- PDF panel ---
    ax_pdf.plot(x, pdf, color="steelblue", linewidth=lw_curve, label="PDF")
    ax_pdf.fill_between(x, pdf, alpha=0.15, color="steelblue")

    if show_reflines:
        for label, val, color, ls in ref_lines:
            ax_pdf.axvline(val, color=color, linestyle=ls, linewidth=lw_ref,
                           label=f"{label} = {val:.4g}")

    ax_pdf.set_ylabel("Probability Density", fontsize=fs_label)
    ax_pdf.legend(fontsize=fs_legend, loc="upper right")
    ax_pdf.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax_pdf.tick_params(axis="both", labelsize=fs_legend)
    ax_pdf.grid(True, linestyle=":", alpha=0.5)

    if show_annotation:
        ann_text = make_annotation(st)
        ax_pdf.text(
            0.02, 0.97, ann_text,
            transform=ax_pdf.transAxes,
            fontsize=fs_annot, verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      alpha=0.8, edgecolor="gray"),
        )

    # --- CDF panel ---
    ax_cdf.plot(x, cdf, color="darkorchid", linewidth=lw_curve, label="CDF")

    if show_reflines:
        for label, val, color, ls in ref_lines:
            ax_cdf.axvline(val, color=color, linestyle=ls, linewidth=lw_ref)

    ax_cdf.axhline(0.5, color="gray", linestyle=":", linewidth=lw_half, label="CDF = 0.50")
    ax_cdf.set_ylabel("Cumulative Probability", fontsize=fs_label)
    ax_cdf.set_xlabel("x", fontsize=fs_label)
    ax_cdf.set_ylim(0, 1.05)
    ax_cdf.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax_cdf.tick_params(axis="both", labelsize=fs_legend)
    ax_cdf.legend(fontsize=fs_legend, loc="lower right")
    ax_cdf.grid(True, linestyle=":", alpha=0.5)

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
        description="Plot PDF and CDF of a parametric distribution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dist",
        type=str.lower,
        choices=list(BUILDERS),
        default="normal",
        metavar="DIST",
        help=(
            "Distribution to plot. Choices: "
            + ", ".join(BUILDERS)
            + ". (default: normal)"
        ),
    )

    # --- Normal ---
    g_norm = parser.add_argument_group("Normal parameters")
    g_norm.add_argument("--mean", type=float, default=0.0,
                        help="Mean of the normal distribution.")
    g_norm.add_argument("--std",  type=float, default=1.0,
                        help="Std dev of the normal distribution (must be > 0).")

    # --- Uniform ---
    g_unif = parser.add_argument_group("Uniform parameters")
    g_unif.add_argument("--low",  type=float, default=0.0,
                        help="Lower bound of the uniform distribution.")
    g_unif.add_argument("--high", type=float, default=1.0,
                        help="Upper bound of the uniform distribution (must be > --low).")

    # --- Exponential ---
    g_exp = parser.add_argument_group("Exponential parameters")
    g_exp.add_argument("--rate", type=float, default=1.0,
                       help="Rate (lambda = 1/mean) of the exponential distribution.")

    # --- Triangular ---
    g_tri = parser.add_argument_group("Triangular parameters")
    # --low and --high are shared with uniform; add only mode here
    g_tri.add_argument("--tri-mode", type=float, default=0.5,
                       help="Mode of the triangular distribution (must be in [--low, --high]).")

    # --- Lognormal ---
    g_logn = parser.add_argument_group("Lognormal parameters")
    g_logn.add_argument("--mu",    type=float, default=0.0,
                        help="Log-scale mean (mu) of the lognormal distribution.")
    g_logn.add_argument("--sigma", type=float, default=1.0,
                        help="Log-scale std dev (sigma) of the lognormal distribution (must be > 0).")

    # --- Weibull ---
    g_wei = parser.add_argument_group("Weibull parameters")
    g_wei.add_argument("--shape", type=float, default=2.0,
                       help="Shape parameter (k) of the Weibull distribution (must be > 0).")
    g_wei.add_argument("--scale", type=float, default=1.0,
                       help="Scale parameter (lambda) of the Weibull distribution (must be > 0).")

    # --- Plot controls (shared) ---
    g_plot = parser.add_argument_group("Plot controls")
    g_plot.add_argument("--percentile", type=float, default=99.5,
                        help="Upper x-axis cutoff as a CDF percentile; must be in (50, 100).")
    g_plot.add_argument("--out", type=str, default=None,
                        help="Optional output file path (e.g. plot.png, plot.pdf).")
    g_plot.add_argument("--width",  type=int, default=None,
                        help="Output image width in pixels (requires --out). Default: 1200.")
    g_plot.add_argument("--height", type=int, default=None,
                        help="Output image height in pixels (requires --out). Default: 1050.")
    g_plot.add_argument("--dpi",    type=int, default=150,
                        help="Resolution in dots per inch for the saved file. (default: 150)")
    g_plot.add_argument("--no-reflines",   action="store_true",
                        help="Suppress mean/median/mode vertical reference lines.")
    g_plot.add_argument("--no-annotation", action="store_true",
                        help="Suppress the stats call-out box on the PDF panel.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Shared validation
    if not (50.0 < args.percentile < 100.0):
        _die("--percentile must be in (50, 100).")

    # Normal std validation (other per-dist validations happen inside builders)
    if args.dist == "normal" and args.std <= 0:
        _die("--std must be > 0.")

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
