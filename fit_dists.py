"""
fit_dists.py
--------------------
Fits numerical observations to five parametric distributions and reports
descriptive statistics, MLE parameters, and GoF p-values.

Distributions fitted
--------------------
  normal      - N(mu, sigma)
  uniform     - U(loc, loc+scale)
  exponential - Exp(loc, scale=1/lambda)
  triangular  - Tri(c, loc, scale)   [c is the shape: mode=(loc + c*scale)]
  weibull     - Weibull(k, lambda)   [k=shape, lambda=scale, loc fixed at 0]
  lognormal   - LogN(s, loc, scale)  [s=sigma of the underlying normal]

Goodness-of-fit
---------------
  Kolmogorov-Smirnov (KS) test is used for all distributions.
  p-value > 0.05 -> insufficient evidence to reject the fit at the 5 % level.

Dependencies
------------
  numpy, scipy  (both ship with most scientific Python environments)
  Install: pip install numpy scipy
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DescriptiveStats:
    n: int
    mean: float
    std: float          # sample std (ddof=1)
    variance: float
    median: float
    skewness: float
    kurtosis: float     # excess kurtosis
    minimum: float
    maximum: float

    def __str__(self) -> str:
        lines = [
            "-- Descriptive Statistics -------------------------------------",
            f"  n          : {self.n}",
            f"  mean       : {self.mean:.6g}",
            f"  std (s)    : {self.std:.6g}",
            f"  variance   : {self.variance:.6g}",
            f"  median     : {self.median:.6g}",
            f"  skewness   : {self.skewness:.6g}",
            f"  kurtosis   : {self.kurtosis:.6g}  (excess)",
            f"  min        : {self.minimum:.6g}",
            f"  max        : {self.maximum:.6g}",
        ]
        return "\n".join(lines)


@dataclass
class FitResult:
    distribution: str
    params: dict[str, float]
    ks_statistic: float
    p_value: float
    success: bool
    error: str | None = None

    def __str__(self) -> str:
        if not self.success:
            return (
                f"  {self.distribution:<14}: FIT FAILED - {self.error}"
            )
        param_str = ",  ".join(f"{k}={v:.6g}" for k, v in self.params.items())
        return (
            f"  {self.distribution:<14}: {param_str}\n"
            f"  {'':14}  KS={self.ks_statistic:.4f},  p={self.p_value:.4f}"
            + ("  [good fit]" if self.p_value >= 0.05 else "  [poor fit]")
        )


@dataclass
class FitReport:
    descriptive: DescriptiveStats
    fits: list[FitResult] = field(default_factory=list)

    def __str__(self) -> str:
        sep = "-" * 63
        lines = [sep, str(self.descriptive), sep, "-- Distribution Fits ------------------------------------------"]
        for fr in self.fits:
            lines.append(str(fr))
        lines.append(sep)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core fitting logic
# ---------------------------------------------------------------------------

def _compute_descriptive(data: np.ndarray) -> DescriptiveStats:
    return DescriptiveStats(
        n=len(data),
        mean=float(np.mean(data)),
        std=float(np.std(data, ddof=1)),
        variance=float(np.var(data, ddof=1)),
        median=float(np.median(data)),
        skewness=float(stats.skew(data, bias=False)),
        kurtosis=float(stats.kurtosis(data, bias=False)),  # excess
        minimum=float(np.min(data)),
        maximum=float(np.max(data)),
    )


def _fit_normal(data: np.ndarray) -> FitResult:
    mu, sigma = stats.norm.fit(data)
    ks, p = stats.kstest(data, "norm", args=(mu, sigma))
    return FitResult(
        distribution="normal",
        params={"mu": mu, "sigma": sigma},
        ks_statistic=ks,
        p_value=p,
        success=True,
    )


def _fit_uniform(data: np.ndarray) -> FitResult:
    loc, scale = stats.uniform.fit(data)
    ks, p = stats.kstest(data, "uniform", args=(loc, scale))
    return FitResult(
        distribution="uniform",
        params={"a (loc)": loc, "b (loc+scale)": loc + scale},
        ks_statistic=ks,
        p_value=p,
        success=True,
    )


def _fit_exponential(data: np.ndarray) -> FitResult:
    # scipy's expon: CDF = 1 - exp(-(x-loc)/scale), where scale = 1/lambda
    loc, scale = stats.expon.fit(data, floc=np.min(data))
    lam = 1.0 / scale
    ks, p = stats.kstest(data, "expon", args=(loc, scale))
    return FitResult(
        distribution="exponential",
        params={"loc": loc, "scale (1/lambda)": scale, "lambda": lam},
        ks_statistic=ks,
        p_value=p,
        success=True,
    )


def _fit_triangular(data: np.ndarray) -> FitResult:
    # scipy's triang: shape c in (0,1), mode = loc + c*scale
    try:
        c, loc, scale = stats.triang.fit(data)
        mode = loc + c * scale
        ks, p = stats.kstest(data, "triang", args=(c, loc, scale))
        return FitResult(
            distribution="triangular",
            params={"a (min)": loc, "b (max)": loc + scale, "c (mode)": mode},
            ks_statistic=ks,
            p_value=p,
            success=True,
        )
    except Exception as exc:
        return FitResult(
            distribution="triangular",
            params={},
            ks_statistic=float("nan"),
            p_value=float("nan"),
            success=False,
            error=str(exc),
        )


def _fit_weibull(data: np.ndarray) -> FitResult:
    # scipy's weibull_min: CDF = 1 - exp(-((x-loc)/scale)^c)
    # where c=shape (k), scale=lambda. Fix loc=0 for the standard 2-param form.
    if np.any(data <= 0):
        return FitResult(
            distribution="weibull",
            params={},
            ks_statistic=float("nan"),
            p_value=float("nan"),
            success=False,
            error="weibull requires strictly positive data",
        )
    try:
        c, loc, scale = stats.weibull_min.fit(data, floc=0)
        ks, p = stats.kstest(data, "weibull_min", args=(c, loc, scale))
        return FitResult(
            distribution="weibull",
            params={"k (shape)": c, "lambda (scale)": scale, "loc": loc},
            ks_statistic=ks,
            p_value=p,
            success=True,
        )
    except Exception as exc:
        return FitResult(
            distribution="weibull",
            params={},
            ks_statistic=float("nan"),
            p_value=float("nan"),
            success=False,
            error=str(exc),
        )


def _fit_lognormal(data: np.ndarray) -> FitResult:
    if np.any(data <= 0):
        return FitResult(
            distribution="lognormal",
            params={},
            ks_statistic=float("nan"),
            p_value=float("nan"),
            success=False,
            error="lognormal requires strictly positive data",
        )
    try:
        s, loc, scale = stats.lognorm.fit(data, floc=0)
        mu_ln = np.log(scale)   # mean of underlying normal
        sigma_ln = s            # std of underlying normal
        ks, p = stats.kstest(data, "lognorm", args=(s, loc, scale))
        return FitResult(
            distribution="lognormal",
            params={"mu_ln": mu_ln, "sigma_ln": sigma_ln, "loc": loc},
            ks_statistic=ks,
            p_value=p,
            success=True,
        )
    except Exception as exc:
        return FitResult(
            distribution="lognormal",
            params={},
            ks_statistic=float("nan"),
            p_value=float("nan"),
            success=False,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_dists(
    data: list[float] | np.ndarray,
    *,
    verbose: bool = True,
) -> FitReport:
    """
    Fit five parametric distributions to *data* and return a FitReport.

    Parameters
    ----------
    data : array-like of floats
        The observations to fit (must contain at least 3 values).
    verbose : bool, default True
        If True, print the formatted report to stdout.

    Returns
    -------
    FitReport
        Contains DescriptiveStats and a list of FitResult objects.
    """
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 1:
        raise ValueError("data must be a 1-D array-like.")
    if len(arr) < 3:
        raise ValueError("At least 3 observations are required.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("data contains NaN or infinite values.")

    descriptive = _compute_descriptive(arr)

    fitters = [
        _fit_normal,
        _fit_uniform,
        _fit_exponential,
        _fit_triangular,
        _fit_weibull,
        _fit_lognormal,
    ]

    fits: list[FitResult] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for fitter in fitters:
            fits.append(fitter(arr))

    report = FitReport(descriptive=descriptive, fits=fits)

    if verbose:
        print(report)

    return report
