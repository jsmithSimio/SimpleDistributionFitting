# fit_dists

A Python utility that fits six parametric distributions to a 1-D sample of observations, reports descriptive statistics and MLE parameters, performs a Kolmogorov-Smirnov goodness-of-fit test for each distribution, and generates ready-to-paste [Simio](https://www.simio.com/) random-variate expressions.

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Distributions](#distributions)
- [Goodness-of-Fit](#goodness-of-fit)
- [Simio Expressions](#simio-expressions)
- [Location Shift](#location-shift)
- [API Reference](#api-reference)
  - [`fit_dists()`](#fit_dists-1)
  - [`FitReport`](#fitreport)
  - [`FitResult`](#fitresult)
  - [`DescriptiveStats`](#descriptivestats)
- [Sample Output](#sample-output)
- [Notes and Limitations](#notes-and-limitations)

---

## Requirements

| Package | Tested version |
|---------|---------------|
| Python  | >= 3.10        |
| NumPy   | >= 1.24        |
| SciPy   | >= 1.11        |

## Installation

```bash
pip install numpy scipy
```

No additional setup is required -- `fit_dists.py` is a single self-contained module.

---

## Quick Start

```python
from fit_dists import fit_dists
import numpy as np

# Any array-like of floats with at least 3 observations
data = np.random.weibull(1.5, size=500) * 10

report = fit_dists(data)                    # prints report to stdout by default
report = fit_dists(data, verbose=False)     # suppress printing, use object directly

# Programmatic access
print(report.descriptive.mean)
for result in report.fits:
    if result.success and result.p_value >= 0.05:
        print(result.distribution, "->", result.simio_expression)
```

---

## Distributions

Six distributions are fitted using **maximum likelihood estimation (MLE)** via `scipy.stats`:

| Distribution | Parameters reported | scipy function | Data constraint |
|---|---|---|---|
| Normal | `mu`, `sigma` | `norm.fit` | None |
| Uniform | `a` (min), `b` (max) | `uniform.fit` | None |
| Exponential | `loc`, `scale` (= 1/lambda), `lambda` | `expon.fit` | None (loc fixed to sample min) |
| Triangular | `a` (min), `b` (max), `c` (mode) | `triang.fit` | None |
| Weibull | `k` (shape), `lambda` (scale) | `weibull_min.fit` | **Strictly positive** (`x > 0`) |
| Lognormal | `mu_ln`, `sigma_ln` | `lognorm.fit` | **Strictly positive** (`x > 0`) |

### Parameter notes

**Exponential** -- SciPy's `expon` is a shifted exponential `CDF = 1 - exp(-(x - loc) / scale)`. `loc` is fixed to the sample minimum so that the support matches the data range. The Simio expression uses only `scale` (= mean inter-arrival time), which corresponds to `Random.Exponential(scale)`.

**Triangular** -- SciPy uses an internal shape parameter `c in (0, 1)` where `mode = loc + c * scale`. The reported parameters are converted to the more intuitive (`a`, `b`, `c`) = (min, max, mode) form before display and before generating the Simio expression.

**Weibull** -- `loc` is fixed to 0 (standard 2-parameter form). Fitting will fail gracefully if any value is <= 0.

**Lognormal** -- `loc` is fixed to 0 and MLE is performed on the log-transformed data. `mu_ln` and `sigma_ln` are the mean and standard deviation of the underlying normal distribution. Fitting will fail gracefully if any value is <= 0.

---

## Goodness-of-Fit

The **Kolmogorov-Smirnov (KS) test** is used for all distributions via `scipy.stats.kstest`.

| Field | Meaning |
|---|---|
| `ks_statistic` | Maximum absolute deviation between empirical and fitted CDF |
| `p_value` | Probability of observing a deviation this large under the null hypothesis (data follows the fitted distribution) |

A `p_value >= 0.05` means there is insufficient evidence at the 5% level to reject the fit -- the result is labelled **`[good fit]`** in the printed report. A `p_value < 0.05` is labelled **`[poor fit]`**.

> **Important:** The KS test becomes more sensitive as sample size grows. With very large samples (n > 1000), even small, practically insignificant deviations can yield low p-values. Use engineering judgment alongside the statistical result.

---

## Simio Expressions

Each successful `FitResult` includes a `simio_expression` string ready to paste directly into a Simio model property.

| Distribution | Expression format |
|---|---|
| Normal | `Random.Normal(mu, sigma)` |
| Uniform | `Random.Uniform(a, b)` |
| Exponential | `Random.Exponential(scale)` |
| Triangular | `Random.Triangular(a, c, b)` |
| Weibull | `Random.Weibull(k, lambda)` |
| Lognormal | `Random.Lognormal(mu_ln, sigma_ln)` |

> Note the argument order for `Random.Triangular`: Simio uses **(min, mode, max)**, which differs from the scipy internal representation.

When `loc_shift` is nonzero, all expressions are prefixed with the shift value so they remain correct in the original scale (see [Location Shift](#location-shift)):

| Distribution | Expression format with shift |
|---|---|
| Normal | `loc_shift + Random.Normal(mu, sigma)` |
| Uniform | `loc_shift + Random.Uniform(a, b)` |
| Exponential | `loc_shift + Random.Exponential(scale)` |
| Triangular | `loc_shift + Random.Triangular(a, c, b)` |
| Weibull | `loc_shift + Random.Weibull(k, lambda)` |
| Lognormal | `loc_shift + Random.Lognormal(mu_ln, sigma_ln)` |

Failed fits (`success=False`) have `simio_expression = None` and no expression is printed.

---

## Location Shift

Some processes have a known non-zero minimum -- for example, a service time that can never be less than 30 seconds, or a delivery time with a guaranteed minimum of two days. In these cases, fitting the raw data forces the distributions to accommodate the shift, which can obscure the underlying shape and degrade goodness-of-fit results.

The `loc_shift` parameter subtracts a user-supplied threshold from every observation before fitting. A positive value shifts the data left toward zero. All fitters then operate on the shifted data, and the shift is restored in the Simio expressions afterward.

### Workflow

Run `fit_dists` twice and compare the results:

```python
report_raw     = fit_dists(data)                  # baseline, no shift
report_shifted = fit_dists(data, loc_shift=100.0) # known minimum of 100
```

Distributions that were failing purely because of the offset will show a marked improvement in p-value; distributions that were already fitting well will be largely unaffected.

### What changes with loc_shift != 0

- **Descriptive statistics** reflect the shifted data (mean, min, max, etc. are all reduced by `loc_shift`). The section header is annotated to make this explicit:

  ```
  -- Descriptive Statistics (data shifted by -100) ----
  ```

- **Fitted parameters** are estimated from the shifted data and describe the shifted distribution.

- **Simio expressions** are restored to the original scale by prepending the shift:

  ```
  100 + Random.Weibull(2.01, 5.03)
  ```

- The unshifted run (`loc_shift=0`) is unchanged in all respects, providing the baseline for comparison.

### Edge case

`loc_shift` must be strictly less than `min(data)`. If `loc_shift >= min(data)`, subtraction would leave non-positive values, which are invalid for Weibull and lognormal. A `ValueError` is raised immediately with a message identifying the offending shift value and the sample minimum.

---

## API Reference

### `fit_dists()`

```python
fit_dists(
    data: list[float] | np.ndarray,
    *,
    verbose: bool = True,
    loc_shift: float = 0.0,
) -> FitReport
```

Fit all six distributions to `data` and return a `FitReport`.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | `list[float]` or `np.ndarray` | -- | 1-D array-like of observations. Must have >= 3 finite values. |
| `verbose` | `bool` | `True` | If `True`, print the formatted report to stdout. |
| `loc_shift` | `float` | `0.0` | Known minimum threshold to subtract from every observation before fitting. Must be strictly less than `min(data)`. See [Location Shift](#location-shift). |

**Returns** -- `FitReport`

**Raises**

| Exception | Condition |
|---|---|
| `ValueError` | `data` is not 1-D |
| `ValueError` | Fewer than 3 observations |
| `ValueError` | Data contains `NaN` or `inf` |
| `ValueError` | `loc_shift >= min(data)` (would leave non-positive values after subtraction) |

---

### `FitReport`

```python
@dataclass
class FitReport:
    descriptive: DescriptiveStats
    fits: list[FitResult]
```

Top-level result object returned by `fit_dists()`.

| Attribute | Type | Description |
|---|---|---|
| `descriptive` | `DescriptiveStats` | Summary statistics of the (possibly shifted) data |
| `fits` | `list[FitResult]` | One `FitResult` per distribution, in fitting order |

`str(report)` produces the full formatted console report.

---

### `FitResult`

```python
@dataclass
class FitResult:
    distribution: str
    params: dict[str, float]
    ks_statistic: float
    p_value: float
    success: bool
    error: str | None
    simio_expression: str | None
```

| Attribute | Type | Description |
|---|---|---|
| `distribution` | `str` | Distribution name (e.g. `"normal"`, `"weibull"`) |
| `params` | `dict[str, float]` | MLE parameter estimates with human-readable keys; reflect the shifted data when `loc_shift != 0` |
| `ks_statistic` | `float` | KS test statistic (`nan` on failure) |
| `p_value` | `float` | KS test p-value (`nan` on failure) |
| `success` | `bool` | `False` if fitting raised an exception or data constraints were not met |
| `error` | `str \| None` | Error message when `success=False`, otherwise `None` |
| `simio_expression` | `str \| None` | Ready-to-use Simio expression string in the original (unshifted) scale, or `None` on failure |

---

### `DescriptiveStats`

```python
@dataclass
class DescriptiveStats:
    n: int
    mean: float
    std: float        # sample std (ddof=1)
    variance: float
    median: float
    skewness: float
    kurtosis: float   # excess kurtosis (Fisher definition)
    minimum: float
    maximum: float
    loc_shift: float  # shift subtracted before fitting; 0.0 means no shift
```

All statistics are computed over the data as presented to the fitters -- that is, after subtracting `loc_shift` when it is nonzero.

- `std` and `variance` use **ddof=1** (sample, not population).
- `skewness` and `kurtosis` use the **bias-corrected** (unbiased) estimators via `scipy.stats.skew` and `scipy.stats.kurtosis`.
- `kurtosis` is **excess kurtosis** (normal distribution = 0).
- `loc_shift` is stored on the dataclass and used only to annotate the printed header; it does not affect any computed value.

---

## Sample Output

### No shift (loc_shift=0)

```
---------------------------------------------------------------
-- Descriptive Statistics -------------------------------------
  n          : 200
  mean       : 2.66919
  std (s)    : 1.71803
  variance   : 2.95162
  median     : 2.33099
  skewness   : 1.0015
  kurtosis   : 1.14737  (excess)
  min        : 0.229362
  max        : 9.26985
---------------------------------------------------------------
-- Distribution Fits ------------------------------------------
  normal        : mu=2.66919,  sigma=1.71373
                  KS=0.1100,  p=0.0146  [poor fit]
                  Simio: Random.Normal(2.66919, 1.71373)
  uniform       : a (loc)=0.229362,  b (loc+scale)=9.26985
                  KS=0.3940,  p=0.0000  [poor fit]
                  Simio: Random.Uniform(0.229362, 9.26985)
  exponential   : loc=0.229362,  scale (1/lambda)=2.43983,  lambda=0.409865
                  KS=0.1409,  p=0.0006  [poor fit]
                  Simio: Random.Exponential(2.43983)
  triangular    : a (min)=0.157336,  b (max)=9.385,  c (mode)=0.512252
                  KS=0.1680,  p=0.0000  [poor fit]
                  Simio: Random.Triangular(0.157336, 0.512252, 9.385)
  weibull       : k (shape)=1.62598,  lambda (scale)=2.99069,  loc=0
                  KS=0.0614,  p=0.4204  [good fit]
                  Simio: Random.Weibull(1.62598, 2.99069)
  lognormal     : mu_ln=0.749777,  sigma_ln=0.729824,  loc=0
                  KS=0.0893,  p=0.0775  [good fit]
                  Simio: Random.Lognormal(0.749777, 0.729824)
---------------------------------------------------------------
```

### With shift (loc_shift=100)

Descriptive statistics and fitted parameters reflect the shifted data.
Simio expressions are restored to the original scale.

```
---------------------------------------------------------------
-- Descriptive Statistics (data shifted by -100) ----
  n          : 200
  mean       : 2.66919
  std (s)    : 1.71803
  variance   : 2.95162
  median     : 2.33099
  skewness   : 1.0015
  kurtosis   : 1.14737  (excess)
  min        : 0.229362
  max        : 9.26985
---------------------------------------------------------------
-- Distribution Fits ------------------------------------------
  normal        : mu=2.66919,  sigma=1.71373
                  KS=0.1100,  p=0.0146  [poor fit]
                  Simio: 100 + Random.Normal(2.66919, 1.71373)
  uniform       : a (loc)=0.229362,  b (loc+scale)=9.26985
                  KS=0.3940,  p=0.0000  [poor fit]
                  Simio: 100 + Random.Uniform(0.229362, 9.26985)
  exponential   : loc=0.229362,  scale (1/lambda)=2.43983,  lambda=0.409865
                  KS=0.1409,  p=0.0006  [poor fit]
                  Simio: 100 + Random.Exponential(2.43983)
  triangular    : a (min)=0.157336,  b (max)=9.385,  c (mode)=0.512252
                  KS=0.1680,  p=0.0000  [poor fit]
                  Simio: 100 + Random.Triangular(0.157336, 0.512252, 9.385)
  weibull       : k (shape)=1.62598,  lambda (scale)=2.99069,  loc=0
                  KS=0.0614,  p=0.4204  [good fit]
                  Simio: 100 + Random.Weibull(1.62598, 2.99069)
  lognormal     : mu_ln=0.749777,  sigma_ln=0.729824,  loc=0
                  KS=0.0893,  p=0.0775  [good fit]
                  Simio: 100 + Random.Lognormal(0.749777, 0.729824)
---------------------------------------------------------------
```

---

## Notes and Limitations

- **KS test validity** -- The KS p-values are computed using the MLE-fitted parameters rather than known true parameters, which makes the test slightly anti-conservative (p-values tend to be somewhat inflated). For rigorous testing consider a parametric bootstrap.
- **Weibull and Lognormal** require strictly positive data (`x > 0`). Passing data with zeros or negative values will produce a `FitResult` with `success=False` for those distributions; the remaining distributions will still be fitted normally. The same constraint applies after subtracting `loc_shift`.
- **Exponential `loc`** -- Fixing `loc` to the sample minimum means the fitted distribution has its support starting at `min(data)`, not at zero. When `loc_shift` is used, `min(data)` refers to the shifted minimum, which will be close to zero if the shift is chosen well.
- **loc_shift and descriptive statistics** -- All values in `DescriptiveStats` reflect the shifted data. To recover unshifted summary statistics, add `loc_shift` back to `mean`, `median`, `minimum`, and `maximum`; `std`, `variance`, `skewness`, and `kurtosis` are shift-invariant and require no adjustment.
- **SciPy warnings** -- Optimizer convergence warnings from SciPy are suppressed internally. If a fit fails silently, check `FitResult.success` and `FitResult.error`.
