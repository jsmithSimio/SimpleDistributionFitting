"""
Compute lognormal distribution parameters (mu, sigma) from
the mean and standard deviation of the observed (non-log) data.

For X ~ Lognormal(mu, sigma), where mu and sigma are the mean
and std of ln(X):

    E[X]   = exp(mu + sigma^2 / 2)
    Var[X] = (exp(sigma^2) - 1) * exp(2*mu + sigma^2)

Solving for mu and sigma given E[X] and Std[X]:

    sigma^2 = ln(1 + (std / mean)^2)
    mu      = ln(mean) - sigma^2 / 2
"""

import math
import scipy.stats as stats


def lognormal_params_from_moments(mean: float, std: float) -> tuple[float, float]:
    """
    Return (mu, sigma) of the underlying normal (i.e., of ln(X))
    given the mean and std of the observed lognormal variable X.
    """
    if mean <= 0:
        raise ValueError(f"mean must be positive, got {mean}")
    if std <= 0:
        raise ValueError(f"std must be positive, got {std}")

    cv2 = (std / mean) ** 2           # squared coefficient of variation
    sigma2 = math.log(1.0 + cv2)      # variance of ln(X)
    sigma = math.sqrt(sigma2)
    mu = math.log(mean) - sigma2 / 2  # mean of ln(X)
    return mu, sigma


def verify_params(mu: float, sigma: float) -> tuple[float, float]:
    """Back-compute mean and std from (mu, sigma) as a sanity check."""
    mean_back = math.exp(mu + sigma**2 / 2)
    std_back = math.sqrt((math.exp(sigma**2) - 1) * math.exp(2 * mu + sigma**2))
    return mean_back, std_back


def main():
    mean = 100.0
    std = 15.0

    mu, sigma = lognormal_params_from_moments(mean, std)

    mean_back, std_back = verify_params(mu, sigma)

    # scipy uses s=sigma, scale=exp(mu) for lognormal_params
    scipy_s = sigma
    scipy_scale = math.exp(mu)

    print("Lognormal Parameters from Moments")
    print("=" * 38)
    print(f"  Input mean : {mean}")
    print(f"  Input std  : {std}")
    print()
    print("Underlying normal (of ln(X)):")
    print(f"  mu    = {mu:.6f}")
    print(f"  sigma = {sigma:.6f}")
    print()
    print("scipy.stats.lognorm parameterization:")
    print(f"  s     = {scipy_s:.6f}   (shape, = sigma)")
    print(f"  scale = {scipy_scale:.6f}  (= exp(mu))")
    print()
    print("Verification (back-computed from parameters):")
    print(f"  mean  = {mean_back:.6f}  (expected {mean})")
    print(f"  std   = {std_back:.6f}  (expected {std})")
    print()

    # Show a few quantiles using scipy
    dist = stats.lognorm(s=scipy_s, scale=scipy_scale)
    print("Selected quantiles:")
    for p in [0.05, 0.25, 0.50, 0.75, 0.95]:
        print(f"  P{int(p*100):02d} = {dist.ppf(p):.4f}")


if __name__ == "__main__":
    main()