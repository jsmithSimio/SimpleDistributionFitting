#
# Computes parameters for a Weibull distribution with a
# given mean and standard deviation.
#
from scipy.optimize import brentq
from scipy.special import gamma
import numpy as np

mu, sigma = 100, 15
cv = sigma / mu  # 0.15

def cv_equation(k):
    return gamma(1 + 2/k) / gamma(1 + 1/k)**2 - (cv**2 + 1)

# gamma(1 + 2/k) overflows for k < ~0.5, so start the search there
k = brentq(cv_equation, 0.5, 1000)
lam = mu / gamma(1 + 1/k)

print(f"k (shape) = {k:.6f}")
print(f"λ (scale) = {lam:.6f}")
