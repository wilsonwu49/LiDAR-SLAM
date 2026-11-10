import numpy as np
from scipy.stats import chi2


def nis(y: np.ndarray, S: np.ndarray) -> float:
    """Return the normalized innovation squared"""
    z = np.linalg.solve(S,y)
    return y.T @ z

def nis_bounds(m: int, N: int, alpha: float = 0.05):
    """Return (low, high) two-sided bounds on the *average* NIS over N steps
    of measurement dim m, using the chi-square(N*m) quantiles / N.
    Use scipy.stats.chi2.ppf(alpha/2, N*m)/N and ppf(1-alpha/2, N*m)/N.
    """
    low  = chi2.ppf(alpha / 2, N * m) / N
    high = chi2.ppf(1 - alpha / 2, N * m) / N
    return low, high