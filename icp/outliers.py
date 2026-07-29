import numpy as np

def robust_weights(residuals: np.ndarray, delta: float = None) -> np.ndarray:
    """Per-correspondence weights from a robust kernel, using huber.\n
    delta = c * MAD(residuals), makes MAD comparable to a normal distribution's standard deviation for well-behaved data
    """
    abs_res = np.abs(residuals) 

    if delta is None:
        median_res = np.median(residuals)
        mad = np.median(np.abs(residuals - median_res))
        delta = 1.4826 * mad                        # adaptive cutoff, scaled from this data's own spread

    weights = np.where(abs_res <= delta, 1.0, delta / (abs_res + 1e-12))

    return weights

def trim_correspondences(residuals: np.ndarray, keep_fraction: float):
    """Return a boolean mask keeping the keep_fraction smallest-residual pairs."""
    abs_res = np.abs(residuals) 
    threshold = np.percentile(abs_res, 100 * keep_fraction)
    mask = abs_res <= threshold
    return mask