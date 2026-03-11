"""Wall curvature computation along a grid i-line."""

# --------------------------------------------------
# load necessary libraries
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from lst_tools.core import Grid


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)

# --------------------------------------------------
# function to compute curvature at the wall
# --------------------------------------------------
def curvature(
    grid: Grid,
    *,
    j: int = 0,
    smooth: bool = True,
    debug_path: Path | str | None = None,
    method: str = "spline",
    method_params: dict[str, Any] | None = None,
) -> np.ndarray:

    """Compute wall curvature along an i-line.

    Treats the wall as y(x) and computes:
        kappa = -y'' / (1 + (y')^2)^(3/2)

    Args:
        grid: Structured grid with shape (ny, nx).
        j: Row index (0..ny-1) along which to compute curvature.
        smooth: If True, apply smoothing to the raw curvature.
        debug_path: Directory for debug output (Tecplot ASCII).
        method: Smoothing method ("spline", "savgol", "gaussian", "robust").
        method_params: Extra keyword arguments passed to the smoother.

    Returns:
        1-D array of curvature values along the i-line.
    """

    # get grid dimensions
    ny, nx = grid.shape

    # check row index j for which the curvature is computed (note: default j = 0)
    if not (0 <= j < ny):
        raise IndexError(f"row index j={j} out of range [0,{ny - 1}]")

    # get x and y coordinates for specified j location to compute curvature (default j = 0)
    x = np.asarray(grid.x[j, :], dtype=float)
    y = np.asarray(grid.y[j, :], dtype=float)

    # compute first and second derivatives of y with respect to x (handles nonuniform x)
    yp = np.gradient(y, x, edge_order=2)
    ypp = np.gradient(yp, x, edge_order=2)

    # compute raw (unsmoothed) curvature
    denom = (1.0 + yp**2.0) ** 1.5
    kappa = -ypp / denom

    # optional smoothing (different methods)
    if smooth:
        # get keyword arguments for smoothing method (either empty if none provided or as specified in method_params dictionary)
        params = dict(method_params or {})
        # smooth curvature
        kappa_smoothed = smooth_kappa(x, kappa, method=method, **params)
    else:
        # no smoothing, return raw curvature in kappa_smoothed (used downstream)
        kappa_smoothed = kappa

    # debug output
    if debug_path is not None:

        # normalize to a Path and ensure directory exists
        dbg_dir = Path(debug_path)
        dbg_dir.mkdir(parents=True, exist_ok=True)

        # write tecplot readable ascii file
        from lst_tools.data_io.tecplot_ascii import write_tecplot_ascii

        write_tecplot_ascii(
            dbg_dir / "curvature.dat",
            {"x": x, "y": y, "yp": yp, "ypp": ypp, "kappa": kappa, "kappa_smoothed": kappa_smoothed},
            title="curvature debug",
            zone="row_j",
        )

    # return solution to calling routine
    return kappa_smoothed


# --------------------------------------------------
# smoothing helper functions: Savitzky–Golay filter (with fallback)
# --------------------------------------------------
def smooth_savgol(
    kappa: np.ndarray, window_frac: float = 0.03, polyorder: int = 3
) -> np.ndarray:
    """Savitzky-Golay smoothing with moving-average fallback.

    Args:
        kappa: Raw curvature array.
        window_frac: Window size as a fraction of array length (forced odd).
        polyorder: Polynomial order (typically 2-3).

    Returns:
        Smoothed curvature array.
    """
    n = len(kappa)
    if n == 0:
        return kappa
    w = max(3, int(n * window_frac) | 1)
    w = min(w, n if n % 2 == 1 else n - 1)
    try:
        from scipy.signal import savgol_filter

        return savgol_filter(
            kappa, window_length=w, polyorder=min(polyorder, w - 1), mode="interp"
        )
    except ImportError:
        # Fallback: moving average
        kernel = np.ones(w) / w
        pad = w // 2
        return np.convolve(np.pad(kappa, pad, mode="edge"), kernel, mode="valid")


# --------------------------------------------------
# smoothing helper functions: gaussian
# --------------------------------------------------
def smooth_gaussian(kappa: np.ndarray, sigma_frac: float = 0.02) -> np.ndarray:
    """Gaussian smoothing with moving-average fallback.

    Args:
        kappa: Raw curvature array.
        sigma_frac: Gaussian sigma as a fraction of array length.

    Returns:
        Smoothed curvature array.
    """
    n = len(kappa)
    if n == 0:
        return kappa
    try:
        from scipy.ndimage import gaussian_filter1d

        sigma = max(1.0, n * sigma_frac)
        return gaussian_filter1d(kappa, sigma=sigma, mode="nearest")
    except ImportError:
        w = max(3, int(n * sigma_frac * 6) | 1)
        w = min(w, n if n % 2 == 1 else n - 1)
        kernel = np.ones(w) / w
        pad = w // 2
        return np.convolve(np.pad(kappa, pad, mode="edge"), kernel, mode="valid")


# --------------------------------------------------
# smoothing helper functions: spline
# --------------------------------------------------
def smooth_spline(
    x: np.ndarray, kappa: np.ndarray, s_factor: float = 1e-2
) -> np.ndarray:
    """Smoothing spline on non-uniform x with moving-average fallback.

    The smoothing parameter *s* is derived from a median-based noise
    estimate scaled by *s_factor*.

    Args:
        x: Coordinate array (non-uniform spacing OK).
        kappa: Raw curvature array.
        s_factor: Scale factor for the smoothing parameter.

    Returns:
        Smoothed curvature array.
    """
    n = len(kappa)
    if n == 0:
        return kappa
    try:
        from scipy.interpolate import UnivariateSpline

        # crude noise estimate
        if n > 1:
            sigma = float(np.median(np.abs(np.diff(kappa)))) / np.sqrt(2.0)
        else:
            sigma = 0.0
        s = float(s_factor) * (sigma**2) * n
        spl = UnivariateSpline(x, kappa, s=s)
        return spl(x)
    except ImportError:
        pass
    except Exception:
        # UnivariateSpline may raise non-standard _dfitpack.error
        # on degenerate data -- fall back to moving average.
        logger.debug("spline smoothing failed, falling back to moving average")

    # Fallback: gentle moving average (~1% of length)
    w = max(3, int(n * 0.01) | 1)
    w = min(w, n if n % 2 == 1 else n - 1)
    kernel = np.ones(w) / w
    pad = w // 2
    return np.convolve(np.pad(kappa, pad, mode="edge"), kernel, mode="valid")


# --------------------------------------------------
# smoothing helper functions: median filter + gaussian smoothing
# --------------------------------------------------
def smooth_robust(
    kappa: np.ndarray, median_frac: float = 0.01, gauss_frac: float = 0.02
) -> np.ndarray:
    """Median filter to remove spikes, then Gaussian smooth.

    Args:
        kappa: Raw curvature array.
        median_frac: Median filter window as a fraction of array length.
        gauss_frac: Gaussian sigma fraction passed to ``smooth_gaussian``.

    Returns:
        Smoothed curvature array.
    """

    n = len(kappa)
    if n == 0:
        return kappa
    mwin = max(3, int(n * median_frac) | 1)
    mwin = min(mwin, n if n % 2 == 1 else n - 1)
    # median
    try:
        from scipy.signal import medfilt

        med = medfilt(kappa, kernel_size=mwin)
    except ImportError:
        pad = mwin // 2
        padded = np.pad(kappa, pad, mode="edge")
        med = np.array([np.median(padded[i : i + mwin]) for i in range(n)])
    # gaussian
    return smooth_gaussian(med, sigma_frac=gauss_frac)


# --------------------------------------------------
# curvature computation and smoothing dispatch
# --------------------------------------------------
def smooth_kappa(
    x: np.ndarray, kappa: np.ndarray, *, method: str = "spline", **kwargs
) -> np.ndarray:
    """Dispatch to a chosen smoothing method.

    Args:
        x: Coordinate array.
        kappa: Raw curvature array.
        method: One of "spline", "savgol", "gaussian", "robust".
        **kwargs: Forwarded to the chosen smoother.

    Returns:
        Smoothed curvature array.
    """
    m = (method or "spline").lower()
    if m == "savgol":
        return smooth_savgol(kappa, **kwargs)
    elif m == "gaussian":
        return smooth_gaussian(kappa, **kwargs)
    elif m == "robust":
        return smooth_robust(kappa, **kwargs)
    elif m == "spline":
        return smooth_spline(x, kappa, **kwargs)
    else:
        raise ValueError(f"unknown smoothing method: {method!r}")
