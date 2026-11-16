"""Loader for the vendored "Let's Be Rational" native solver.

The C++ (greeks/native/lbr/, Peter Jäckel's reference, his licence preserved) is built by setup.py
into ``greeks/solver/_lbr<ext>.so``. This module loads it with ctypes and exposes the one batch entry
point. There is NO pure-Python fallback: if the library is missing or will not load, importing this
module — and therefore importing greeks.solver — raises NativeSolverUnavailable naming the file and
the reason, so a broken build fails loudly instead of silently reverting to a slower, different path.
"""

from __future__ import annotations

import ctypes
import glob
import os

import numpy as np
import numpy.typing as npt

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_SO_GLOB = os.path.join(_PKG_DIR, "_lbr*.so")


class NativeSolverUnavailable(RuntimeError):
    """The compiled Let's Be Rational library could not be found or loaded."""


def _load() -> "ctypes._NamedFuncPointer":
    matches = sorted(glob.glob(_SO_GLOB))
    if not matches:
        raise NativeSolverUnavailable(
            f"native 'Let's Be Rational' library not found (looked for {_SO_GLOB}). "
            "Build it with `pip install -e .` — needs a C++ compiler (build-essential / g++).")
    path = matches[0]
    try:
        lib = ctypes.CDLL(path)
    except OSError as e:
        raise NativeSolverUnavailable(f"failed to load native solver {path}: {e}") from e
    try:
        fn = lib.lbr_implied_vol_batch
    except AttributeError as e:
        raise NativeSolverUnavailable(
            f"native solver {path} has no symbol lbr_implied_vol_batch: {e}") from e
    f64 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    i32 = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_size_t, f64, f64, f64, f64, i32, f64]
    return fn


_LBR_BATCH = _load()  # at import: loud failure if the native library is absent


def implied_vol_batch(
    price: npt.NDArray[np.float64],
    forward: npt.NDArray[np.float64],
    strike: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    is_call: npt.NDArray[np.int32],
) -> npt.NDArray[np.float64]:
    """Undiscounted Black ``price`` → implied Black volatility, via Jäckel's reference routine. Writes
    NaN where the routine signals no solution (below-intrinsic / above-maximum). The caller has already
    applied the pre-checks and passes only the live rows, all 1-D arrays of equal length ``n``."""
    n = int(price.shape[0])
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    _LBR_BATCH(
        n,
        np.ascontiguousarray(price, dtype=np.float64),
        np.ascontiguousarray(forward, dtype=np.float64),
        np.ascontiguousarray(strike, dtype=np.float64),
        np.ascontiguousarray(t, dtype=np.float64),
        np.ascontiguousarray(is_call, dtype=np.int32),
        out,
    )
    return out
