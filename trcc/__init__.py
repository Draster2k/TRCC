"""TRCC — T-Regulated Cytokine Clustering.

Public API:
    TRCC               clustering estimator (sklearn-compatible)
    has_extension      bool flag — True if the C++ kd-tree fast path is loaded

The compiled extension (`trcc.trcc_native`) is namespaced inside this
package, so it lives next to `core.py` after install. If it is missing
(no compiler at install time, `TRCC_NO_EXTENSION=1`, or platform without
a wheel), `has_extension` is False and `core.py` transparently falls
back to a bit-identical pure-Python implementation.
"""
from __future__ import annotations

__version__ = "1.1.0"


# --- Native-extension probe -------------------------------------------------
#
# We try the relative import first (post-namespacing). For backwards
# compatibility with installs that placed the .so at the site-packages
# root (pre-1.1.0), we fall back to a top-level import. Either path sets
# `_native_module` and `has_extension`.
try:
    from . import trcc_native as _native_module  # type: ignore[attr-defined]
    has_extension: bool = True
except ImportError:
    try:
        import trcc_native as _native_module  # type: ignore[no-redef]
        has_extension = True
    except ImportError:
        _native_module = None  # type: ignore[assignment]
        has_extension = False


# --- Public surface ---------------------------------------------------------
from .core import TRCC

__all__ = ["TRCC", "has_extension", "__version__"]
