"""
Build script for the TRCC C++ extension (`trcc_native`).

The pure-Python implementation falls back gracefully when the extension
is unavailable, so a missing C++ toolchain at install time does NOT
break the package — it just disables the kd-tree fast path. Most package
metadata (name, version, deps) is in `pyproject.toml`; this file exists
only to declare the optional native extension.

Cross-platform compile flags:
- POSIX (Linux/macOS):    -O3 -ffast-math
- Windows MSVC:           /O2 /fp:fast
- OpenMP is opt-in via env var `TRCC_BUILD_OMP=1` (off by default
  because Apple clang ships without libomp on macOS).
"""
from __future__ import annotations

import os
import platform
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


def _compile_args() -> list[str]:
    if sys.platform == "win32":
        args = ["/O2", "/fp:fast", "/EHsc"]
    else:
        args = ["-O3", "-ffast-math", "-fvisibility=hidden"]
        # -funroll-loops on GCC/Clang gives ~5-10% on the inner kd-tree query
        args.append("-funroll-loops")
    if os.environ.get("TRCC_BUILD_OMP") == "1":
        if sys.platform == "win32":
            args.append("/openmp")
        else:
            args.append("-fopenmp")
    return args


def _link_args() -> list[str]:
    if os.environ.get("TRCC_BUILD_OMP") == "1" and sys.platform != "win32":
        return ["-fopenmp"]
    return []


def _make_extension() -> Pybind11Extension:
    # Namespaced as `trcc.trcc_native` so the compiled .so/.pyd is
    # installed *inside* the trcc package directory rather than at the
    # site-packages root. This keeps `from . import trcc_native` working
    # for end users and avoids polluting the global namespace.
    return Pybind11Extension(
        name="trcc.trcc_native",
        sources=["trcc/_native/trcc_ext.cpp"],
        include_dirs=["trcc/_native/third_party"],
        cxx_std=17,
        extra_compile_args=_compile_args(),
        extra_link_args=_link_args(),
    )


# Allow installing without the C++ extension if the toolchain is missing
# or the user opts out via TRCC_NO_EXTENSION=1. The Python fallback is
# bit-identical in output, just slower.
ext_modules: list[Pybind11Extension] = []
if os.environ.get("TRCC_NO_EXTENSION") != "1":
    try:
        ext_modules = [_make_extension()]
    except Exception as e:  # pragma: no cover
        print(f"[trcc setup] WARNING: extension setup failed ({e!r}); "
              f"falling back to pure Python.", file=sys.stderr)
        ext_modules = []

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
