#!/usr/bin/env bash
# Local install verification for the namespaced C++ extension.
# Run from the repo root:
#     bash scripts/local_test.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "▶ 1. Wiping build artifacts"
rm -rf build/ dist/ ./*.egg-info trcc/*.egg-info
find . -type d -name __pycache__ -prune -exec rm -rf {} +
# Remove any previously-installed top-level .so to confirm namespacing
find . -maxdepth 2 -name 'trcc_native*.so' -not -path './.venv/*' -delete 2>/dev/null || true
echo "  ✓ clean"

echo "▶ 2. Building wheel + sdist via PEP 517"
python3 -m build >/tmp/trcc_build.log 2>&1 || { tail -40 /tmp/trcc_build.log; exit 1; }
ls -la dist/
WHEEL="$(ls -t dist/*.whl | head -1)"
echo "  ✓ built $WHEEL"

echo "▶ 3. Reinstalling wheel"
# `pip uninstall` can choke on stale editable installs missing RECORD; nuke
# the install directory directly to guarantee a clean slate.
SITE="$(python3 -c 'import site, sys; print(site.getsitepackages()[0] if site.getsitepackages() else sys.prefix)')"
rm -rf "$SITE"/trcc "$SITE"/trcc-*.dist-info "$SITE"/trcc.egg-link \
       "$SITE"/__editable__.trcc-*.pth "$SITE"/__editable___trcc_*finder.py 2>/dev/null || true
pip install --no-deps --ignore-installed "$WHEEL" >/tmp/trcc_install.log 2>&1 || {
    tail -40 /tmp/trcc_install.log; exit 1; }
echo "  ✓ installed"

echo "▶ 4. Verifying install layout from outside the source tree"
cd /tmp
python3 - <<'PY'
import os, trcc, importlib.util

loc = os.path.dirname(trcc.__file__)
print(f"trcc install location : {loc}")
print(f"trcc.__version__      : {trcc.__version__}")
print(f"trcc.has_extension    : {trcc.has_extension}")
print()
print("Files inside the package directory:")
for f in sorted(os.listdir(loc)):
    full = os.path.join(loc, f)
    tag = "<dir>" if os.path.isdir(full) else f"{os.path.getsize(full):>10d} B"
    print(f"  {tag}  {f}")

# Resolve the exact native module file when present
spec = importlib.util.find_spec("trcc.trcc_native")
print()
if spec and spec.origin:
    print(f"trcc.trcc_native      : {spec.origin}")
    assert spec.origin.startswith(loc), "native ext is OUTSIDE the package directory!"
    print("  ✓ native extension lives inside the trcc/ package")
else:
    print("trcc.trcc_native      : not loaded (pure-Python fallback active)")
PY
