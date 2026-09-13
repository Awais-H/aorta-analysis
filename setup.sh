#!/usr/bin/env bash
# Single setup command for Branchseed.
#
# Installs from the vendored wheels in vendor/ when they are present, so the
# install works on a machine with no internet access. Falls back to PyPI when
# vendor/ is empty. See tools/vendor_wheels.sh to populate vendor/ for the
# target platform before submitting.
set -euo pipefail

PYTHON="${PYTHON:-python3}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if compgen -G "${here}/vendor/*.whl" > /dev/null; then
    echo "Installing from vendored wheels (offline)."
    "${PYTHON}" -m pip install --no-index --find-links "${here}/vendor" -r "${here}/requirements.txt"
else
    echo "vendor/ holds no wheels; installing from PyPI."
    echo "For an offline install, run tools/vendor_wheels.sh first."
    "${PYTHON}" -m pip install -r "${here}/requirements.txt"
fi

echo "Branchseed dependencies installed."
