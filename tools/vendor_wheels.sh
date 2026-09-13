#!/usr/bin/env bash
# Populate vendor/ so that setup.sh can install with no internet access.
#
# Run this on a connected machine, targeting the interpreter and platform of
# the evaluation box, then commit vendor/. Defaults target 64-bit Linux, which
# is the usual evaluation environment; override for anything else.
#
#   tools/vendor_wheels.sh                                  # linux x86_64, py3.11
#   tools/vendor_wheels.sh --python-version 3.12
#   tools/vendor_wheels.sh --platform macosx_11_0_arm64 --python-version 3.11
#   tools/vendor_wheels.sh --native                         # this machine exactly
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
platform="manylinux2014_x86_64"
python_version="3.11"
native=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform) platform="$2"; shift 2 ;;
        --python-version) python_version="$2"; shift 2 ;;
        --native) native=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "${here}/vendor"
if [[ "${native}" == "1" ]]; then
    echo "Downloading wheels for this machine's interpreter."
    "${PYTHON}" -m pip download -r "${here}/requirements.txt" -d "${here}/vendor"
else
    echo "Downloading wheels for platform=${platform} python=${python_version}."
    "${PYTHON}" -m pip download -r "${here}/requirements.txt" -d "${here}/vendor" \
        --platform "${platform}" --python-version "${python_version}" --only-binary=:all:
fi

echo
echo "vendor/ now holds $(ls -1 "${here}/vendor"/*.whl 2>/dev/null | wc -l | tr -d ' ') wheels,"
echo "$(du -sh "${here}/vendor" | cut -f1) total."
echo
echo "vendor/*.whl is gitignored so a wrong-platform download cannot be committed by"
echo "accident. When these are the right wheels, add them explicitly and verify:"
echo "  git add -f vendor/*.whl"
echo "  ./setup.sh          # with the network disabled"
