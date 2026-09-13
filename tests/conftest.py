"""Shared fixtures. Phantoms are session-scoped: rasterising one costs a few
seconds and every test in a module wants the same volume."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for path in (str(ROOT), str(ROOT / "tests")):
    if path not in sys.path:
        sys.path.insert(0, path)

import phantoms  # noqa: E402
from branchseed.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


def _phantom_fixture(builder, **kwargs):
    @pytest.fixture(scope="session")
    def _fixture():
        return phantoms.build_phantom(builder(**kwargs))
    return _fixture


standard_phantom = _phantom_fixture(phantoms.standard)
arched_phantom = _phantom_fixture(phantoms.arched)
nearby_pair_phantom = _phantom_fixture(phantoms.nearby_pair)
common_trunk_phantom = _phantom_fixture(phantoms.common_trunk)
daughter_of_daughter_phantom = _phantom_fixture(phantoms.daughter_of_daughter)
short_stub_phantom = _phantom_fixture(phantoms.short_stub)
leaking_phantom = _phantom_fixture(phantoms.leaking)
near_cut_face_phantom = _phantom_fixture(phantoms.near_cut_face)
no_branch_phantom = _phantom_fixture(phantoms.no_branches)
