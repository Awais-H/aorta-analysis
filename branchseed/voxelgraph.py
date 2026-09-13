"""Sparse 26-connectivity graphs over a boolean voxel set.

Shared by the centreline extraction in Stage 3 and the geodesic trace in
Stage 5. Edge weights are physical distances, so a path length in the graph is
a length in millimetres.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy import sparse

# The 13 unique 26-connectivity offsets; the other 13 are their negatives.
OFFSETS: List[Tuple[int, int, int]] = [
    (dz, dy, dx)
    for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
    if (dz, dy, dx) > (0, 0, 0)
]

NEIGHBOURHOOD_26 = np.ones((3, 3, 3), dtype=bool)


def voxel_graph(mask_zyx: np.ndarray, spacing_zyx: np.ndarray):
    """``(coords_zyx, graph_csr)`` over the True voxels of ``mask_zyx``.

    Row *i* of ``coords_zyx`` is the voxel of node *i*. The graph is symmetric
    and weighted by the physical distance between voxel centres, which is
    1, sqrt(2) or sqrt(3) times the spacing for an isotropic grid and the
    anisotropic equivalent otherwise.
    """
    coords = np.argwhere(mask_zyx)
    n = coords.shape[0]
    if n == 0:
        return coords, sparse.csr_matrix((0, 0))

    index = np.full(mask_zyx.shape, -1, dtype=np.int64)
    index[coords[:, 0], coords[:, 1], coords[:, 2]] = np.arange(n)

    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    data: List[np.ndarray] = []
    for dz, dy, dx in OFFSETS:
        src = index[
            max(0, -dz): index.shape[0] - max(0, dz),
            max(0, -dy): index.shape[1] - max(0, dy),
            max(0, -dx): index.shape[2] - max(0, dx),
        ]
        dst = index[
            max(0, dz): index.shape[0] + min(0, dz),
            max(0, dy): index.shape[1] + min(0, dy),
            max(0, dx): index.shape[2] + min(0, dx),
        ]
        ok = (src >= 0) & (dst >= 0)
        if not ok.any():
            continue
        weight = float(np.linalg.norm(np.array([dz, dy, dx], dtype=np.float64) * spacing_zyx))
        rows.append(src[ok])
        cols.append(dst[ok])
        data.append(np.full(int(ok.sum()), weight))

    if not rows:
        return coords, sparse.csr_matrix((n, n))
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    w = np.concatenate(data)
    graph = sparse.coo_matrix(
        (np.concatenate([w, w]), (np.concatenate([r, c]), np.concatenate([c, r]))),
        shape=(n, n),
    ).tocsr()
    return coords, graph
