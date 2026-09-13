"""Rotation-minimising frame along a discrete curve.

Double-reflection method of Wang et al., "Computation of rotation minimizing
frames", ACM TOG 2008. Second-order accurate and unconditionally stable.

Why not the alternatives:

* a fixed global vector projected into the normal plane degenerates wherever
  the tangent turns parallel to it, which is guaranteed somewhere on an aortic
  arch, and produces a discontinuous jump in the unrolled map;
* the Frenet frame flips at inflection points and is undefined where curvature
  is zero, i.e. along most of a straight aortic segment.
"""

from __future__ import annotations

import numpy as np


def _normalise(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def initial_normal(tangent0: np.ndarray) -> np.ndarray:
    """Any unit vector perpendicular to the first tangent."""
    t = _normalise(np.asarray(tangent0, dtype=np.float64))
    seed = np.array([0.0, 0.0, 1.0]) if abs(t[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = seed - np.dot(seed, t) * t
    return _normalise(u)


def rotation_minimising_frame(points: np.ndarray, tangents: np.ndarray) -> tuple:
    """Propagate an RMF along a polyline.

    Parameters
    ----------
    points : (N, 3) curve samples.
    tangents : (N, 3) unit tangents at those samples.

    Returns ``(u, v)``, each (N, 3), with ``u x v = t`` to numerical precision.
    """
    pts = np.asarray(points, dtype=np.float64)
    tan = _normalise(np.asarray(tangents, dtype=np.float64))
    n = pts.shape[0]
    u = np.zeros((n, 3), dtype=np.float64)
    u[0] = initial_normal(tan[0])

    for i in range(n - 1):
        # Reflection 1: reflect the frame in the plane bisecting the two points.
        v1 = pts[i + 1] - pts[i]
        c1 = float(v1 @ v1)
        if c1 < 1e-18:
            u[i + 1] = u[i]
            continue
        u_l = u[i] - (2.0 / c1) * float(v1 @ u[i]) * v1
        t_l = tan[i] - (2.0 / c1) * float(v1 @ tan[i]) * v1
        # Reflection 2: reflect again to land on the next tangent.
        v2 = tan[i + 1] - t_l
        c2 = float(v2 @ v2)
        if c2 < 1e-18:
            u[i + 1] = _normalise(u_l)
            continue
        u_next = u_l - (2.0 / c2) * float(v2 @ u_l) * v2
        # Re-orthogonalise against drift accumulated over hundreds of samples.
        u_next = u_next - float(u_next @ tan[i + 1]) * tan[i + 1]
        u[i + 1] = _normalise(u_next)

    v = np.cross(tan, u)
    return _normalise(u), _normalise(v)


def rotate_frame(u: np.ndarray, v: np.ndarray, angle: float) -> tuple:
    """Rotate the whole frame about its tangent by a single global angle."""
    c, s = np.cos(angle), np.sin(angle)
    return c * u + s * v, -s * u + c * v


def align_to_reference(
    u: np.ndarray, v: np.ndarray, tangents: np.ndarray, reference: np.ndarray, at: int
) -> tuple:
    """Apply one global rotation so theta = 0 points along ``reference``.

    ``reference`` is projected into the normal plane at sample ``at``; the frame
    is rotation-minimising, so a single global rotation fixes theta = 0 for the
    whole curve without reintroducing twist.
    """
    ref = np.asarray(reference, dtype=np.float64)
    t = tangents[at]
    proj = ref - float(ref @ t) * t
    norm = np.linalg.norm(proj)
    if norm < 1e-6:
        # Reference is parallel to the tangent here; nothing meaningful to
        # align to at this sample, so leave the frame as propagated.
        return u, v
    proj /= norm
    angle = np.arctan2(float(proj @ v[at]), float(proj @ u[at]))
    return rotate_frame(u, v, angle)
