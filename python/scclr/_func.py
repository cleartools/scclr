"""Functional, array-based API over the ``scclr._core`` Rust extension.

Works with bare numpy / scipy — no anndata required. The scverse in-place API
(:mod:`scclr.pp`, :mod:`scclr.tl`) is built on top of these functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from . import _core

__all__ = [
    "ShiftedCLR",
    "PCAResult",
    "overdispersion",
    "normalize",
    "pca",
    "normalize_pca",
]


def _as_csr(X):
    """Normalize any matrix-like to ``(data f64, indices i64, indptr i64, shape)``.

    scipy chooses int32 or int64 index dtypes depending on size; we always hand int64 to
    Rust so the boundary is uniform.
    """
    import scipy.sparse as sp

    if sp.issparse(X):
        csr = X.tocsr()
    else:
        csr = sp.csr_matrix(np.asarray(X, dtype=np.float64))
    csr.sort_indices()
    data = np.ascontiguousarray(csr.data, dtype=np.float64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int64)
    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    return data, indices, indptr, (int(csr.shape[0]), int(csr.shape[1]))


def _resolve_target(target):
    """Map a user ``target`` to the ``(name, fixed)`` pair the Rust layer expects."""
    if isinstance(target, bool):
        raise ValueError("target must be 'mean','median','auto', or a number")
    if isinstance(target, (int, float)):
        return "fixed", float(target)
    if target in ("mean", "median", "auto"):
        return target, None
    raise ValueError(f"target must be 'mean','median','auto', or a number; got {target!r}")


@dataclass
class ShiftedCLR:
    """Sparse shifted-CLR result: the PFlog values plus the per-cell mean vector.

    The dense value is ``sparse[i, j] - row_center[i]``. Kept sparse so PCA runs without
    densifying.
    """

    sparse: "object"  # scipy.sparse.csr_matrix
    row_center: np.ndarray
    k: Optional[float] = None
    alpha: Optional[float] = None

    @property
    def shape(self):
        return self.sparse.shape

    def to_dense(self) -> np.ndarray:
        """Materialize ``sparse - row_center[:, None]`` (densifies — for small data / inspection)."""
        return np.asarray(self.sparse.todense()) - np.asarray(self.row_center)[:, None]


@dataclass
class PCAResult:
    """Sparse PCA result, mirroring scanpy/sklearn fields."""

    scores: np.ndarray  # (n_obs, n_comps)
    components: np.ndarray  # (n_comps, n_vars)
    mean: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    singular_values: np.ndarray
    noise_variance: float
    n_samples: int
    n_features: int
    n_components: int
    k: Optional[float] = None
    alpha: Optional[float] = None


def overdispersion(X) -> dict:
    """Estimate the negative-binomial overdispersion ``alpha`` (Var ≈ μ + α·μ²) across genes.

    Returns ``{"alpha", "mean_depth", "k"}`` with ``k = 4·alpha·mean_depth``.
    """
    data, indices, indptr, shape = _as_csr(X)
    return _core.overdispersion(data, indices, indptr, shape)


def normalize(X, target="mean", alpha=None, log1p=True, center=True) -> ShiftedCLR:
    """PFlog / shifted-CLR normalization.

    ``target`` is ``"mean"``, ``"median"``, ``"auto"`` (estimate α), a numeric fixed ``K``, or
    pass ``alpha`` directly. The ``"auto"``/``alpha`` path is **PFlog**: the centered log-ratio of
    the counts shifted by ``1/(4·α)``, ``center(log(x + 1/(4·α)))`` (computed as the equivalent
    sparsity-preserving ``center(log1p(4·α·x))``). Depth targets keep the classic PF scale ``K/s_i``.
    """
    import scipy.sparse as sp

    data, indices, indptr, shape = _as_csr(X)
    tname, fixed = _resolve_target(target)
    odata, oindices, oindptr, row_center, k, a = _core.normalize(
        data, indices, indptr, shape, tname, fixed, alpha, log1p, center
    )
    sparse = sp.csr_matrix((odata, oindices, oindptr), shape=shape)
    return ShiftedCLR(sparse=sparse, row_center=np.asarray(row_center), k=k, alpha=a)


def pca(
    X: Union[ShiftedCLR, "object"],
    n_components: int = 50,
    ncv: Optional[int] = None,
    maxiter: Optional[int] = None,
    seed: int = 0,
    tol: float = 0.0,
) -> PCAResult:
    """Sparse PCA. Pass a :class:`ShiftedCLR` to run the implicit-centered shifted-CLR path, or a
    plain matrix for ordinary sparse PCA."""
    if isinstance(X, ShiftedCLR):
        data, indices, indptr, shape = _as_csr(X.sparse)
        rc = np.ascontiguousarray(X.row_center, dtype=np.float64)
        k, a = X.k, X.alpha
    else:
        data, indices, indptr, shape = _as_csr(X)
        rc = None
        k = a = None
    res = _core.pca(data, indices, indptr, shape, rc, n_components, ncv, maxiter, seed, tol)
    return PCAResult(k=k, alpha=a, **res)


def normalize_pca(
    X,
    n_components: int = 50,
    target="auto",
    alpha=None,
    ncv: Optional[int] = None,
    maxiter: Optional[int] = None,
    seed: int = 0,
    tol: float = 0.0,
) -> PCAResult:
    """One-shot raw counts → shifted-CLR → sparse PCA (all in Rust)."""
    data, indices, indptr, shape = _as_csr(X)
    tname, fixed = _resolve_target(target)
    res = _core.normalize_pca(
        data, indices, indptr, shape, n_components, tname, fixed, alpha, ncv, maxiter, seed, tol
    )
    return PCAResult(**res)
