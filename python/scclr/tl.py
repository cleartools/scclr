"""Tools (``scclr.tl``): scanpy-shaped, in-place sparse PCA on AnnData / MuData."""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import _func

__all__ = ["pca"]


def _is_mudata(obj) -> bool:
    return obj.__class__.__name__ == "MuData" and hasattr(obj, "mod")


def pca(
    adata,
    *,
    n_comps: int = 50,
    layer: Optional[str] = "pflogpf",
    center_key: Optional[str] = None,
    ncv: Optional[int] = None,
    maxiter: Optional[int] = None,
    seed: int = 0,
    tol: float = 0.0,
):
    """Sparse shifted-CLR PCA, written into the slots scanpy uses.

    Reads the sparse PFlogPF matrix from ``adata.layers[layer]`` and the per-cell center from
    ``adata.obs[center_key]`` (default ``f"{layer}_center"``) so PCA runs on the implicit
    ``layer - center`` without densifying. Writes:

    * ``adata.obsm["X_pca"]``  — scores (n_obs × n_comps)
    * ``adata.varm["PCs"]``    — loadings (n_vars × n_comps)
    * ``adata.uns["pca"]``     — ``{"variance", "variance_ratio"}``

    Falls back to plain sparse PCA on ``layer`` (or ``X``) when no center vector is present.
    For a ``MuData`` object, applies independently to each modality.
    """
    if _is_mudata(adata):
        for mod in adata.mod.values():
            pca(
                mod,
                n_comps=n_comps,
                layer=layer,
                center_key=center_key,
                ncv=ncv,
                maxiter=maxiter,
                seed=seed,
                tol=tol,
            )
        return None

    if layer == "pflogpf" and layer not in adata.layers and "log1ppf" in adata.layers:
        layer = "log1ppf"

    ck = center_key if center_key is not None else (f"{layer}_center" if layer else None)

    if layer is not None and layer in adata.layers:
        mat = adata.layers[layer]
        if ck is not None and ck in adata.obs:
            row_center = np.asarray(adata.obs[ck].to_numpy(), dtype=np.float64)
            target = _func.ShiftedCLR(sparse=mat, row_center=row_center)
        else:
            target = mat
    else:
        target = adata.X

    res = _func.pca(target, n_components=n_comps, ncv=ncv, maxiter=maxiter, seed=seed, tol=tol)

    adata.obsm["X_pca"] = np.ascontiguousarray(res.scores)
    adata.varm["PCs"] = np.ascontiguousarray(res.components.T)
    adata.uns["pca"] = {
        "variance": res.explained_variance,
        "variance_ratio": res.explained_variance_ratio,
    }
    return None
