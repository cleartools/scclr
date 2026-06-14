"""Preprocessing (``scclr.pp``): scanpy-shaped, in-place normalization on AnnData / MuData."""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import _func

__all__ = ["pflogpf", "pflog1ppf", "overdispersion"]


def _is_mudata(obj) -> bool:
    return obj.__class__.__name__ == "MuData" and hasattr(obj, "mod")


def _get_matrix(adata, layer: Optional[str]):
    return adata.layers[layer] if layer is not None else adata.X


def _pflogpf_impl(
    adata,
    *,
    target="auto",
    alpha: Optional[float] = None,
    layer: Optional[str] = None,
    key_added: str,
    center: bool = True,
    log1p: bool = True,
    densify: bool = False,
):
    if _is_mudata(adata):
        for mod in adata.mod.values():
            _pflogpf_impl(
                mod,
                target=target,
                alpha=alpha,
                layer=layer,
                key_added=key_added,
                center=center,
                log1p=log1p,
                densify=densify,
            )
        return None

    X = _get_matrix(adata, layer)
    sclr = _func.normalize(X, target=target, alpha=alpha, log1p=log1p, center=center)

    adata.layers[key_added] = sclr.sparse
    if center:
        adata.obs[f"{key_added}_center"] = np.asarray(sclr.row_center)
    adata.uns[key_added] = {
        "k": float(sclr.k),
        "alpha": (float(sclr.alpha) if sclr.alpha is not None else None),
        "target": target,
        "center": bool(center),
        "log1p": bool(log1p),
    }
    if densify:
        adata.X = sclr.to_dense()
    return None


def pflogpf(
    adata,
    *,
    target="auto",
    alpha: Optional[float] = None,
    layer: Optional[str] = None,
    key_added: str = "pflogpf",
    center: bool = True,
    log1p: bool = True,
    densify: bool = False,
):
    """Compute PFlogPF / shifted-CLR in place.

    Writes the sparse PFlogPF matrix to ``adata.layers[key_added]`` and (when ``center``) the
    per-cell mean to ``adata.obs[f"{key_added}_center"]``; records the chosen ``k``/``alpha`` in
    ``adata.uns[key_added]``. With ``densify=True`` also writes the dense shifted-CLR into
    ``adata.X`` (scanpy-style, for generic downstream tools).

    For a ``MuData`` object, applies independently to each modality.
    """
    return _pflogpf_impl(
        adata,
        target=target,
        alpha=alpha,
        layer=layer,
        key_added=key_added,
        center=center,
        log1p=log1p,
        densify=densify,
    )


def pflog1ppf(
    adata,
    *,
    target="auto",
    alpha: Optional[float] = None,
    layer: Optional[str] = None,
    key_added: str = "log1ppf",
    center: bool = True,
    log1p: bool = True,
    densify: bool = False,
):
    """Backward-compatible alias for :func:`pflogpf`.

    The legacy name stores results in ``adata.layers["log1ppf"]`` by default. New code should
    prefer :func:`pflogpf`, which stores results under ``"pflogpf"``.
    """
    return _pflogpf_impl(
        adata,
        target=target,
        alpha=alpha,
        layer=layer,
        key_added=key_added,
        center=center,
        log1p=log1p,
        densify=densify,
    )


def overdispersion(adata, *, layer: Optional[str] = None):
    """Estimate overdispersion and store it in ``adata.uns["overdispersion"]``.

    Returns the ``{"alpha", "mean_depth", "k"}`` dict (or, for ``MuData``, a dict keyed by
    modality name).
    """
    if _is_mudata(adata):
        return {name: overdispersion(mod, layer=layer) for name, mod in adata.mod.items()}

    od = _func.overdispersion(_get_matrix(adata, layer))
    adata.uns["overdispersion"] = od
    return od
