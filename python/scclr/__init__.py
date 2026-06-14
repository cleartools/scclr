"""scclr — single-cell shifted-CLR (PFlogPF) normalization + sparse PCA.

A scverse drop-in. Two ways to use it:

* **Functional / array API** (works with bare scipy/numpy)::

      sclr = scclr.normalize(X, target="auto")     # -> ShiftedCLR
      res  = scclr.pca(sclr, n_components=50)       # -> PCAResult

* **In-place scverse API** (AnnData / MuData), shaped like scanpy::

      scclr.pp.pflogpf(adata, target="auto")        # -> adata.layers["pflogpf"] + obs center
      scclr.tl.pca(adata, n_comps=50)               # -> adata.obsm["X_pca"], varm["PCs"], uns["pca"]

  which swaps in for ``sc.pp.normalize_total + sc.pp.log1p + sc.tl.pca``; downstream
  ``sc.pp.neighbors`` / ``sc.tl.umap`` are untouched.
"""

from ._func import (
    PCAResult,
    ShiftedCLR,
    normalize,
    normalize_pca,
    overdispersion,
    pca,
)
from . import pp, tl

__all__ = [
    "ShiftedCLR",
    "PCAResult",
    "overdispersion",
    "normalize",
    "pca",
    "normalize_pca",
    "pp",
    "tl",
]

__version__ = "0.1.0"
