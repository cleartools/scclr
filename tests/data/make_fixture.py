#!/usr/bin/env python
"""Regenerate `pbmc_subset.mtx` — the real-data test fixture.

A deterministic subset of the classic 10x PBMC 3k dataset (400 cells x 800 genes of raw counts),
written as Matrix Market (cells x genes, integer). Used by both the scclr Python tests and the
runorm Rust tests (`runorm/tests/data/pbmc_subset.mtx` is a copy).

Run from the scclr venv (needs scanpy + scipy):  python tests/data/make_fixture.py
"""

import os
import warnings

import numpy as np
import scipy.io
import scipy.sparse as sp

warnings.simplefilter("ignore")
import scanpy as sc  # noqa: E402


def main():
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # Deterministic, representative slice (first 400 cells; first 800 reasonably-detected genes).
    sub = adata[:400, :].copy()
    detected = np.asarray((sub.X > 0).sum(0)).ravel()
    keep = np.where(detected >= 5)[0][:800]
    sub = sub[:, keep].copy()

    X = sp.csr_matrix(sub.X)
    X.data = np.rint(X.data).astype(np.int64)
    X.eliminate_zeros()
    X = X[np.diff(X.indptr) > 0]                       # drop empty cells
    X = X[:, np.asarray(X.sum(0)).ravel() > 0]         # drop unobserved genes
    X = sp.csr_matrix(X)
    X.data = X.data.astype(np.int64)

    here = os.path.dirname(os.path.abspath(__file__))
    dests = [
        os.path.join(here, "pbmc_subset.mtx"),
        os.path.abspath(os.path.join(here, "..", "..", "..", "runorm", "tests", "data", "pbmc_subset.mtx")),
    ]
    for path in dests:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        scipy.io.mmwrite(path, X, field="integer", comment="PBMC3k subset (cells x genes, raw counts)")
        print("wrote", path)
    print(f"shape={X.shape} nnz={X.nnz} density={X.nnz / (X.shape[0] * X.shape[1]):.3f}")


if __name__ == "__main__":
    main()
