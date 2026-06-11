"""Real-data tests on a PBMC 3k subset (400 cells x 800 genes of genuine 10x counts).

Exercises the full scverse drop-in on real single-cell data and validates it against
numpy / sklearn references. The fixture is committed at ``tests/data/pbmc_subset.mtx``.
"""

import os

import numpy as np
import scipy.io
import scipy.sparse as sp
import pytest

import scclr

DATA = os.path.join(os.path.dirname(__file__), "data", "pbmc_subset.mtx")


def load_counts():
    return sp.csr_matrix(scipy.io.mmread(DATA)).astype(np.float64)


def ref_pflog1ppf(counts_dense, k):
    s = counts_dense.sum(1, keepdims=True)
    L = np.log1p(counts_dense * (k / s))
    return L - L.mean(1, keepdims=True)


def test_normalize_matches_reference_on_pbmc():
    X = load_counts()
    assert X.shape == (400, 800)
    sclr = scclr.normalize(X, target="mean")
    k = np.asarray(X.sum(1)).ravel().mean()
    assert abs(sclr.k - k) < 1e-9
    np.testing.assert_allclose(sclr.to_dense(), ref_pflog1ppf(np.asarray(X.todense()), k), atol=1e-9)
    assert sclr.sparse.nnz == X.nnz  # sparsity preserved


def test_overdispersion_matches_curve_fit_on_pbmc():
    curve_fit = pytest.importorskip("scipy.optimize").curve_fit
    X = load_counts()
    Xd = np.asarray(X.todense())
    mu, var = Xd.mean(0), Xd.var(0)  # population variance (ddof=0), matching runorm
    alpha_ref = curve_fit(lambda x, a: x + a * x**2, mu, var)[0][0]
    od = scclr.overdispersion(X)
    assert od["alpha"] > 0, "real single-cell counts should be overdispersed"
    assert abs(od["alpha"] - alpha_ref) < 1e-6
    assert abs(od["k"] - 4 * od["alpha"] * od["mean_depth"]) < 1e-6


def test_scverse_pipeline_on_pbmc_matches_sklearn():
    ad = pytest.importorskip("anndata")
    sc = pytest.importorskip("scanpy")
    PCA = pytest.importorskip("sklearn.decomposition").PCA

    X = load_counts()
    adata = ad.AnnData(X.copy())

    # Drop-in for sc.pp.normalize_total + sc.pp.log1p + sc.tl.pca, with overdispersion-derived K.
    scclr.pp.pflog1ppf(adata, target="auto")
    assert adata.uns["log1ppf"]["alpha"] is not None and adata.uns["log1ppf"]["alpha"] > 0
    scclr.tl.pca(adata, n_comps=10)
    assert adata.obsm["X_pca"].shape == (400, 10)
    assert adata.varm["PCs"].shape == (800, 10)

    # Sparse shifted-CLR PCA matches a full-SVD reference on the dense shifted-CLR matrix.
    Y = adata.layers["log1ppf"].toarray() - adata.obs["log1ppf_center"].to_numpy()[:, None]
    sk = PCA(n_components=10, svd_solver="full").fit(Y)
    np.testing.assert_allclose(adata.uns["pca"]["variance"], sk.explained_variance_, rtol=1e-4)
    # The top (cell-type-separating) components also agree in score space.
    sk_scores = sk.transform(Y)
    for j in range(3):
        corr = abs(np.corrcoef(adata.obsm["X_pca"][:, j], sk_scores[:, j])[0, 1])
        assert corr > 0.99, f"PC{j} correlation with full-SVD = {corr}"

    # Downstream scanpy steps work on the slots we wrote.
    sc.pp.neighbors(adata, n_pcs=10, use_rep="X_pca")
    assert "neighbors" in adata.uns
