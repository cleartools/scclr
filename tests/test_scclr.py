"""Smoke + parity tests for scclr (functional API and the scverse drop-in)."""

import numpy as np
import scipy.sparse as sp
import pytest

import scclr


def nb_counts(n_cells=120, n_genes=30, r=2, p=0.2, seed=0):
    """Negative-binomial counts: overdispersed (alpha = 1/r), no all-zero rows."""
    rng = np.random.default_rng(seed)
    counts = rng.negative_binomial(r, p, size=(n_cells, n_genes)).astype(float)
    counts[:, 0] += 1.0  # guarantee positive row sums
    return counts


def structured_counts(n_per=40, n_genes=30, n_groups=3, boost=25, seed=0):
    """NB counts with `n_groups` clusters, each over-expressing a distinct gene block.

    Gives a low-rank signal with well-separated top eigenvalues — the regime where any
    PCA solver (iterative or full SVD) agrees, so it is suitable for cross-checking.
    """
    rng = np.random.default_rng(seed)
    counts = rng.negative_binomial(2, 0.2, size=(n_groups * n_per, n_genes)).astype(float)
    block = n_genes // (n_groups + 1)
    for g in range(n_groups):
        counts[g * n_per : (g + 1) * n_per, g * block : (g + 1) * block] += boost
    counts[:, 0] += 1.0
    return counts


def ref_pflogpf(counts, k):
    """Independent dense PFlogPF reference."""
    s = counts.sum(1, keepdims=True)
    L = np.log1p(counts * (k / s))
    return L - L.mean(1, keepdims=True)


def align_sign(a, b):
    """Flip each column of b to best match a (PCA sign ambiguity)."""
    out = b.copy()
    for j in range(a.shape[1]):
        if np.dot(a[:, j], b[:, j]) < 0:
            out[:, j] = -b[:, j]
    return out


def test_normalize_matches_numpy_reference():
    counts = nb_counts(seed=1)
    X = sp.csr_matrix(counts)
    sclr = scclr.normalize(X, target="mean")
    k = counts.sum(1).mean()
    assert abs(sclr.k - k) < 1e-9
    np.testing.assert_allclose(sclr.to_dense(), ref_pflogpf(counts, k), atol=1e-9)


def test_normalize_preserves_sparsity_pattern():
    counts = nb_counts(seed=2)
    X = sp.csr_matrix(counts)
    sclr = scclr.normalize(X, target="auto")
    # PFlogPF keeps the same nonzeros as the input (log1p(0)=0).
    assert sclr.sparse.nnz == X.nnz


def test_overdispersion_matches_scipy_curve_fit():
    curve_fit = pytest.importorskip("scipy.optimize").curve_fit
    counts = nb_counts(n_cells=400, n_genes=40, seed=3)
    X = sp.csr_matrix(counts)
    mu = counts.mean(0)
    var = counts.var(0)  # population variance (ddof=0), matching runorm
    alpha_ref = curve_fit(lambda x, a: x + a * x**2, mu, var)[0][0]
    od = scclr.overdispersion(X)
    assert od["alpha"] > 0
    assert abs(od["alpha"] - alpha_ref) < 1e-6
    assert abs(od["k"] - 4 * od["alpha"] * od["mean_depth"]) < 1e-6


def test_normalize_pca_shapes_and_parity():
    counts = nb_counts(seed=4)
    X = sp.csr_matrix(counts)
    res = scclr.normalize_pca(X, n_components=5, target="auto")
    assert res.scores.shape == (X.shape[0], 5)
    assert res.components.shape == (5, X.shape[1])
    assert res.k is not None and res.alpha is not None

    # normalize_pca == normalize then pca.
    sclr = scclr.normalize(X, target="auto")
    res2 = scclr.pca(sclr, n_components=5)
    np.testing.assert_allclose(res.explained_variance, res2.explained_variance, atol=1e-6)
    np.testing.assert_allclose(
        align_sign(res.scores, res2.scores), res.scores, atol=1e-5
    )


def test_scverse_dropin_anndata():
    ad = pytest.importorskip("anndata")
    sc = pytest.importorskip("scanpy")
    PCA = pytest.importorskip("sklearn.decomposition").PCA

    counts = structured_counts(n_per=40, n_genes=30, n_groups=3, seed=5)
    X = sp.csr_matrix(counts)
    adata = ad.AnnData(X.copy())

    # Drop-in: replaces sc.pp.normalize_total + sc.pp.log1p + sc.tl.pca.
    scclr.pp.pflogpf(adata, target="mean")
    assert "pflogpf" in adata.layers
    assert "pflogpf_center" in adata.obs
    assert adata.uns["pflogpf"]["k"] is not None

    # The stored layer + center reconstruct PFlogPF exactly (normalization correctness
    # through the AnnData path).
    Y = adata.layers["pflogpf"].toarray() - adata.obs["pflogpf_center"].to_numpy()[:, None]
    np.testing.assert_allclose(Y, ref_pflogpf(counts, adata.uns["pflogpf"]["k"]), atol=1e-9)

    scclr.tl.pca(adata, n_comps=5)
    assert adata.obsm["X_pca"].shape == (adata.n_obs, 5)
    assert adata.varm["PCs"].shape == (adata.n_vars, 5)
    assert "variance" in adata.uns["pca"] and "variance_ratio" in adata.uns["pca"]
    # Explained variance is non-increasing and finite.
    var = adata.uns["pca"]["variance"]
    assert np.all(np.isfinite(var)) and np.all(np.diff(var) <= 1e-9)

    # Downstream scanpy interop must just work on the slots we wrote.
    sc.pp.neighbors(adata, n_pcs=5, use_rep="X_pca")
    assert "neighbors" in adata.uns

    # Numerical correctness: the sparse shifted-CLR PCA matches a full-SVD reference on the dense
    # shifted-CLR matrix. Explained variances match to high precision; the well-separated top
    # components also match in score space (the cluster-separating axes).
    sk = PCA(n_components=5, svd_solver="full").fit(Y)
    np.testing.assert_allclose(adata.uns["pca"]["variance"], sk.explained_variance_, rtol=1e-6)
    sk_scores = sk.transform(Y)
    for j in range(2):
        corr = abs(np.corrcoef(adata.obsm["X_pca"][:, j], sk_scores[:, j])[0, 1])
        assert corr > 0.999, f"PC{j} correlation with full-SVD = {corr}"

    # Determinism: re-running the drop-in yields a bit-identical embedding.
    a2 = ad.AnnData(X.copy())
    scclr.pp.pflogpf(a2, target="mean")
    scclr.tl.pca(a2, n_comps=5)
    np.testing.assert_allclose(adata.obsm["X_pca"], a2.obsm["X_pca"], atol=1e-10)


def test_scverse_legacy_name_keeps_log1ppf_layer():
    ad = pytest.importorskip("anndata")

    counts = nb_counts(n_cells=20, n_genes=10, seed=8)
    adata = ad.AnnData(sp.csr_matrix(counts))
    scclr.pp.pflog1ppf(adata, target="mean")

    assert "log1ppf" in adata.layers
    assert "log1ppf_center" in adata.obs
    assert "pflogpf" not in adata.layers

    # The PCA default still auto-detects the legacy layer.
    scclr.tl.pca(adata, n_comps=3)
    assert adata.obsm["X_pca"].shape == (adata.n_obs, 3)


def test_scverse_dropin_mudata():
    ad = pytest.importorskip("anndata")
    md = pytest.importorskip("mudata")

    rna = ad.AnnData(sp.csr_matrix(nb_counts(n_cells=80, n_genes=25, seed=6)))
    adt = ad.AnnData(sp.csr_matrix(nb_counts(n_cells=80, n_genes=12, seed=7)))
    mdata = md.MuData({"rna": rna, "adt": adt})

    scclr.pp.pflogpf(mdata, target="mean")
    scclr.tl.pca(mdata, n_comps=3)
    for mod in mdata.mod.values():
        assert "pflogpf" in mod.layers
        assert mod.obsm["X_pca"].shape == (mod.n_obs, 3)
