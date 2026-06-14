# scclr

Single-cell **shifted-CLR / PFlogPF** normalization and **sparse PCA** for Python — a
scverse drop-in built on the Rust crates [`runorm`](../runorm) (normalization) and
[`rupca`](../rupca) (sparse PCA), with [`ruanndata`](../ruanndata) for I/O.

The normalization keeps the matrix sparse (the PFlogPF values plus a per-cell mean vector); PCA
runs on the implicit `sparse − row_center` matrix without densifying.

## Install (dev)

```bash
uv venv
uv pip install -e ".[test]" maturin
uv run maturin develop --release
```

## Use

Functional (bare scipy/numpy):

```python
import scclr
sclr = scclr.normalize(X, target="auto")   # estimate alpha -> K = 4*alpha*s
res  = scclr.pca(sclr, n_components=50)     # res.scores, res.components, ...
```

scverse in-place (AnnData / MuData), shaped like scanpy:

```python
import scclr
scclr.pp.pflogpf(adata, target="auto")      # -> adata.layers["pflogpf"] + obs center
scclr.tl.pca(adata, n_comps=50)             # -> adata.obsm["X_pca"], varm["PCs"], uns["pca"]
# downstream sc.pp.neighbors(adata) / sc.tl.umap(adata) work unchanged
```

This swaps in for `sc.pp.normalize_total` + `sc.pp.log1p` + `sc.tl.pca`.
