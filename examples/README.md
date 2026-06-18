# Examples

## Angelidis lung pseudobulk

`angelidis_pseudobulk_scclr.ipynb` mirrors the `scclrR` Seurat vignette in
Python. It runs sparse PFlog normalization and PCA with `scclr`, then compares
PC1 loadings to an old-vs-young differential expression estimate from
`edgepython`.

The notebook is committed with rendered outputs. To rerun it locally:

```bash
uv pip install -e ".[examples]" maturin
uv run maturin develop --release
uv run jupyter notebook examples/angelidis_pseudobulk_scclr.ipynb
```

The data files in `examples/data/` are copied from the `scclrR` vignette data.
