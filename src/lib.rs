//! `scclr._core` — the Rust extension module behind the `scclr` Python package.
//!
//! Stays purely array-based: it accepts the three CSR arrays (`data`, `indices`, `indptr`) plus a
//! shape and returns numpy arrays. All AnnData/MuData ergonomics live in the Python layer. The
//! heavy compute runs with the GIL released.

use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use runorm::{
    estimate_overdispersion, normalize_csr, CsrCounts, NormParams, PfTarget, ShiftedClrMatrix,
};
use rupca::{
    pca_scanpy_sparse_csr, pca_shifted_clr_sparse_csr, CsrMatrix, ScanpyPcaParams,
    ScanpyPcaResult, ShiftedClrCsrMatrix,
};

fn norm_err(e: runorm::NormError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn pca_err(e: rupca::RuPcaError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

const SOLVER_PANIC_MSG: &str =
    "sparse PCA solver failed (the eigensolver did not converge — often a near-degenerate \
     spectrum). Try fewer components (n_components), a larger ncv, or denser/structured input.";

/// Run a rupca PCA call, converting both error returns and solver panics into clean Python errors.
fn run_pca<F>(py: Python<'_>, f: F) -> PyResult<ScanpyPcaResult>
where
    F: FnOnce() -> Result<ScanpyPcaResult, rupca::RuPcaError> + Send,
{
    let outcome = py.allow_threads(|| std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)));
    match outcome {
        Ok(Ok(r)) => Ok(r),
        Ok(Err(e)) => Err(pca_err(e)),
        Err(_) => Err(PyValueError::new_err(SOLVER_PANIC_MSG)),
    }
}

/// Build runorm `CsrCounts` from borrowed numpy arrays (copies indices `i64 -> usize`).
fn build_counts(
    data: &PyReadonlyArray1<f64>,
    indices: &PyReadonlyArray1<i64>,
    indptr: &PyReadonlyArray1<i64>,
    shape: (usize, usize),
) -> PyResult<CsrCounts> {
    let data = data.as_slice()?.to_vec();
    let indices = indices.as_slice()?.iter().map(|&x| x as usize).collect();
    let indptr = indptr.as_slice()?.iter().map(|&x| x as usize).collect();
    CsrCounts::new(shape.0, shape.1, data, indices, indptr).map_err(norm_err)
}

fn parse_target(target: &str, fixed: Option<f64>, alpha: Option<f64>) -> PyResult<PfTarget> {
    if let Some(a) = alpha {
        return Ok(PfTarget::Alpha(a));
    }
    match target {
        "mean" => Ok(PfTarget::MeanDepth),
        "median" => Ok(PfTarget::MedianDepth),
        "auto" => Ok(PfTarget::EstimateAlpha),
        "fixed" => fixed
            .map(PfTarget::Fixed)
            .ok_or_else(|| PyValueError::new_err("target='fixed' requires a numeric value")),
        other => Err(PyValueError::new_err(format!(
            "unknown target '{other}' (expected mean|median|auto|fixed)"
        ))),
    }
}

/// Estimate overdispersion alpha and the derived K = 4·alpha·s.
#[pyfunction]
fn overdispersion<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, i64>,
    indptr: PyReadonlyArray1<'py, i64>,
    shape: (usize, usize),
) -> PyResult<Bound<'py, PyDict>> {
    let counts = build_counts(&data, &indices, &indptr, shape)?;
    let od = py.allow_threads(|| estimate_overdispersion(&counts)).map_err(norm_err)?;
    let d = PyDict::new(py);
    d.set_item("alpha", od.alpha)?;
    d.set_item("mean_depth", od.mean_depth)?;
    d.set_item("k", od.k)?;
    Ok(d)
}

type NormalizeOut<'py> = (
    Bound<'py, PyArray1<f64>>, // data
    Bound<'py, PyArray1<i64>>, // indices
    Bound<'py, PyArray1<i64>>, // indptr
    Bound<'py, PyArray1<f64>>, // row_center
    f64,                       // k
    Option<f64>,               // alpha (if derived)
);

/// PFlogPF / shifted-CLR normalization. Returns the sparse PFlogPF arrays plus `row_center`.
#[pyfunction]
#[pyo3(signature = (data, indices, indptr, shape, target="mean", fixed=None, alpha=None, log1p=true, center=true))]
#[allow(clippy::too_many_arguments)]
fn normalize<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, i64>,
    indptr: PyReadonlyArray1<'py, i64>,
    shape: (usize, usize),
    target: &str,
    fixed: Option<f64>,
    alpha: Option<f64>,
    log1p: bool,
    center: bool,
) -> PyResult<NormalizeOut<'py>> {
    let counts = build_counts(&data, &indices, &indptr, shape)?;
    let params = NormParams { target: parse_target(target, fixed, alpha)?, log1p, center };
    let (m, report) = py.allow_threads(|| normalize_csr(&counts, &params)).map_err(norm_err)?;

    let out_indices: Vec<i64> = m.indices.iter().map(|&x| x as i64).collect();
    let out_indptr: Vec<i64> = m.indptr.iter().map(|&x| x as i64).collect();
    Ok((
        m.data.into_pyarray(py),
        out_indices.into_pyarray(py),
        out_indptr.into_pyarray(py),
        m.row_center.into_pyarray(py),
        report.k,
        report.alpha,
    ))
}

/// Assemble the PCA result dict (matches `ScanpyPcaResult`).
fn pca_result_dict<'py>(py: Python<'py>, r: ScanpyPcaResult) -> PyResult<Bound<'py, PyDict>> {
    let (n, k, f) = (r.n_samples, r.n_components, r.n_features);
    let d = PyDict::new(py);
    d.set_item("scores", r.scores.into_pyarray(py).reshape((n, k))?)?;
    d.set_item("components", r.components.into_pyarray(py).reshape((k, f))?)?;
    d.set_item("mean", r.mean.into_pyarray(py))?;
    d.set_item("explained_variance", r.explained_variance.into_pyarray(py))?;
    d.set_item("explained_variance_ratio", r.explained_variance_ratio.into_pyarray(py))?;
    d.set_item("singular_values", r.singular_values.into_pyarray(py))?;
    d.set_item("noise_variance", r.noise_variance)?;
    d.set_item("n_samples", n)?;
    d.set_item("n_features", f)?;
    d.set_item("n_components", k)?;
    Ok(d)
}

fn pca_params(
    n_components: usize,
    ncv: Option<usize>,
    maxiter: Option<usize>,
    seed: u64,
    tol: f64,
) -> ScanpyPcaParams {
    ScanpyPcaParams { n_components, tol, ncv, maxiter, seed }
}

/// Sparse PCA. If `row_center` is given, runs the shifted-CLR path (implicit `data - row_center`);
/// otherwise the plain sparse path.
#[pyfunction]
#[pyo3(signature = (data, indices, indptr, shape, row_center=None, n_components=50, ncv=None, maxiter=None, seed=0, tol=0.0))]
#[allow(clippy::too_many_arguments)]
fn pca<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, i64>,
    indptr: PyReadonlyArray1<'py, i64>,
    shape: (usize, usize),
    row_center: Option<PyReadonlyArray1<'py, f64>>,
    n_components: usize,
    ncv: Option<usize>,
    maxiter: Option<usize>,
    seed: u64,
    tol: f64,
) -> PyResult<Bound<'py, PyDict>> {
    // Extract owned data before releasing the GIL.
    let sparse = CsrMatrix {
        n_rows: shape.0,
        n_cols: shape.1,
        data: data.as_slice()?.to_vec(),
        indices: indices.as_slice()?.iter().map(|&x| x as usize).collect(),
        indptr: indptr.as_slice()?.iter().map(|&x| x as usize).collect(),
    };
    let center = match &row_center {
        Some(rc) => Some(rc.as_slice()?.to_vec()),
        None => None,
    };
    let params = pca_params(n_components, ncv, maxiter, seed, tol);

    let result = run_pca(py, move || match center {
        Some(rc) => {
            let m = ShiftedClrCsrMatrix { sparse, row_center: rc };
            pca_shifted_clr_sparse_csr(&m, params)
        }
        None => pca_scanpy_sparse_csr(&sparse, params),
    })?;
    pca_result_dict(py, result)
}

/// One-shot: normalize raw counts to shifted-CLR, then run sparse PCA on it. Returns the PCA dict
/// augmented with the chosen `k`/`alpha`.
#[pyfunction]
#[pyo3(signature = (data, indices, indptr, shape, n_components=50, target="auto", fixed=None, alpha=None, ncv=None, maxiter=None, seed=0, tol=0.0))]
#[allow(clippy::too_many_arguments)]
fn normalize_pca<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, i64>,
    indptr: PyReadonlyArray1<'py, i64>,
    shape: (usize, usize),
    n_components: usize,
    target: &str,
    fixed: Option<f64>,
    alpha: Option<f64>,
    ncv: Option<usize>,
    maxiter: Option<usize>,
    seed: u64,
    tol: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let counts = build_counts(&data, &indices, &indptr, shape)?;
    let params = NormParams { target: parse_target(target, fixed, alpha)?, log1p: true, center: true };
    let pparams = pca_params(n_components, ncv, maxiter, seed, tol);

    // Keep `PyErr` out of the GIL-released closures (it is not `Send`); convert after each step.
    let (m, report) = py.allow_threads(|| normalize_csr(&counts, &params)).map_err(norm_err)?;
    let ShiftedClrMatrix { n_rows, n_cols, data, indices, indptr, row_center } = m;
    // Move runorm's buffers straight into rupca's type (no element copies).
    let shifted = ShiftedClrCsrMatrix {
        sparse: CsrMatrix { n_rows, n_cols, data, indices, indptr },
        row_center,
    };
    let result = run_pca(py, move || pca_shifted_clr_sparse_csr(&shifted, pparams))?;

    let d = pca_result_dict(py, result)?;
    d.set_item("k", report.k)?;
    d.set_item("alpha", report.alpha)?;
    Ok(d)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(overdispersion, m)?)?;
    m.add_function(wrap_pyfunction!(normalize, m)?)?;
    m.add_function(wrap_pyfunction!(pca, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_pca, m)?)?;
    Ok(())
}
