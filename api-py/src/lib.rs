use numpy::PyArrayMethods;
use pyo3::prelude::*;

/// Embed a string or a list of strings. Returns a float32 array of shape
/// `(len(texts), DIMENSIONS)`; a single string gives one row.
#[pyfunction]
fn encode<'py>(py: Python<'py>, texts: Py<PyAny>) -> Bound<'py, numpy::PyArray2<f32>> {
    let texts: Vec<String> = if let Ok(s) = texts.extract::<String>(py) {
        vec![s]
    } else {
        texts
            .extract::<Vec<String>>(py)
            .expect("expected str or list of str")
    };
    let vecs = encoders::encode(texts);
    let rows = vecs.len();
    let cols = encoders::DIMENSIONS;
    let flat: Vec<f32> = vecs.into_iter().flatten().collect();
    numpy::PyArray1::from_vec(py, flat)
        .reshape([rows, cols])
        .unwrap()
}

/// Width of every embedding `encode` returns.
#[pyfunction]
fn dimensions() -> usize {
    encoders::DIMENSIONS
}

#[pymodule]
fn ese(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode, m)?)?;
    m.add_function(wrap_pyfunction!(dimensions, m)?)?;
    m.add("DIMENSIONS", encoders::DIMENSIONS)?;
    Ok(())
}
