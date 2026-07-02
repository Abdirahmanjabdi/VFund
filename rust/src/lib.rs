//! Native core for VFund's cross-sectional simulation loop.
//!
//! This mirrors `vfund/backtest/sim.py::simulate_py` exactly — same inputs,
//! same arithmetic, same outputs — but runs the T-length loop in Rust with no
//! per-bar Python overhead. It's the hot path re-run thousands of times in
//! bootstraps and walk-forwards.
//!
//! Build with `maturin develop --release` from the `rust/` directory; the
//! Python side (`vfund/backtest/_accel.py`) picks it up automatically.

use numpy::ndarray::ArrayView2;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (rets, reb_indices, reb_weights, initial, cost_rate, short_cost_per_bar, funding=None, tradable=None))]
#[allow(clippy::too_many_arguments)]
fn simulate<'py>(
    py: Python<'py>,
    rets: PyReadonlyArray2<'py, f64>,
    reb_indices: PyReadonlyArray1<'py, i64>,
    reb_weights: PyReadonlyArray2<'py, f64>,
    initial: f64,
    cost_rate: f64,
    short_cost_per_bar: f64,
    funding: Option<PyReadonlyArray2<'py, f64>>,
    tradable: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let rets = rets.as_array();
    let (t_len, n) = rets.dim();
    let reb_idx = reb_indices.as_array();
    let reb_w = reb_weights.as_array();
    let funding: Option<ArrayView2<f64>> = funding.as_ref().map(|f| f.as_array());
    let tradable: Option<ArrayView2<f64>> = tradable.as_ref().map(|f| f.as_array());

    let mut equity = vec![0.0_f64; t_len];
    let mut gross = vec![0.0_f64; t_len];
    let n_reb = reb_idx.len();
    let mut turnovers = vec![0.0_f64; n_reb];

    let mut w_active = vec![0.0_f64; n];
    let mut w_drifted = vec![0.0_f64; n];
    let mut eq = initial;
    equity[0] = initial;
    let mut ptr = 0usize;

    for t in 1..t_len {
        let r = rets.row(t);

        // Price return and short notional in one pass.
        let mut price_ret = 0.0;
        let mut short_notional = 0.0;
        for i in 0..n {
            price_ret += w_active[i] * r[i];
            if w_active[i] < 0.0 {
                short_notional += -w_active[i];
            }
        }

        let fund_ret = match &funding {
            Some(f) => {
                let fr = f.row(t);
                let mut s = 0.0;
                for i in 0..n {
                    s += w_active[i] * fr[i];
                }
                -s
            }
            None => 0.0,
        };
        let short_ret = -short_notional * short_cost_per_bar;
        eq *= 1.0 + price_ret + fund_ret + short_ret;

        let growth = 1.0 + price_ret;
        if growth != 0.0 {
            for i in 0..n {
                w_drifted[i] = w_active[i] * (1.0 + r[i]) / growth;
            }
        } else {
            w_drifted.copy_from_slice(&w_active);
        }
        if let Some(tr) = &tradable {
            let trow = tr.row(t);
            for i in 0..n {
                if trow[i] <= 0.0 {
                    w_drifted[i] = 0.0;
                }
            }
        }

        if ptr < n_reb && reb_idx[ptr] as usize == t {
            let w_target = reb_w.row(ptr);
            let mut turnover = 0.0;
            for i in 0..n {
                turnover += (w_target[i] - w_drifted[i]).abs();
            }
            eq *= 1.0 - turnover * cost_rate;
            turnovers[ptr] = turnover;
            for i in 0..n {
                w_active[i] = w_target[i];
            }
            ptr += 1;
        } else {
            w_active.copy_from_slice(&w_drifted);
        }
        equity[t] = eq;
        let mut g = 0.0;
        for i in 0..n {
            g += w_active[i].abs();
        }
        gross[t] = g;
    }

    Ok((
        equity.into_pyarray_bound(py),
        turnovers.into_pyarray_bound(py),
        gross.into_pyarray_bound(py),
    ))
}

/// Market-making matching loop (mirrors `microstructure/simulator.py::mm_loop_py`).
///
/// Given a pre-generated value path and order stream, run the sequential fill
/// loop and return (realised P&L, number of fills). One fill per side per step.
#[pyfunction]
fn mm_loop(
    value: PyReadonlyArray1<i64>,
    n_orders: PyReadonlyArray1<i64>,
    informed: PyReadonlyArray1<i64>,
    noise: PyReadonlyArray1<i64>,
    half_spread: i64,
) -> PyResult<(f64, i64)> {
    let value = value.as_array();
    let n_orders = n_orders.as_array();
    let informed = informed.as_array();
    let noise = noise.as_array();
    let h = half_spread;

    let mut total_pnl = 0.0_f64;
    let mut n_fills = 0_i64;
    let mut ptr = 0usize;

    for t in 0..n_orders.len() {
        let v = value[t];
        let future = value[t + 1] as f64;
        let up = value[t + 1] > value[t];
        let mut ask_avail = true;
        let mut bid_avail = true;
        for _ in 0..n_orders[t] {
            let side = if informed[ptr] != 0 {
                if up { 1 } else { -1 }
            } else {
                noise[ptr]
            };
            ptr += 1;
            if side > 0 && ask_avail {
                total_pnl += (v + h) as f64 - future;
                n_fills += 1;
                ask_avail = false;
            } else if side < 0 && bid_avail {
                total_pnl += future - (v - h) as f64;
                n_fills += 1;
                bid_avail = false;
            }
        }
    }
    Ok((total_pnl, n_fills))
}

#[pymodule]
fn vfund_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate, m)?)?;
    m.add_function(wrap_pyfunction!(mm_loop, m)?)?;
    Ok(())
}
