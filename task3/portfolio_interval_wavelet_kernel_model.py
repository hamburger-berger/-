# -*- coding: utf-8 -*-
"""
Wavelet-denoised interval portfolio model with kernel-similarity penalty.

This script implements the model in "最终模型_小波去噪与核相似度(1).pdf"
without changing the model structure:

    max_w  sum_i [(1-alpha) rL_i + alpha rU_i] w_i - lambda * w' A w
    s.t.   sum_i [(1-beta) qa_i + beta qL_i] w_i <= (1-beta) bc + beta bU
           sum_i w_i = 1, w_i >= 0

The four-stock universe is selected from the supplied 50-stock data set:
招商银行(sh.600036), 长江电力(sh.600900), 美的集团(sz.000333), 迈瑞医疗(sz.300760).

All outputs are saved under the same directory as this script.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import linprog, minimize

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Paths and model parameters
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "all_50_stocks_2019_2025_baostock.csv"
OUTPUT_DIR = BASE_DIR / "组合投资模型_输出"

STOCKS: Dict[str, str] = {
    "sh.600036": "招商银行",
    "sh.600900": "长江电力",
    "sz.000333": "美的集团",
    "sz.300760": "迈瑞医疗",
}

# Rolling estimation window H and final-model defaults.
H = 252
BOOTSTRAP_B = 5000
TAU = 0.05
RISK_TOLERANCE_RHO = 1.0
WAVELET = "db4"
WAVELET_LEVEL = 2
RANDOM_SEED = 20260608

# Sensitivity grids.
ALPHA_GRID = np.round(np.linspace(0.0, 1.0, 11), 2)
BETA_GRID = np.round(np.linspace(0.0, 1.0, 11), 2)
LAMBDA_GRID = np.array([0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2], dtype=float)
RHO_GRID = np.round(np.linspace(0.70, 1.30, 13), 2)

# Main case used in single-scenario figures.
BASE_ALPHA = 0.50
BASE_BETA = 0.50
BASE_LAMBDA = 1e-2


# Nature-style palette inspired by NPG/ggsci colors.
NATURE_COLORS = {
    "blue": "#3C5488",
    "cyan": "#4DBBD5",
    "green": "#00A087",
    "red": "#E64B35",
    "orange": "#F39B7F",
    "lavender": "#8491B4",
    "mint": "#91D1C2",
    "brown": "#7E6148",
    "gray": "#4A4A4A",
}

STOCK_COLORS = {
    "招商银行": NATURE_COLORS["blue"],
    "长江电力": NATURE_COLORS["cyan"],
    "美的集团": NATURE_COLORS["green"],
    "迈瑞医疗": NATURE_COLORS["red"],
}


@dataclass
class ModelInputs:
    dates: pd.DatetimeIndex
    returns_window: pd.DataFrame
    denoised_window: pd.DataFrame
    residuals_window: pd.DataFrame
    return_interval: pd.DataFrame
    loss_interval: pd.DataFrame
    risk_budget: pd.Series
    kernel_matrix: pd.DataFrame
    sigma: float
    h_block: int


def configure_matplotlib() -> None:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 320,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "font.size": 10,
        }
    )


def save_figure(fig: matplotlib.figure.Figure, name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.16, facecolor="white")
    plt.close(fig)
    return path


def pct_axis(ax: matplotlib.axes.Axes, axis: str = "y", decimals: int = 1) -> None:
    fmt = matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=decimals)
    if axis == "y":
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)


def read_four_stock_data(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"date", "code", "close", "tradestatus"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = df[df["code"].isin(STOCKS.keys())].copy()
    if df.empty:
        raise ValueError("No rows found for the four selected stock codes.")

    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df[df["tradestatus"].eq(1)].dropna(subset=["date", "code", "close"])
    prices = (
        df.pivot_table(index="date", columns="code", values="close", aggfunc="last")
        .sort_index()
        .rename(columns=STOCKS)
    )
    prices = prices[list(STOCKS.values())].dropna(how="any")
    returns = prices.pct_change().dropna(how="any")

    if len(returns) < H:
        raise ValueError(f"Need at least H={H} aligned return rows, found {len(returns)}.")
    return prices, returns


# ---------------------------------------------------------------------------
# Wavelet denoising
# ---------------------------------------------------------------------------

DB4_DEC_LO = np.array(
    [
        -0.010597401785069032,
        0.032883011666982945,
        0.030841381835986965,
        -0.18703481171888114,
        -0.02798376941698385,
        0.6308807679295904,
        0.7148465705529154,
        0.2303778133088964,
    ],
    dtype=float,
)
DB4_DEC_HI = np.array(
    [
        -0.2303778133088964,
        0.7148465705529154,
        -0.6308807679295904,
        -0.02798376941698385,
        0.18703481171888114,
        0.030841381835986965,
        -0.032883011666982945,
        -0.010597401785069032,
    ],
    dtype=float,
)


def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def dwt_periodic_even(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = len(x)
    if n % 2:
        raise ValueError("Fallback db4 DWT requires an even-length signal at each level.")
    half = n // 2
    approx = np.zeros(half)
    detail = np.zeros(half)
    for k in range(half):
        idx = (2 * k + np.arange(len(DB4_DEC_LO))) % n
        approx[k] = float(np.dot(DB4_DEC_LO, x[idx]))
        detail[k] = float(np.dot(DB4_DEC_HI, x[idx]))
    return approx, detail


def idwt_periodic_even(approx: np.ndarray, detail: np.ndarray) -> np.ndarray:
    half = len(approx)
    n = half * 2
    y = np.zeros(n)
    for k in range(half):
        idx = (2 * k + np.arange(len(DB4_DEC_LO))) % n
        y[idx] += DB4_DEC_LO * approx[k] + DB4_DEC_HI * detail[k]
    return y


def fallback_db4_denoise(x: np.ndarray, level: int) -> np.ndarray:
    current = np.asarray(x, dtype=float).copy()
    details: List[np.ndarray] = []
    actual_level = 0
    for _ in range(level):
        if len(current) % 2:
            break
        current, detail = dwt_periodic_even(current)
        details.append(detail)
        actual_level += 1
    if actual_level == 0:
        return np.asarray(x, dtype=float).copy()

    sigma = np.median(np.abs(details[0])) / 0.6745
    threshold = sigma * math.sqrt(2.0 * math.log(len(x)))
    details = [soft_threshold(d, threshold) for d in details]

    reconstructed = current
    for detail in reversed(details):
        reconstructed = idwt_periodic_even(reconstructed, detail)
    return reconstructed[: len(x)]


def wavelet_denoise(x: np.ndarray, level: int = WAVELET_LEVEL) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    try:
        import pywt  # type: ignore

        max_level = pywt.dwt_max_level(len(x), pywt.Wavelet(WAVELET).dec_len)
        use_level = max(1, min(level, max_level))
        coeffs = pywt.wavedec(x, WAVELET, mode="periodization", level=use_level)
        finest_detail = coeffs[-1]
        sigma = np.median(np.abs(finest_detail)) / 0.6745
        threshold = sigma * math.sqrt(2.0 * math.log(len(x)))
        coeffs_denoised = [coeffs[0]] + [soft_threshold(c, threshold) for c in coeffs[1:]]
        y = pywt.waverec(coeffs_denoised, WAVELET, mode="periodization")
        return np.asarray(y[: len(x)], dtype=float)
    except Exception:
        return fallback_db4_denoise(x, level=level)


# ---------------------------------------------------------------------------
# Final model inputs and optimization
# ---------------------------------------------------------------------------


def moving_block_bootstrap_means(
    denoised: np.ndarray,
    residuals_centered: np.ndarray,
    n_boot: int,
    block_len: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(denoised)
    k_blocks = int(math.ceil(n / block_len))
    starts = rng.integers(0, n - block_len + 1, size=(n_boot, k_blocks))
    offsets = np.arange(block_len)
    sampled_idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_boot, -1)[:, :n]
    sampled_residuals = residuals_centered[sampled_idx]
    boot_series = denoised[None, :] + sampled_residuals
    return boot_series.mean(axis=1)


def quantiles(values: Iterable[float], probs: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        arr = np.array([0.0])
    return np.quantile(arr, list(probs))


def build_model_inputs(returns: pd.DataFrame) -> ModelInputs:
    window = returns.iloc[-H:].copy()
    rng = np.random.default_rng(RANDOM_SEED)

    denoised_data = {
        col: wavelet_denoise(window[col].to_numpy(dtype=float), level=WAVELET_LEVEL)
        for col in window.columns
    }
    denoised = pd.DataFrame(denoised_data, index=window.index)
    residuals = window - denoised
    h_block = max(2, int(math.floor(H ** (1.0 / 3.0))))

    return_rows = []
    loss_rows = []
    for name in window.columns:
        eps_centered = residuals[name].to_numpy(dtype=float)
        eps_centered = eps_centered - eps_centered.mean()
        boot_means = moving_block_bootstrap_means(
            denoised[name].to_numpy(dtype=float),
            eps_centered,
            n_boot=BOOTSTRAP_B,
            block_len=h_block,
            rng=rng,
        )
        r_l, r_u = np.quantile(boot_means, [TAU, 1.0 - TAU])
        return_rows.append({"stock": name, "rL": r_l, "rU": r_u})

        loss = -window[name].to_numpy(dtype=float)
        loss = loss[loss > 0]
        qa, q_l, q_u, qc = quantiles(loss, [0.05, 0.25, 0.75, 0.95])
        loss_rows.append({"stock": name, "qa": qa, "qL": q_l, "qU": q_u, "qc": qc})

    return_interval = pd.DataFrame(return_rows).set_index("stock")
    loss_interval = pd.DataFrame(loss_rows).set_index("stock")

    benchmark_returns = window.mean(axis=1)
    benchmark_loss = -benchmark_returns.to_numpy(dtype=float)
    benchmark_loss = benchmark_loss[benchmark_loss > 0]
    ba, b_l, b_u, bc = quantiles(benchmark_loss, [0.50, 0.75, 0.90, 0.95])
    risk_budget = pd.Series({"ba": ba, "bL": b_l, "bU": b_u, "bc": bc}, name="B")

    z = denoised.copy()
    z = (z - z.mean(axis=0)) / z.std(axis=0, ddof=0).replace(0, np.nan)
    z = z.fillna(0.0)
    z_values = z.T.to_numpy(dtype=float)
    n_assets = z_values.shape[0]
    dists = np.zeros((n_assets, n_assets))
    for i in range(n_assets):
        for j in range(n_assets):
            dists[i, j] = np.linalg.norm(z_values[i] - z_values[j])
    upper = dists[np.triu_indices(n_assets, k=1)]
    sigma = float(np.median(upper[upper > 0])) if np.any(upper > 0) else 1.0
    kernel = np.exp(-(dists**2) / (2.0 * sigma**2))
    np.fill_diagonal(kernel, 1.0)
    kernel_matrix = pd.DataFrame(kernel, index=window.columns, columns=window.columns)

    return ModelInputs(
        dates=window.index,
        returns_window=window,
        denoised_window=denoised,
        residuals_window=residuals,
        return_interval=return_interval,
        loss_interval=loss_interval,
        risk_budget=risk_budget,
        kernel_matrix=kernel_matrix,
        sigma=sigma,
        h_block=h_block,
    )


def model_vectors(
    inputs: ModelInputs,
    alpha: float,
    beta: float,
    rho: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    ri = inputs.return_interval
    qi = inputs.loss_interval
    b = inputs.risk_budget * rho
    mu = (1.0 - alpha) * ri["rL"].to_numpy(dtype=float) + alpha * ri["rU"].to_numpy(dtype=float)
    risk_coef = (1.0 - beta) * qi["qa"].to_numpy(dtype=float) + beta * qi["qL"].to_numpy(dtype=float)
    risk_bound = float((1.0 - beta) * b["bc"] + beta * b["bU"])
    return mu, risk_coef, risk_bound


def feasible_start(risk_coef: np.ndarray, risk_bound: float) -> Optional[np.ndarray]:
    n = len(risk_coef)
    res = linprog(
        c=np.zeros(n),
        A_ub=[risk_coef],
        b_ub=[risk_bound],
        A_eq=[np.ones(n)],
        b_eq=[1.0],
        bounds=[(0.0, 1.0)] * n,
        method="highs",
    )
    if res.success:
        return np.asarray(res.x, dtype=float)
    return None


def solve_portfolio(
    inputs: ModelInputs,
    alpha: float,
    beta: float,
    lam: float,
    rho: float = RISK_TOLERANCE_RHO,
) -> Dict[str, float]:
    names = list(inputs.return_interval.index)
    n = len(names)
    mu, risk_coef, risk_bound = model_vectors(inputs, alpha=alpha, beta=beta, rho=rho)
    A = inputs.kernel_matrix.to_numpy(dtype=float)

    feasible = feasible_start(risk_coef, risk_bound)
    if feasible is None:
        row: Dict[str, float] = {
            "alpha": alpha,
            "beta": beta,
            "lambda": lam,
            "rho": rho,
            "status": "infeasible",
            "expected_return": np.nan,
            "risk_value": np.nan,
            "risk_bound": risk_bound,
            "risk_slack": np.nan,
            "similarity": np.nan,
            "objective": np.nan,
            "effective_assets": np.nan,
        }
        for name in names:
            row[f"w_{name}"] = np.nan
        return row

    if lam <= 0:
        res = linprog(
            c=-mu,
            A_ub=[risk_coef],
            b_ub=[risk_bound],
            A_eq=[np.ones(n)],
            b_eq=[1.0],
            bounds=[(0.0, 1.0)] * n,
            method="highs",
        )
        success = res.success
        w = np.asarray(res.x if success else feasible, dtype=float)
        status = "optimal" if success else "fallback_feasible"
    else:
        def objective(w_vec: np.ndarray) -> float:
            return float(lam * w_vec @ A @ w_vec - mu @ w_vec)

        def gradient(w_vec: np.ndarray) -> np.ndarray:
            return 2.0 * lam * (A @ w_vec) - mu

        constraints = [
            {"type": "eq", "fun": lambda w_vec: np.sum(w_vec) - 1.0, "jac": lambda w_vec: np.ones(n)},
            {
                "type": "ineq",
                "fun": lambda w_vec: risk_bound - risk_coef @ w_vec,
                "jac": lambda w_vec: -risk_coef,
            },
        ]
        starts = [feasible, np.ones(n) / n]
        best_res = None
        for x0 in starts:
            if risk_coef @ x0 <= risk_bound + 1e-10:
                res = minimize(
                    objective,
                    x0=x0,
                    jac=gradient,
                    method="SLSQP",
                    bounds=[(0.0, 1.0)] * n,
                    constraints=constraints,
                    options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
                )
                if best_res is None or (res.success and res.fun < best_res.fun):
                    best_res = res
        if best_res is not None and best_res.success:
            w = np.asarray(best_res.x, dtype=float)
            status = "optimal"
        else:
            w = feasible
            status = "fallback_feasible"

    w[np.abs(w) < 1e-12] = 0.0
    total = w.sum()
    if total > 0:
        w = w / total

    expected_return = float(mu @ w)
    risk_value = float(risk_coef @ w)
    similarity = float(w @ A @ w)
    objective_value = expected_return - lam * similarity
    row = {
        "alpha": float(alpha),
        "beta": float(beta),
        "lambda": float(lam),
        "rho": float(rho),
        "status": status,
        "expected_return": expected_return,
        "risk_value": risk_value,
        "risk_bound": risk_bound,
        "risk_slack": risk_bound - risk_value,
        "similarity": similarity,
        "objective": objective_value,
        "effective_assets": float(1.0 / np.sum(w**2)),
    }
    for name, value in zip(names, w):
        row[f"w_{name}"] = float(value)
    return row


def run_sensitivity(inputs: ModelInputs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grid_rows = []
    for lam in LAMBDA_GRID:
        for alpha in ALPHA_GRID:
            for beta in BETA_GRID:
                grid_rows.append(solve_portfolio(inputs, alpha, beta, lam, rho=RISK_TOLERANCE_RHO))
    sensitivity = pd.DataFrame(grid_rows)

    lambda_rows = [
        solve_portfolio(inputs, BASE_ALPHA, BASE_BETA, lam, rho=RISK_TOLERANCE_RHO)
        for lam in LAMBDA_GRID
    ]
    lambda_sensitivity = pd.DataFrame(lambda_rows)

    rho_rows = [
        solve_portfolio(inputs, BASE_ALPHA, BASE_BETA, BASE_LAMBDA, rho=rho)
        for rho in RHO_GRID
    ]
    rho_sensitivity = pd.DataFrame(rho_rows)

    return sensitivity, lambda_sensitivity, rho_sensitivity


# ---------------------------------------------------------------------------
# Output tables and figures
# ---------------------------------------------------------------------------


def annualize_return(daily_return: float) -> float:
    return (1.0 + daily_return) ** 252 - 1.0


def write_outputs(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    inputs: ModelInputs,
    sensitivity: pd.DataFrame,
    lambda_sensitivity: pd.DataFrame,
    rho_sensitivity: pd.DataFrame,
    base_result: Dict[str, float],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    interval_table = inputs.return_interval.join(inputs.loss_interval)
    interval_table.loc["风险承受上限B", ["rL", "rU"]] = np.nan
    interval_table.loc["风险承受上限B", ["qa", "qL", "qU", "qc"]] = [
        inputs.risk_budget["ba"],
        inputs.risk_budget["bL"],
        inputs.risk_budget["bU"],
        inputs.risk_budget["bc"],
    ]

    base_df = pd.DataFrame([base_result])
    interval_table.to_csv(OUTPUT_DIR / "01_interval_parameters.csv", encoding="utf-8-sig")
    inputs.kernel_matrix.to_csv(OUTPUT_DIR / "02_kernel_similarity_matrix.csv", encoding="utf-8-sig")
    base_df.to_csv(OUTPUT_DIR / "03_base_solution.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(OUTPUT_DIR / "04_alpha_beta_lambda_sensitivity.csv", index=False, encoding="utf-8-sig")
    lambda_sensitivity.to_csv(OUTPUT_DIR / "05_lambda_sensitivity.csv", index=False, encoding="utf-8-sig")
    rho_sensitivity.to_csv(OUTPUT_DIR / "06_rho_sensitivity.csv", index=False, encoding="utf-8-sig")

    summary = {
        "data_file": str(DATA_FILE),
        "output_dir": str(OUTPUT_DIR),
        "stocks": STOCKS,
        "price_date_start": str(prices.index.min().date()),
        "price_date_end": str(prices.index.max().date()),
        "return_rows_total": int(len(returns)),
        "estimation_window_H": H,
        "estimation_start": str(inputs.returns_window.index.min().date()),
        "estimation_end": str(inputs.returns_window.index.max().date()),
        "bootstrap_B": BOOTSTRAP_B,
        "bootstrap_tau": TAU,
        "bootstrap_block_h": inputs.h_block,
        "wavelet": WAVELET,
        "wavelet_level": WAVELET_LEVEL,
        "kernel_sigma": inputs.sigma,
        "base_alpha": BASE_ALPHA,
        "base_beta": BASE_BETA,
        "base_lambda": BASE_LAMBDA,
        "base_rho": RISK_TOLERANCE_RHO,
        "base_solution": base_result,
    }
    with open(OUTPUT_DIR / "00_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def plot_cumulative_returns(prices: pd.DataFrame) -> Path:
    cumulative = prices / prices.iloc[0] - 1.0
    fig, ax = plt.subplots(figsize=(10.8, 5.8), constrained_layout=True)
    for name in cumulative.columns:
        ax.plot(cumulative.index, cumulative[name], lw=1.9, color=STOCK_COLORS[name], label=name)
    ax.axhline(0, color="#333333", lw=0.8, alpha=0.55)
    ax.grid(axis="y", alpha=0.75)
    ax.set_title("四只股票累计收益表现（2019-2025）", fontsize=15, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("累计收益率")
    pct_axis(ax, "y", decimals=0)
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0.0, 1.02))
    return save_figure(fig, "fig01_cumulative_returns.png")


def plot_wavelet_denoising(inputs: ModelInputs) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.4), sharex=True, constrained_layout=True)
    axes = axes.ravel()
    for ax, name in zip(axes, inputs.returns_window.columns):
        ax.plot(
            inputs.returns_window.index,
            inputs.returns_window[name],
            color="#B8B8B8",
            lw=0.8,
            alpha=0.72,
            label="原始日收益率",
        )
        ax.plot(
            inputs.denoised_window.index,
            inputs.denoised_window[name],
            color=STOCK_COLORS[name],
            lw=1.7,
            label="小波去噪收益率",
        )
        ax.axhline(0, color="#333333", lw=0.7, alpha=0.55)
        ax.set_title(name, fontsize=12, pad=7)
        ax.grid(axis="y", alpha=0.65)
        pct_axis(ax, "y", decimals=1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(f"最新 {H} 个交易日收益率的小波去噪", fontsize=15, y=1.08)
    return save_figure(fig, "fig02_wavelet_denoising.png")


def plot_intervals(inputs: ModelInputs) -> Path:
    names = list(inputs.return_interval.index)
    y = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), constrained_layout=True)

    r_l = inputs.return_interval["rL"].to_numpy(dtype=float)
    r_u = inputs.return_interval["rU"].to_numpy(dtype=float)
    r_mid = (r_l + r_u) / 2.0
    r_err = np.vstack([r_mid - r_l, r_u - r_mid])
    for i, name in enumerate(names):
        axes[0].errorbar(
            r_mid[i],
            y[i],
            xerr=r_err[:, [i]],
            fmt="o",
            ms=6,
            lw=2.0,
            capsize=4,
            color=STOCK_COLORS[name],
        )
    axes[0].axvline(0, color="#333333", lw=0.8, alpha=0.6)
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", alpha=0.7)
    axes[0].set_title(f"Bootstrap 平均收益区间（{int((1 - 2 * TAU) * 100)}%）", fontsize=13, pad=10)
    axes[0].set_xlabel("日均收益率区间")
    pct_axis(axes[0], "x", decimals=2)

    q = inputs.loss_interval
    for i, name in enumerate(names):
        axes[1].plot([q.loc[name, "qa"], q.loc[name, "qc"]], [y[i], y[i]], lw=5.0, color="#D5D5D5")
        axes[1].plot([q.loc[name, "qL"], q.loc[name, "qU"]], [y[i], y[i]], lw=5.0, color=STOCK_COLORS[name])
        axes[1].scatter(q.loc[name, ["qa", "qL", "qU", "qc"]], [y[i]] * 4, s=[25, 35, 35, 25], color=STOCK_COLORS[name])
    b = inputs.risk_budget
    axes[1].axvspan(b["bU"], b["bc"], color=NATURE_COLORS["orange"], alpha=0.16, label="风险承受上限核心区间")
    axes[1].axvline(b["bU"], color=NATURE_COLORS["orange"], lw=1.3)
    axes[1].axvline(b["bc"], color=NATURE_COLORS["orange"], lw=1.3, ls="--")
    axes[1].set_yticks(y, names)
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", alpha=0.7)
    axes[1].set_title("风险损失率梯形区间参数", fontsize=13, pad=10)
    axes[1].set_xlabel("日损失率")
    pct_axis(axes[1], "x", decimals=2)
    axes[1].legend(loc="lower right")

    return save_figure(fig, "fig03_return_and_risk_intervals.png")


def plot_kernel_similarity(inputs: ModelInputs) -> Path:
    mat = inputs.kernel_matrix
    fig, ax = plt.subplots(figsize=(6.4, 5.6), constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list(
        "nature_kernel",
        ["#F7F7F7", "#91D1C2", "#4DBBD5", "#3C5488"],
    )
    im = ax.imshow(mat.to_numpy(dtype=float), vmin=0, vmax=1, cmap=cmap)
    ax.set_xticks(np.arange(len(mat.columns)), mat.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(mat.index)), mat.index)
    for i in range(len(mat.index)):
        for j in range(len(mat.columns)):
            value = mat.iloc[i, j]
            color = "white" if value > 0.62 else "#222222"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=10)
    ax.set_title("基于去噪收益特征的核相似度矩阵 A", fontsize=14, pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("相似度")
    return save_figure(fig, "fig04_kernel_similarity_matrix.png")


def weight_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("w_")]


def clean_weight_label(col: str) -> str:
    return col.replace("w_", "")


def plot_base_solution(base_result: Dict[str, float]) -> Path:
    weight_items = [(k.replace("w_", ""), v) for k, v in base_result.items() if k.startswith("w_")]
    names = [k for k, _ in weight_items]
    weights = np.array([v for _, v in weight_items], dtype=float)
    colors = [STOCK_COLORS[name] for name in names]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.4), constrained_layout=True)
    axes[0].bar(names, weights, color=colors, width=0.62)
    axes[0].set_ylim(0, max(0.25, weights.max() * 1.22))
    axes[0].grid(axis="y", alpha=0.7)
    axes[0].set_title(f"基准情形最优权重 α={BASE_ALPHA}, β={BASE_BETA}, λ={BASE_LAMBDA:g}", fontsize=13, pad=10)
    axes[0].set_ylabel("投资比例")
    axes[0].tick_params(axis="x", rotation=20)
    pct_axis(axes[0], "y", decimals=0)
    for i, value in enumerate(weights):
        axes[0].text(i, value + max(0.01, weights.max() * 0.03), f"{value:.1%}", ha="center", va="bottom", fontsize=10)

    metric_names = ["期望收益", "风险约束值", "风险上限", "相似度惩罚项"]
    metric_values = [
        base_result["expected_return"],
        base_result["risk_value"],
        base_result["risk_bound"],
        base_result["similarity"] * BASE_LAMBDA,
    ]
    metric_colors = [NATURE_COLORS["green"], NATURE_COLORS["red"], NATURE_COLORS["orange"], NATURE_COLORS["blue"]]
    axes[1].bar(metric_names, metric_values, color=metric_colors, width=0.62)
    axes[1].grid(axis="y", alpha=0.7)
    axes[1].set_title("基准情形目标与约束", fontsize=13, pad=10)
    axes[1].tick_params(axis="x", rotation=18)
    pct_axis(axes[1], "y", decimals=2)
    for i, value in enumerate(metric_values):
        axes[1].text(i, value + max(0.0001, max(metric_values) * 0.035), f"{value:.3%}", ha="center", va="bottom", fontsize=9)
    return save_figure(fig, "fig05_base_optimal_solution.png")


def pivot_metric(df: pd.DataFrame, lam: float, metric: str) -> pd.DataFrame:
    sub = df[np.isclose(df["lambda"], lam)].copy()
    return sub.pivot(index="beta", columns="alpha", values=metric).sort_index(ascending=False)


def plot_alpha_beta_sensitivity(sensitivity: pd.DataFrame, metric: str, title: str, filename: str) -> Path:
    table = pivot_metric(sensitivity, BASE_LAMBDA, metric)
    values = table.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 6.6), constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list(
        "nature_heat",
        ["#F7F7F7", "#91D1C2", "#4DBBD5", "#3C5488"],
    )
    im = ax.imshow(values, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(table.columns)), [f"{c:.1f}" for c in table.columns])
    ax.set_yticks(np.arange(len(table.index)), [f"{c:.1f}" for c in table.index])
    ax.set_xlabel("乐观系数 α")
    ax.set_ylabel("满意水平 β")
    ax.set_title(title, fontsize=14, pad=12)
    finite = values[np.isfinite(values)]
    threshold = np.nanmedian(finite) if finite.size else 0.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isfinite(value):
                text_color = "white" if value > threshold else "#222222"
                ax.text(j, i, f"{value:.2%}", ha="center", va="center", fontsize=8.5, color=text_color)
            else:
                ax.text(j, i, "不可行", ha="center", va="center", fontsize=8, color="#777777")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=1))
    return save_figure(fig, filename)


def plot_weight_sensitivity_beta(sensitivity: pd.DataFrame) -> Path:
    sub = sensitivity[
        np.isclose(sensitivity["lambda"], BASE_LAMBDA) & np.isclose(sensitivity["alpha"], BASE_ALPHA)
    ].copy()
    sub = sub.sort_values("beta")
    cols = weight_columns(sub)
    fig, ax = plt.subplots(figsize=(10.4, 5.8), constrained_layout=True)
    bottom = np.zeros(len(sub))
    x = sub["beta"].to_numpy(dtype=float)
    for col in cols:
        name = clean_weight_label(col)
        values = sub[col].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, width=0.072, color=STOCK_COLORS[name], label=name)
        bottom += values
    ax.set_title(f"β 灵敏度下的权重变化（α={BASE_ALPHA}, λ={BASE_LAMBDA:g}）", fontsize=14, pad=12)
    ax.set_xlabel("满意水平 β")
    ax.set_ylabel("投资比例")
    ax.set_xticks(x, [f"{v:.1f}" for v in x])
    pct_axis(ax, "y", decimals=0)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    ax.set_ylim(0, 1.02)
    return save_figure(fig, "fig08_beta_weight_sensitivity.png")


def plot_lambda_sensitivity(lambda_sensitivity: pd.DataFrame) -> Path:
    sub = lambda_sensitivity.sort_values("lambda").copy()
    x = np.arange(len(sub))
    labels = [f"{v:g}" for v in sub["lambda"]]
    cols = weight_columns(sub)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.2), constrained_layout=True)
    ax = axes[0]
    ax.plot(x, sub["expected_return"], marker="o", lw=2.0, color=NATURE_COLORS["green"], label="期望收益")
    ax.plot(x, sub["objective"], marker="s", lw=2.0, color=NATURE_COLORS["blue"], label="目标函数值")
    ax2 = ax.twinx()
    ax2.plot(x, sub["similarity"], marker="^", lw=1.9, color=NATURE_COLORS["red"], label="核相似度")
    ax.set_xticks(x, labels)
    ax.set_xlabel("λ")
    ax.set_ylabel("日收益率 / 目标函数值")
    ax2.set_ylabel("WᵀAW")
    ax.grid(axis="y", alpha=0.7)
    pct_axis(ax, "y", decimals=2)
    ax.set_title(f"λ 灵敏度：收益与相似度（α={BASE_ALPHA}, β={BASE_BETA}）", fontsize=13, pad=10)
    lines, line_labels = ax.get_legend_handles_labels()
    lines2, line_labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, line_labels + line_labels2, loc="best")

    bottom = np.zeros(len(sub))
    for col in cols:
        name = clean_weight_label(col)
        values = sub[col].to_numpy(dtype=float)
        axes[1].bar(x, values, bottom=bottom, color=STOCK_COLORS[name], width=0.62, label=name)
        bottom += values
    axes[1].set_xticks(x, labels)
    axes[1].set_xlabel("λ")
    axes[1].set_ylabel("投资比例")
    pct_axis(axes[1], "y", decimals=0)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("λ 灵敏度：最优权重结构", fontsize=13, pad=10)
    axes[1].legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.23))
    return save_figure(fig, "fig09_lambda_sensitivity.png")


def plot_rho_sensitivity(rho_sensitivity: pd.DataFrame) -> Path:
    sub = rho_sensitivity.sort_values("rho").copy()
    cols = weight_columns(sub)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.2), constrained_layout=True)
    axes[0].plot(sub["rho"], sub["expected_return"], marker="o", lw=2.0, color=NATURE_COLORS["green"], label="期望收益")
    axes[0].plot(sub["rho"], sub["risk_value"], marker="s", lw=1.9, color=NATURE_COLORS["red"], label="风险约束左端")
    axes[0].plot(sub["rho"], sub["risk_bound"], marker="^", lw=1.9, color=NATURE_COLORS["orange"], label="风险承受上限")
    axes[0].set_title(f"ρ 灵敏度：风险承受水平变化（α={BASE_ALPHA}, β={BASE_BETA}, λ={BASE_LAMBDA:g}）", fontsize=13, pad=10)
    axes[0].set_xlabel("风险承受系数 ρ")
    axes[0].set_ylabel("日收益率 / 日损失率")
    pct_axis(axes[0], "y", decimals=2)
    axes[0].grid(axis="y", alpha=0.7)
    axes[0].legend(loc="best")

    bottom = np.zeros(len(sub))
    for col in cols:
        name = clean_weight_label(col)
        values = sub[col].to_numpy(dtype=float)
        axes[1].bar(sub["rho"], values, bottom=bottom, width=0.038, color=STOCK_COLORS[name], label=name)
        bottom += values
    axes[1].set_title("ρ 灵敏度：最优权重结构", fontsize=13, pad=10)
    axes[1].set_xlabel("风险承受系数 ρ")
    axes[1].set_ylabel("投资比例")
    axes[1].set_ylim(0, 1.02)
    pct_axis(axes[1], "y", decimals=0)
    axes[1].legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.23))
    return save_figure(fig, "fig10_rho_sensitivity.png")


def plot_annualized_alpha_beta(sensitivity: pd.DataFrame) -> Path:
    sub = sensitivity[np.isclose(sensitivity["lambda"], BASE_LAMBDA)].copy()
    sub["annual_expected_return"] = sub["expected_return"].apply(annualize_return)
    table = sub.pivot(index="beta", columns="alpha", values="annual_expected_return").sort_index(ascending=False)
    values = table.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 6.6), constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list("nature_annual", ["#F7F7F7", "#F39B7F", "#E64B35", "#7E6148"])
    im = ax.imshow(values, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(table.columns)), [f"{c:.1f}" for c in table.columns])
    ax.set_yticks(np.arange(len(table.index)), [f"{c:.1f}" for c in table.index])
    ax.set_xlabel("乐观系数 α")
    ax.set_ylabel("满意水平 β")
    ax.set_title(f"α-β 灵敏度：最优年化收益率（λ={BASE_LAMBDA:g}）", fontsize=14, pad=12)
    finite = values[np.isfinite(values)]
    threshold = np.nanmedian(finite) if finite.size else 0.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isfinite(value):
                ax.text(
                    j,
                    i,
                    f"{value:.1%}",
                    ha="center",
                    va="center",
                    fontsize=8.3,
                    color="white" if value > threshold else "#222222",
                )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0))
    return save_figure(fig, "fig07_alpha_beta_annual_return_sensitivity.png")


def plot_all_figures(
    prices: pd.DataFrame,
    inputs: ModelInputs,
    sensitivity: pd.DataFrame,
    lambda_sensitivity: pd.DataFrame,
    rho_sensitivity: pd.DataFrame,
    base_result: Dict[str, float],
) -> List[Path]:
    paths = [
        plot_cumulative_returns(prices),
        plot_wavelet_denoising(inputs),
        plot_intervals(inputs),
        plot_kernel_similarity(inputs),
        plot_base_solution(base_result),
        plot_alpha_beta_sensitivity(
            sensitivity,
            metric="expected_return",
            title=f"α-β 灵敏度：最优日均收益率（λ={BASE_LAMBDA:g}）",
            filename="fig06_alpha_beta_daily_return_sensitivity.png",
        ),
        plot_annualized_alpha_beta(sensitivity),
        plot_weight_sensitivity_beta(sensitivity),
        plot_lambda_sensitivity(lambda_sensitivity),
        plot_rho_sensitivity(rho_sensitivity),
    ]
    return paths


def main() -> None:
    configure_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prices, returns = read_four_stock_data(DATA_FILE)
    inputs = build_model_inputs(returns)
    sensitivity, lambda_sensitivity, rho_sensitivity = run_sensitivity(inputs)
    base_result = solve_portfolio(
        inputs,
        alpha=BASE_ALPHA,
        beta=BASE_BETA,
        lam=BASE_LAMBDA,
        rho=RISK_TOLERANCE_RHO,
    )

    write_outputs(prices, returns, inputs, sensitivity, lambda_sensitivity, rho_sensitivity, base_result)
    figure_paths = plot_all_figures(prices, inputs, sensitivity, lambda_sensitivity, rho_sensitivity, base_result)

    print("Done.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Base solution status: {base_result['status']}")
    print(f"Base expected daily return: {base_result['expected_return']:.6%}")
    print(f"Base annualized return: {annualize_return(base_result['expected_return']):.2%}")
    print("Base weights:")
    for key, value in base_result.items():
        if key.startswith("w_"):
            print(f"  {key[2:]}: {value:.2%}")
    print("Figures:")
    for path in figure_paths:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
