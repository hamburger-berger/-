# ============================================================
# Task 2: Four-stock Portfolio Optimization
# Based on existing clustering outputs
# ============================================================

import os
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

warnings.filterwarnings("ignore")


# ============================================================
# 0. Parameters
# ============================================================
CLUSTER_DIR = "cluster_results"
OUTPUT_DIR = "solve_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_RETURN_FILE = os.path.join(CLUSTER_DIR, "log_return_matrix.csv")
SELECTED_FILE = os.path.join(CLUSTER_DIR, "selected_4_stocks.csv")

TRADING_DAYS_PER_YEAR = 252

# Model parameters
RISK_FREE_RATE = 0.00       # rf, annualized
TARGET_RETURN = 0.10       # R0, annualized target return

LOWER_BOUND = 0.05
UPPER_BOUND = 0.40
PAIR_SUM_LIMIT = 0.65

# lambda scan
LAMBDA_MIN = 0.0001
LAMBDA_MAX = 0.1000
LAMBDA_STEP = 0.0001


# ============================================================
# 1. Read selected stocks and log returns
# ============================================================
def read_selected_codes(selected_file):
    selected_df = pd.read_csv(selected_file, dtype={"股票代码": str})

    if "股票代码" not in selected_df.columns:
        raise ValueError("selected_4_stocks.csv 中没有 股票代码 列")

    selected_df["股票代码"] = selected_df["股票代码"].astype(str).str.zfill(6)
    selected_codes = selected_df["股票代码"].tolist()

    if len(selected_codes) != 4:
        raise ValueError(f"入选股票数量不是 4，目前为 {len(selected_codes)}")

    return selected_codes, selected_df


def read_log_returns(log_return_file, selected_codes):
    if not os.path.exists(log_return_file):
        raise FileNotFoundError(f"找不到收益率文件：{log_return_file}")

    log_ret = pd.read_csv(log_return_file, index_col=0)
    log_ret.index = pd.to_datetime(log_ret.index)

    # 保证股票代码列都是 6 位字符串
    log_ret.columns = [str(c).zfill(6) for c in log_ret.columns]

    missing = [c for c in selected_codes if c not in log_ret.columns]
    if missing:
        raise ValueError(f"收益率矩阵中缺少这些股票：{missing}")

    selected_ret = log_ret[selected_codes].copy()
    selected_ret = selected_ret.dropna(axis=0, how="any")

    return selected_ret


# ============================================================
# 2. Compute annualized mean and covariance
# ============================================================
def compute_annual_mu_sigma(selected_ret):
    mu = selected_ret.mean().values * TRADING_DAYS_PER_YEAR
    Sigma = selected_ret.cov().values * TRADING_DAYS_PER_YEAR

    # 数值稳定处理：保证协方差矩阵对称
    Sigma = (Sigma + Sigma.T) / 2

    return mu, Sigma


# ============================================================
# 3. Portfolio metrics
# ============================================================
def portfolio_metrics(w, mu, Sigma, rf=0.0):
    w = np.asarray(w, dtype=float)

    annual_return = float(mu @ w)
    annual_variance = float(w @ Sigma @ w)
    annual_volatility = float(np.sqrt(max(annual_variance, 0)))

    if annual_volatility <= 1e-12:
        sharpe = np.nan
    else:
        sharpe = float((annual_return - rf) / annual_volatility)

    return annual_return, annual_volatility, sharpe


# ============================================================
# 4. Constraints
# ============================================================
def make_constraints(mu, include_target=False, target_return=0.10):
    constraints = []

    # sum(w) = 1
    constraints.append({
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1
    })

    # pair constraints: wi + wj <= 0.65
    for i, j in itertools.combinations(range(4), 2):
        constraints.append({
            "type": "ineq",
            "fun": lambda w, i=i, j=j: PAIR_SUM_LIMIT - w[i] - w[j]
        })

    # target return: mu'w >= R0
    if include_target:
        constraints.append({
            "type": "ineq",
            "fun": lambda w: float(mu @ w - target_return)
        })

    return constraints


def check_constraints(w, mu, target_return=None):
    checks = {}

    checks["sum_w"] = np.sum(w)
    checks["min_w"] = np.min(w)
    checks["max_w"] = np.max(w)

    pair_max = max(w[i] + w[j] for i, j in itertools.combinations(range(4), 2))
    checks["max_pair_sum"] = pair_max

    if target_return is not None:
        checks["target_gap"] = float(mu @ w - target_return)

    return checks


# ============================================================
# 5. General optimizer
# ============================================================
def solve_model(mu, Sigma, lam=0.0, include_target=False, target_return=0.10):
    n = len(mu)

    bounds = [(LOWER_BOUND, UPPER_BOUND) for _ in range(n)]
    constraints = make_constraints(mu, include_target, target_return)

    x0 = np.ones(n) / n

    def objective(w):
        # unified model: min w'Sigma w - lambda(mu'w - rf)
        return float(w @ Sigma @ w - lam * (mu @ w - RISK_FREE_RATE))

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "ftol": 1e-12,
            "maxiter": 2000,
            "disp": False
        }
    )

    if not result.success:
        return None, result.message

    w = result.x.copy()

    # numerical cleaning
    w[np.abs(w) < 1e-12] = 0
    w = w / w.sum()

    return w, "success"


# ============================================================
# 6. Feasibility range under structural constraints
# ============================================================
def optimize_return_bound(mu, maximize=True):
    n = len(mu)
    bounds = [(LOWER_BOUND, UPPER_BOUND) for _ in range(n)]
    constraints = make_constraints(mu, include_target=False)

    x0 = np.ones(n) / n

    if maximize:
        obj = lambda w: -float(mu @ w)
    else:
        obj = lambda w: float(mu @ w)

    result = minimize(
        obj,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "ftol": 1e-12,
            "maxiter": 2000,
            "disp": False
        }
    )

    if not result.success:
        raise RuntimeError(f"收益边界求解失败：{result.message}")

    w = result.x / result.x.sum()
    ret = float(mu @ w)

    return ret, w


# ============================================================
# 7. Solve four portfolio schemes
# ============================================================
def solve_portfolios(selected_codes, selected_ret):
    mu, Sigma = compute_annual_mu_sigma(selected_ret)

    mu_df = pd.DataFrame({
        "股票代码": selected_codes,
        "annual_mu": mu
    })

    Sigma_df = pd.DataFrame(
        Sigma,
        index=selected_codes,
        columns=selected_codes
    )

    mu_df.to_csv(
        os.path.join(OUTPUT_DIR, "selected_4_annual_mu.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    Sigma_df.to_csv(
        os.path.join(OUTPUT_DIR, "selected_4_annual_cov_matrix.csv"),
        encoding="utf-8-sig"
    )

    # Feasible return range
    min_feasible_ret, min_ret_w = optimize_return_bound(mu, maximize=False)
    max_feasible_ret, max_ret_w = optimize_return_bound(mu, maximize=True)

    feasible_info = pd.DataFrame({
        "item": ["min_feasible_return", "max_feasible_return", "target_return_original"],
        "value": [min_feasible_ret, max_feasible_ret, TARGET_RETURN]
    })

    feasible_info.to_csv(
        os.path.join(OUTPUT_DIR, "target_return_feasibility.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n========== Target Return Feasibility ==========")
    print(feasible_info.to_string(index=False))

    # If R0 infeasible, adjust to the middle of feasible range
    actual_target_return = TARGET_RETURN

    if TARGET_RETURN > max_feasible_ret:
        actual_target_return = 0.8 * max_feasible_ret + 0.2 * min_feasible_ret
        print(f"\n原始 R0={TARGET_RETURN:.4f} 不可行，自动调整为 {actual_target_return:.4f}")

    if TARGET_RETURN < min_feasible_ret:
        actual_target_return = min_feasible_ret
        print(f"\n原始 R0={TARGET_RETURN:.4f} 低于可行最小收益，自动调整为 {actual_target_return:.4f}")

    # Scheme 1: equal-weight portfolio
    equal_w = np.ones(4) / 4

    # Scheme 2: minimum variance portfolio
    min_var_w, min_var_msg = solve_model(
        mu,
        Sigma,
        lam=0.0,
        include_target=False
    )

    # Scheme 3: target-return minimum variance portfolio
    target_w, target_msg = solve_model(
        mu,
        Sigma,
        lam=0.0,
        include_target=True,
        target_return=actual_target_return
    )

    # Scheme 4: lambda scan and maximum Sharpe
    lambdas = np.arange(
        LAMBDA_MIN,
        LAMBDA_MAX + LAMBDA_STEP / 2,
        LAMBDA_STEP
    )

    frontier_records = []

    for lam in lambdas:
        w, msg = solve_model(
            mu,
            Sigma,
            lam=lam,
            include_target=False
        )

        if w is None:
            continue

        ret, vol, sharpe = portfolio_metrics(w, mu, Sigma, RISK_FREE_RATE)

        record = {
            "lambda": lam,
            "annual_return": ret,
            "annual_volatility": vol,
            "sharpe": sharpe
        }

        for code, weight in zip(selected_codes, w):
            record[f"w_{code}"] = weight

        frontier_records.append(record)

    frontier_df = pd.DataFrame(frontier_records)

    if frontier_df.empty:
        raise RuntimeError("lambda 扫描没有得到任何可行解")

    best_idx = frontier_df["sharpe"].idxmax()
    best_frontier_row = frontier_df.loc[best_idx].copy()

    tangent_w = np.array([
        best_frontier_row[f"w_{code}"] for code in selected_codes
    ])

    # Collect comparison table
    comparison_rows = []

    def add_scheme(name, w, note=""):
        if w is None:
            row = {
                "组合方案": name,
                "annual_return": np.nan,
                "annual_volatility": np.nan,
                "sharpe": np.nan,
                "备注": note
            }
            for code in selected_codes:
                row[f"w_{code}"] = np.nan

            comparison_rows.append(row)
            return

        ret, vol, sharpe = portfolio_metrics(w, mu, Sigma, RISK_FREE_RATE)

        row = {
            "组合方案": name,
            "annual_return": ret,
            "annual_volatility": vol,
            "sharpe": sharpe,
            "备注": note
        }

        for code, weight in zip(selected_codes, w):
            row[f"w_{code}"] = weight

        checks = check_constraints(
            w,
            mu,
            target_return=actual_target_return if "目标收益" in name else None
        )

        row["sum_w"] = checks["sum_w"]
        row["max_pair_sum"] = checks["max_pair_sum"]

        if "target_gap" in checks:
            row["target_gap"] = checks["target_gap"]

        comparison_rows.append(row)

    add_scheme("等权重组合", equal_w, "w_i=25%")
    add_scheme("最小风险组合", min_var_w, min_var_msg)
    add_scheme(
        "目标收益约束最小风险组合",
        target_w,
        f"R0={actual_target_return:.4f}; {target_msg}"
    )
    add_scheme(
        "最优切点组合_最大夏普",
        tangent_w,
        f"lambda*={best_frontier_row['lambda']:.4f}"
    )

    comparison_df = pd.DataFrame(comparison_rows)

    # Save outputs
    frontier_df.to_csv(
        os.path.join(OUTPUT_DIR, "efficient_frontier.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    comparison_df.to_csv(
        os.path.join(OUTPUT_DIR, "portfolio_comparison.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    return mu, Sigma, frontier_df, comparison_df, actual_target_return


# ============================================================
# 8. Plot efficient frontier and weights
# ============================================================
def plot_efficient_frontier(frontier_df, comparison_df):
    fig, ax = plt.subplots(figsize=(8, 6))

    frontier_plot = (
        frontier_df
        .sort_values(["annual_volatility", "annual_return"])
        .drop_duplicates(subset=["annual_volatility", "annual_return"])
    )

    ax.plot(
        frontier_plot["annual_volatility"],
        frontier_plot["annual_return"],
        linewidth=2,
        label="Efficient Frontier"
    )

    for _, row in comparison_df.iterrows():
        if pd.isna(row["annual_return"]) or pd.isna(row["annual_volatility"]):
            continue

        ax.scatter(
            row["annual_volatility"],
            row["annual_return"],
            s=70
        )

        ax.text(
            row["annual_volatility"],
            row["annual_return"],
            row["组合方案"],
            fontsize=8
        )

    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.set_title("Efficient Frontier and Portfolio Schemes")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "efficient_frontier.png"),
        dpi=300
    )
    plt.close()


def plot_portfolio_weights(comparison_df, selected_codes):
    weight_cols = [f"w_{code}" for code in selected_codes]

    plot_df = comparison_df[["组合方案"] + weight_cols].copy()
    plot_df = plot_df.dropna()

    x = np.arange(len(plot_df))
    bottom = np.zeros(len(plot_df))

    fig, ax = plt.subplots(figsize=(9, 5))

    for code in selected_codes:
        values = plot_df[f"w_{code}"].values
        ax.bar(x, values, bottom=bottom, label=code)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["组合方案"], rotation=20, ha="right")
    ax.set_ylabel("Portfolio Weight")
    ax.set_title("Portfolio Weight Comparison")
    ax.legend(title="Stock Code", ncol=2)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "portfolio_weights.png"),
        dpi=300
    )
    plt.close()


# ============================================================
# 9. Main
# ============================================================
def main():
    print("Step 1: 读取聚类选出的 4 只股票")
    selected_codes, selected_df = read_selected_codes(SELECTED_FILE)

    print("Selected codes:", selected_codes)
    print(selected_df.to_string(index=False))

    print("\nStep 2: 读取对数收益率矩阵")
    selected_ret = read_log_returns(LOG_RETURN_FILE, selected_codes)

    print(f"收益率区间：{selected_ret.index.min().date()} 到 {selected_ret.index.max().date()}")
    print(f"收益率矩阵维度：{selected_ret.shape}")

    selected_ret.to_csv(
        os.path.join(OUTPUT_DIR, "selected_4_log_return_matrix.csv"),
        encoding="utf-8-sig"
    )

    print("\nStep 3: 求解组合优化模型")
    mu, Sigma, frontier_df, comparison_df, actual_target_return = solve_portfolios(
        selected_codes,
        selected_ret
    )

    print("\n========== Annualized Mean Return ==========")
    for code, m in zip(selected_codes, mu):
        print(f"{code}: {m:.6f}")

    print("\n========== Annualized Covariance Matrix ==========")
    print(pd.DataFrame(Sigma, index=selected_codes, columns=selected_codes).round(6))

    print("\n========== Portfolio Comparison ==========")
    display_cols = ["组合方案"] + [f"w_{code}" for code in selected_codes] + [
        "annual_return",
        "annual_volatility",
        "sharpe",
        "备注"
    ]

    print(comparison_df[display_cols].to_string(index=False))

    print("\nStep 4: 输出图像")
    plot_efficient_frontier(frontier_df, comparison_df)
    plot_portfolio_weights(comparison_df, selected_codes)

    print("\n全部求解完成。结果保存在文件夹：", OUTPUT_DIR)

    print("\n重点查看这些文件：")
    print("1. solve_results/portfolio_comparison.csv")
    print("2. solve_results/efficient_frontier.csv")
    print("3. solve_results/selected_4_annual_mu.csv")
    print("4. solve_results/selected_4_annual_cov_matrix.csv")
    print("5. solve_results/target_return_feasibility.csv")
    print("6. solve_results/efficient_frontier.png")
    print("7. solve_results/portfolio_weights.png")


if __name__ == "__main__":
    main()