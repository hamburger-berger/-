import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score, calinski_harabasz_score


warnings.filterwarnings("ignore")


# ============================================================
# 0. 参数设置
# ============================================================
DATA_FILE = "all_50_stocks_2019_2025_baostock.csv"

OUTPUT_DIR = "cluster_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RECENT_TRADING_DAYS = 756
TRADING_DAYS_PER_YEAR = 252

MAX_SUSPEND_DAYS = 30
CORR_THRESHOLD = 0.70


# ============================================================
# 1. 读取数据
# ============================================================
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"没有找到数据文件：{file_path}")

    df = pd.read_csv(
        file_path,
        dtype={
            "股票代码": str,
            "code": str,
            "date": str
        }
    )

    if "股票代码" not in df.columns:
        if "code" in df.columns:
            df["股票代码"] = (
                df["code"]
                .astype(str)
                .str.replace("sh.", "", regex=False)
                .str.replace("sz.", "", regex=False)
                .str.zfill(6)
            )
        else:
            raise ValueError("数据中没有 股票代码 或 code 列")

    if "date" not in df.columns:
        raise ValueError("数据中没有 date 列")

    if "close" not in df.columns:
        raise ValueError("数据中没有 close 列")

    df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df = df.dropna(subset=["股票代码", "date", "close"])
    df = df.sort_values(["date", "股票代码"]).reset_index(drop=True)

    return df


# ============================================================
# 2. 构造价格矩阵
# ============================================================
def build_price_matrix(df):
    price = df.pivot(index="date", columns="股票代码", values="close")
    price = price.sort_index()

    if len(price) > RECENT_TRADING_DAYS:
        price = price.iloc[-RECENT_TRADING_DAYS:]

    return price


# ============================================================
# 3. 停牌过滤
# ============================================================
def find_trade_status_col(df):
    for col in df.columns:
        if col.lower().startswith("tradestatus") or col.lower().startswith("tradestatu"):
            return col
    return None


def filter_suspended_stocks(df, price):
    trade_col = find_trade_status_col(df)

    valid_codes = []
    suspend_records = []

    for code in price.columns:
        sub = df[df["股票代码"] == code].copy()
        sub = sub[sub["date"].isin(price.index)]

        if trade_col is not None:
            suspend_days = (sub[trade_col].astype(str) == "0").sum()
        else:
            suspend_days = 0

        keep = suspend_days <= MAX_SUSPEND_DAYS

        suspend_records.append({
            "股票代码": code,
            "停牌天数": suspend_days,
            "是否保留": keep
        })

        if keep:
            valid_codes.append(code)

    suspend_df = pd.DataFrame(suspend_records)

    price = price[valid_codes]

    return price, suspend_df


# ============================================================
# 4. 计算对数收益率
# ============================================================
def compute_log_returns(price):
    price = price.dropna(axis=1, how="all")

    # 对齐交易日，删除存在缺失价格的交易日
    price = price.dropna(axis=0, how="any")

    log_return = np.log(price / price.shift(1))
    log_return = log_return.dropna(axis=0, how="any")

    return log_return


# ============================================================
# 5. 基础统计量
# ============================================================
def compute_stock_statistics(log_return):
    annual_return = log_return.mean() * TRADING_DAYS_PER_YEAR
    annual_volatility = log_return.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    score = annual_return / annual_volatility

    stat = pd.DataFrame({
        "股票代码": log_return.columns,
        "年化收益率": annual_return.values,
        "年化波动率": annual_volatility.values,
        "收益风险比": score.values
    })

    stat = stat.sort_values("收益风险比", ascending=False).reset_index(drop=True)

    return stat


# ============================================================
# 6. 相关系数矩阵和相关距离矩阵
# ============================================================
def compute_corr_distance(log_return):
    corr = log_return.corr()
    corr = corr.clip(-1, 1)

    # 关键修正：不要直接 np.fill_diagonal(distance.values, 0)
    # 先转成 copy=True 的可写数组
    corr_array = corr.to_numpy(copy=True)
    distance_array = np.sqrt(2 * (1 - corr_array))
    np.fill_diagonal(distance_array, 0)

    distance = pd.DataFrame(
        distance_array,
        index=corr.index,
        columns=corr.columns
    )

    return corr, distance


# ============================================================
# 7. 层次聚类
# ============================================================
def run_hierarchical_clustering(distance):
    condensed_dist = squareform(distance.to_numpy(copy=True), checks=False)
    Z = linkage(condensed_dist, method="average")
    return Z


# ============================================================
# 8. 选择最优 k
# ============================================================
def select_best_k(log_return, distance, Z, k_min=2, k_max=8):
    codes = list(distance.columns)

    # 每只股票是一个样本，每个交易日收益率是一个特征
    X = log_return[codes].T.to_numpy(copy=True)

    # 标准化，避免 CH 指标被尺度影响
    X_mean = X.mean(axis=1, keepdims=True)
    X_std = X.std(axis=1, keepdims=True)
    X_std[X_std == 0] = 1
    X_stdzd = (X - X_mean) / X_std

    distance_array = distance.to_numpy(copy=True)

    records = []

    for k in range(k_min, k_max + 1):
        labels = fcluster(Z, t=k, criterion="maxclust")

        sil = silhouette_score(distance_array, labels, metric="precomputed")
        ch = calinski_harabasz_score(X_stdzd, labels)

        records.append({
            "k": k,
            "silhouette": sil,
            "CH": ch
        })

    metric_df = pd.DataFrame(records)

    metric_df["silhouette_rank"] = metric_df["silhouette"].rank(ascending=False)
    metric_df["CH_rank"] = metric_df["CH"].rank(ascending=False)
    metric_df["rank_sum"] = metric_df["silhouette_rank"] + metric_df["CH_rank"]
    metric_df["distance_to_4"] = (metric_df["k"] - 4).abs()

    # 综合排序：两个指标共同较优，同时优先靠近 4 类
    best_row = (
        metric_df
        .sort_values(["rank_sum", "distance_to_4", "k"], ascending=[True, True, True])
        .iloc[0]
    )

    best_k = int(best_row["k"])

    return best_k, metric_df.sort_values("k").reset_index(drop=True)


# ============================================================
# 9. 聚类分组
# ============================================================
def assign_clusters(distance, Z, best_k):
    labels = fcluster(Z, t=best_k, criterion="maxclust")

    cluster_df = pd.DataFrame({
        "股票代码": distance.columns,
        "cluster": labels
    })

    cluster_df = cluster_df.sort_values(["cluster", "股票代码"]).reset_index(drop=True)

    return cluster_df


# ============================================================
# 10. 每类选代表股票
# ============================================================
def select_representative_stocks(cluster_df, stock_stat, corr):
    merged = cluster_df.merge(stock_stat, on="股票代码", how="left")

    # 先给每个股票在本类内部排序
    merged["类内收益风险比排名"] = (
        merged
        .groupby("cluster")["收益风险比"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    cluster_summary = (
        merged
        .groupby("cluster")
        .agg(
            类内股票数量=("股票代码", "count"),
            类内最高收益风险比=("收益风险比", "max"),
            类内平均收益风险比=("收益风险比", "mean")
        )
        .reset_index()
        .sort_values("类内最高收益风险比", ascending=False)
    )

    selected = []

    # 如果类别数 >= 4，选择类内最高收益风险比较高的 4 个类别
    if merged["cluster"].nunique() >= 4:
        chosen_clusters = cluster_summary["cluster"].head(4).tolist()

        for c in chosen_clusters:
            cand = (
                merged[merged["cluster"] == c]
                .sort_values("收益风险比", ascending=False)
                .reset_index(drop=True)
            )

            chosen_code = None

            for _, row in cand.iterrows():
                code = row["股票代码"]

                if len(selected) == 0:
                    chosen_code = code
                    break

                max_corr = corr.loc[code, selected].max()

                if max_corr <= CORR_THRESHOLD:
                    chosen_code = code
                    break

            if chosen_code is None:
                chosen_code = cand.iloc[0]["股票代码"]

            selected.append(chosen_code)

    # 如果类别数 < 4，先每类选一个，再从剩余股票里补
    else:
        for c in cluster_summary["cluster"].tolist():
            cand = (
                merged[merged["cluster"] == c]
                .sort_values("收益风险比", ascending=False)
                .reset_index(drop=True)
            )
            selected.append(cand.iloc[0]["股票代码"])

        remaining = (
            merged[~merged["股票代码"].isin(selected)]
            .sort_values("收益风险比", ascending=False)
            .reset_index(drop=True)
        )

        for _, row in remaining.iterrows():
            if len(selected) >= 4:
                break

            code = row["股票代码"]

            if corr.loc[code, selected].max() <= CORR_THRESHOLD:
                selected.append(code)

        if len(selected) < 4:
            for _, row in remaining.iterrows():
                if len(selected) >= 4:
                    break

                code = row["股票代码"]

                if code not in selected:
                    selected.append(code)

    selected = selected[:4]

    selected_info = (
        merged[merged["股票代码"].isin(selected)]
        .set_index("股票代码")
        .loc[selected]
        .reset_index()
    )

    selected_info["入选顺序"] = range(1, len(selected_info) + 1)

    selected_corr = corr.loc[selected, selected]

    return selected, selected_info, cluster_summary, merged, selected_corr


# ============================================================
# 11. 作图
# ============================================================
def plot_cluster_metrics(metric_df):
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(metric_df["k"], metric_df["silhouette"], marker="o")
    ax1.set_xlabel("Number of clusters k")
    ax1.set_ylabel("Silhouette score")

    ax2 = ax1.twinx()
    ax2.plot(metric_df["k"], metric_df["CH"], marker="s", linestyle="--")
    ax2.set_ylabel("CH index")

    plt.title("Cluster Evaluation: Silhouette and CH Index")
    fig.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, "cluster_metrics.png"), dpi=300)
    plt.close()


def plot_corr_heatmap(corr):
    fig, ax = plt.subplots(figsize=(11, 9))

    im = ax.imshow(corr.to_numpy(copy=True), aspect="auto")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))

    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(corr.index, fontsize=6)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.title("Correlation Heatmap of Candidate Stocks")
    fig.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap_50.png"), dpi=300)
    plt.close()


def plot_selected_corr_heatmap(selected_corr):
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(selected_corr.to_numpy(copy=True), aspect="auto")

    ax.set_xticks(range(len(selected_corr.columns)))
    ax.set_yticks(range(len(selected_corr.index)))

    ax.set_xticklabels(selected_corr.columns, rotation=45, fontsize=9)
    ax.set_yticklabels(selected_corr.index, fontsize=9)

    for i in range(selected_corr.shape[0]):
        for j in range(selected_corr.shape[1]):
            ax.text(
                j,
                i,
                f"{selected_corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9
            )

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.title("Correlation Matrix of Selected 4 Stocks")
    fig.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, "selected_4_corr_heatmap.png"), dpi=300)
    plt.close()


def plot_dendrogram(Z, labels):
    fig, ax = plt.subplots(figsize=(13, 6))

    dendrogram(
        Z,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=7,
        ax=ax
    )

    ax.set_title("Hierarchical Clustering Dendrogram")
    ax.set_xlabel("Stock Code")
    ax.set_ylabel("Correlation Distance")

    fig.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, "dendrogram.png"), dpi=300)
    plt.close()


# ============================================================
# 12. 主程序
# ============================================================
def main():
    print("Step 1: 读取数据")
    raw_df = load_data(DATA_FILE)

    print("Step 2: 构造最近 3 年价格矩阵")
    price = build_price_matrix(raw_df)

    print(f"价格矩阵维度：{price.shape}")
    print(f"价格区间：{price.index.min().date()} 到 {price.index.max().date()}")

    print("Step 3: 停牌过滤")
    price, suspend_df = filter_suspended_stocks(raw_df, price)

    suspend_df.to_csv(
        os.path.join(OUTPUT_DIR, "suspend_check.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print(f"停牌过滤后股票数量：{price.shape[1]}")

    print("Step 4: 计算对数收益率")
    log_return = compute_log_returns(price)

    print(f"收益率矩阵维度：{log_return.shape}")
    print(f"收益率区间：{log_return.index.min().date()} 到 {log_return.index.max().date()}")

    log_return.to_csv(
        os.path.join(OUTPUT_DIR, "log_return_matrix.csv"),
        encoding="utf-8-sig"
    )

    print("Step 5: 计算候选股票基础统计量")
    stock_stat = compute_stock_statistics(log_return)

    stock_stat.to_csv(
        os.path.join(OUTPUT_DIR, "candidate_stock_statistics.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("Step 6: 计算相关系数矩阵和相关距离矩阵")
    corr, distance = compute_corr_distance(log_return)

    corr.to_csv(
        os.path.join(OUTPUT_DIR, "candidate_corr_matrix.csv"),
        encoding="utf-8-sig"
    )

    distance.to_csv(
        os.path.join(OUTPUT_DIR, "candidate_distance_matrix.csv"),
        encoding="utf-8-sig"
    )

    print("Step 7: average-linkage 层次聚类")
    Z = run_hierarchical_clustering(distance)

    print("Step 8: 计算 k=2 到 8 的轮廓系数和 CH 指标")
    best_k, metric_df = select_best_k(log_return, distance, Z)

    metric_df.to_csv(
        os.path.join(OUTPUT_DIR, "cluster_metrics.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n聚类评价指标：")
    print(metric_df.to_string(index=False))

    print(f"\n自动选择的最优聚类数 k* = {best_k}")

    print("Step 9: 输出聚类结果")
    cluster_df = assign_clusters(distance, Z, best_k)

    selected_codes, selected_info, cluster_summary, cluster_full, selected_corr = select_representative_stocks(
        cluster_df,
        stock_stat,
        corr
    )

    cluster_summary.to_csv(
        os.path.join(OUTPUT_DIR, "cluster_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    cluster_full.to_csv(
        os.path.join(OUTPUT_DIR, "cluster_assignment.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    selected_info.to_csv(
        os.path.join(OUTPUT_DIR, "selected_4_stocks.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    selected_corr.to_csv(
        os.path.join(OUTPUT_DIR, "selected_4_corr_matrix.csv"),
        encoding="utf-8-sig"
    )

    print("\n聚类汇总：")
    print(cluster_summary.to_string(index=False))

    print("\n完整聚类分组：")
    print(cluster_full.sort_values(["cluster", "类内收益风险比排名"]).to_string(index=False))

    print("\n最终入选 4 只股票：")
    print(selected_info.to_string(index=False))

    print("\n最终 4 只股票相关系数矩阵：")
    print(selected_corr.round(4).to_string())

    print("Step 10: 输出图像")
    plot_cluster_metrics(metric_df)
    plot_corr_heatmap(corr)
    plot_selected_corr_heatmap(selected_corr)
    plot_dendrogram(Z, labels=list(distance.columns))

    print("\n聚类部分完成。结果保存在文件夹：", OUTPUT_DIR)
    print("\n重点查看：")
    print("1. cluster_results/cluster_metrics.csv")
    print("2. cluster_results/cluster_assignment.csv")
    print("3. cluster_results/selected_4_stocks.csv")
    print("4. cluster_results/selected_4_corr_matrix.csv")
    print("5. cluster_results/cluster_metrics.png")
    print("6. cluster_results/correlation_heatmap_50.png")
    print("7. cluster_results/dendrogram.png")
    print("8. cluster_results/selected_4_corr_heatmap.png")


if __name__ == "__main__":
    main()