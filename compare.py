import os
import re
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import pulp


STOCK_NAMES = {
    "A": "贵州茅台",
    "B": "牧原股份",
    "C": "比亚迪",
    "D": "中远海特",
}

STOCKS = ["A", "B", "C", "D"]
YEARS = [2021, 2022, 2023, 2024, 2025]

# 投资窗口
WINDOWS = {
    "A": [1, 1, 1, 1, 1],
    "B": [1, 1, 0, 0, 0],
    "C": [0, 1, 1, 1, 1],
    "D": [0, 0, 1, 1, 1],
}

# 模型参数
LAMBDA = 2.0
RF = 0.02
CB = 0.001
CS = 0.001

U = {
    "A": 0.60,
    "B": 0.40,
    "C": 0.40,
    "D": 0.30,
}

U_CYC = 0.70
R_MAX = 0.50
Q_MAX = 0.60
C_MAX = 0.40
W0 = 1_000_000.0

OUTPUT_BASE = "/root/autodl-tmp/yunchou"
OUTPUT_DIR = os.path.join(OUTPUT_BASE, "compare_static_rolling_tables")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILES = {
    "A": "/root/autodl-tmp/yunchou/maotai.csv",
    "B": "/root/autodl-tmp/yunchou/muyuangufen.csv",
    "C": "/root/autodl-tmp/yunchou/biyadi.csv",
    "D": "/root/autodl-tmp/yunchou/zhongyuanhaite.csv",
}


def calc_max_drawdown(prices):
    """
    最大回撤：
    D = max_t (1 - P_t / max_{tau <= t} P_tau)
    """
    cummax = prices.cummax()
    drawdown = 1 - prices / cummax
    return float(drawdown.max())


def calc_annual_param(df_window):

    r = df_window["日收益率"]
    prices = df_window["收盘"]

    ann_r = float(252 * r.mean())
    vol = float(np.sqrt(252) * r.std())
    mdd = calc_max_drawdown(prices)
    rho = 0.5 * vol + 0.5 * mdd

    return {
        "r": ann_r,
        "vol": vol,
        "mdd": mdd,
        "rho": rho,
    }


def fmt_money(x):
    return round(x / 1e4, 4)


print("\n" + "=" * 80)
print("读取股票日频数据")
print("=" * 80)

raw = {}

for k, path in FILES.items():
    df = pd.read_csv(path, parse_dates=["日期"])
    df = df.sort_values("日期").reset_index(drop=True)
    df["日收益率"] = df["收盘"].pct_change().fillna(0)
    raw[k] = df

    print(
        f"{k} | {STOCK_NAMES[k]} | "
        f"{df['日期'].min().date()} ~ {df['日期'].max().date()} | "
        f"{len(df)} 条记录 | 缺失值总数 {df.isna().sum().sum()}"
    )



real_return = {}

for k in STOCKS:
    real_return[k] = {}

    for y in YEARS:
        window = raw[k][raw[k]["日期"].dt.year == y]
        real_return[k][y] = float(252 * window["日收益率"].mean())



rolling_params = {}

for k in STOCKS:
    rolling_params[k] = {}

    for y in YEARS:
        estimate_year = 2021 if y == 2021 else y - 1
        window = raw[k][raw[k]["日期"].dt.year == estimate_year]

        rolling_params[k][y] = calc_annual_param(window)



static_single = {}

for k in STOCKS:
    init_window = raw[k][raw[k]["日期"].dt.year == 2021]
    static_single[k] = calc_annual_param(init_window)

static_params = {
    k: {y: static_single[k] for y in YEARS}
    for k in STOCKS
}



param_rows = []

for k in STOCKS:
    for y in YEARS:
        param_rows.append({
            "股票类别": k,
            "股票名称": STOCK_NAMES[k],
            "年份": y,
            "滚动_预测收益率(%)": round(rolling_params[k][y]["r"] * 100, 4),
            "静态_预测收益率(%)": round(static_params[k][y]["r"] * 100, 4),
            "滚动_年化波动率": round(rolling_params[k][y]["vol"], 6),
            "静态_年化波动率": round(static_params[k][y]["vol"], 6),
            "滚动_最大回撤": round(rolling_params[k][y]["mdd"], 6),
            "静态_最大回撤": round(static_params[k][y]["mdd"], 6),
            "滚动_综合风险rho": round(rolling_params[k][y]["rho"], 6),
            "静态_综合风险rho": round(static_params[k][y]["rho"], 6),
            "投资窗口delta": WINDOWS[k][YEARS.index(y)],
        })

param_df = pd.DataFrame(param_rows)

print("\n" + "=" * 80)
print("参数对比表：滚动参数 vs 静态参数")
print("=" * 80)
print(param_df.to_string(index=False))

param_path = os.path.join(OUTPUT_DIR, "compare_params.csv")
param_df.to_csv(param_path, index=False, encoding="utf-8-sig")



def solve_strategy(param_dict, model_name):

    W = W0
    h_old = {k: 0.0 for k in STOCKS}

    summary_records = []
    holding_records = []

    for yi, y in enumerate(YEARS):
        r_hat = {k: param_dict[k][y]["r"] for k in STOCKS}
        rho = {k: param_dict[k][y]["rho"] for k in STOCKS}
        delta = {k: WINDOWS[k][yi] for k in STOCKS}

        prob = pulp.LpProblem(f"{model_name}_{y}", pulp.LpMaximize)

        x = {k: pulp.LpVariable(f"x_{k}_{y}", lowBound=0) for k in STOCKS}
        b = {k: pulp.LpVariable(f"b_{k}_{y}", lowBound=0) for k in STOCKS}
        s = {k: pulp.LpVariable(f"s_{k}_{y}", lowBound=0) for k in STOCKS}
        c = pulp.LpVariable(f"cash_{y}", lowBound=0)

        cost = (
            CB * pulp.lpSum(b[k] for k in STOCKS)
            + CS * pulp.lpSum(s[k] for k in STOCKS)
        )

        # 目标函数
        prob += (
            pulp.lpSum((r_hat[k] - LAMBDA * rho[k]) * x[k] for k in STOCKS)
            + RF * c
        )

        # 资金平衡约束
        prob += pulp.lpSum(x[k] for k in STOCKS) + c + cost == W

        # 调仓关系与持仓上限
        for k in STOCKS:
            prob += x[k] == h_old[k] + b[k] - s[k]
            prob += x[k] <= delta[k] * U[k] * W

            # 防止无持仓时出现“买入又卖出”的伪交易
            prob += s[k] <= h_old[k]

        # 周期股总仓位约束
        prob += x["B"] + x["C"] + x["D"] <= U_CYC * W

        # 组合风险约束
        prob += pulp.lpSum(rho[k] * x[k] for k in STOCKS) <= R_MAX * W

        # 年度交易规模约束
        prob += pulp.lpSum(b[k] + s[k] for k in STOCKS) <= Q_MAX * W

        # 现金约束
        prob += c <= C_MAX * W

        status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
        status_name = pulp.LpStatus[status]

        if status_name != "Optimal":
            raise RuntimeError(f"{model_name} 在 {y} 年求解失败，状态为 {status_name}")

        x_val = {k: float(pulp.value(x[k])) for k in STOCKS}
        b_val = {k: float(pulp.value(b[k])) for k in STOCKS}
        s_val = {k: float(pulp.value(s[k])) for k in STOCKS}
        c_val = float(pulp.value(c))

        cost_val = CB * sum(b_val.values()) + CS * sum(s_val.values())

        W_start = W

        # 年末资金滚动
        h_new = {
            k: x_val[k] * (1 + real_return[k][y])
            for k in STOCKS
        }

        c_new = c_val * (1 + RF)
        W_end = sum(h_new.values()) + c_new

        stock_total = sum(x_val.values())
        cyc_total = x_val["B"] + x_val["C"] + x_val["D"]

        summary_records.append({
            "模型": model_name,
            "年份": y,
            "年初总资产(元)": round(W_start, 2),
            "年末总资产(元)": round(W_end, 2),
            "年初总资产(万元)": fmt_money(W_start),
            "年末总资产(万元)": fmt_money(W_end),
            "当年收益率(%)": round((W_end / W_start - 1) * 100, 4),
            "现金(元)": round(c_val, 2),
            "现金(万元)": fmt_money(c_val),
            "现金占比(%)": round(c_val / W_start * 100, 4),
            "股票总持仓(元)": round(stock_total, 2),
            "股票总持仓(万元)": fmt_money(stock_total),
            "股票仓位(%)": round(stock_total / W_start * 100, 4),
            "周期股持仓(元)": round(cyc_total, 2),
            "周期股持仓(万元)": fmt_money(cyc_total),
            "周期股占总资产比例(%)": round(cyc_total / W_start * 100, 4),
            "周期股占股票持仓比例(%)": round(cyc_total / stock_total * 100, 4)
            if stock_total > 0 else 0.0,
            "交易成本(元)": round(cost_val, 2),
            "交易成本(万元)": fmt_money(cost_val),
        })

        for k in STOCKS:
            holding_records.append({
                "模型": model_name,
                "年份": y,
                "股票类别": k,
                "股票名称": STOCK_NAMES[k],
                "delta": delta[k],
                "预测收益率(%)": round(r_hat[k] * 100, 4),
                "综合风险rho": round(rho[k], 6),
                "目标持仓(元)": round(x_val[k], 2),
                "目标持仓(万元)": fmt_money(x_val[k]),
                "买入(元)": round(b_val[k], 2),
                "买入(万元)": fmt_money(b_val[k]),
                "卖出(元)": round(s_val[k], 2),
                "卖出(万元)": fmt_money(s_val[k]),
                "持仓占年初资产比例(%)": round(x_val[k] / W_start * 100, 4),
            })

        W = W_end
        h_old = h_new

    return pd.DataFrame(summary_records), pd.DataFrame(holding_records)



print("\n" + "=" * 80)
print("开始求解：动态滚动参数模型 vs 静态固定参数模型")
print("=" * 80)

summary_roll, holdings_roll = solve_strategy(rolling_params, "Rolling")
summary_static, holdings_static = solve_strategy(static_params, "Static")

summary_df = pd.concat([summary_roll, summary_static], ignore_index=True)
holding_df = pd.concat([holdings_roll, holdings_static], ignore_index=True)



overview_rows = []

for model_name in ["Rolling", "Static"]:
    sub = summary_df[summary_df["模型"] == model_name].copy()

    final_asset = sub.iloc[-1]["年末总资产(元)"]
    cum_ret = (final_asset / W0 - 1) * 100

    overview_rows.append({
        "模型": model_name,
        "模型说明": "日频滚动参数模型" if model_name == "Rolling" else "初始静态固定参数模型",
        "最终资产(万元)": fmt_money(final_asset),
        "五年累计收益率(%)": round(cum_ret, 4),
        "年均收益率(%)": round(cum_ret / 5, 4),
        "最佳年收益率(%)": round(sub["当年收益率(%)"].max(), 4),
        "最差年收益率(%)": round(sub["当年收益率(%)"].min(), 4),
        "年收益率标准差": round(sub["当年收益率(%)"].std(), 4),
        "平均现金占比(%)": round(sub["现金占比(%)"].mean(), 4),
        "平均股票仓位(%)": round(sub["股票仓位(%)"].mean(), 4),
        "平均周期股占总资产比例(%)": round(sub["周期股占总资产比例(%)"].mean(), 4),
        "最高周期股占总资产比例(%)": round(sub["周期股占总资产比例(%)"].max(), 4),
        "累计交易成本(万元)": round(sub["交易成本(万元)"].sum(), 4),
    })

overview_df = pd.DataFrame(overview_rows)



yearly_compare_rows = []

for y in YEARS:
    roll_row = summary_df[(summary_df["模型"] == "Rolling") & (summary_df["年份"] == y)].iloc[0]
    stat_row = summary_df[(summary_df["模型"] == "Static") & (summary_df["年份"] == y)].iloc[0]

    yearly_compare_rows.append({
        "年份": y,
        "Rolling_年末资产(万元)": roll_row["年末总资产(万元)"],
        "Static_年末资产(万元)": stat_row["年末总资产(万元)"],
        "资产差值_Rolling-Static(万元)": round(
            roll_row["年末总资产(万元)"] - stat_row["年末总资产(万元)"], 4
        ),
        "Rolling_当年收益率(%)": roll_row["当年收益率(%)"],
        "Static_当年收益率(%)": stat_row["当年收益率(%)"],
        "收益率差值_Rolling-Static(%)": round(
            roll_row["当年收益率(%)"] - stat_row["当年收益率(%)"], 4
        ),
        "Rolling_现金占比(%)": roll_row["现金占比(%)"],
        "Static_现金占比(%)": stat_row["现金占比(%)"],
        "Rolling_周期股占总资产比例(%)": roll_row["周期股占总资产比例(%)"],
        "Static_周期股占总资产比例(%)": stat_row["周期股占总资产比例(%)"],
    })

yearly_compare_df = pd.DataFrame(yearly_compare_rows)




holding_compare_rows = []

for y in YEARS:
    for k in STOCKS:
        roll_row = holding_df[
            (holding_df["模型"] == "Rolling")
            & (holding_df["年份"] == y)
            & (holding_df["股票类别"] == k)
        ].iloc[0]

        stat_row = holding_df[
            (holding_df["模型"] == "Static")
            & (holding_df["年份"] == y)
            & (holding_df["股票类别"] == k)
        ].iloc[0]

        holding_compare_rows.append({
            "年份": y,
            "股票类别": k,
            "股票名称": STOCK_NAMES[k],
            "delta": roll_row["delta"],
            "Rolling_目标持仓(万元)": roll_row["目标持仓(万元)"],
            "Static_目标持仓(万元)": stat_row["目标持仓(万元)"],
            "持仓差值_Rolling-Static(万元)": round(
                roll_row["目标持仓(万元)"] - stat_row["目标持仓(万元)"], 4
            ),
            "Rolling_持仓占比(%)": roll_row["持仓占年初资产比例(%)"],
            "Static_持仓占比(%)": stat_row["持仓占年初资产比例(%)"],
            "Rolling_买入(万元)": roll_row["买入(万元)"],
            "Static_买入(万元)": stat_row["买入(万元)"],
            "Rolling_卖出(万元)": roll_row["卖出(万元)"],
            "Static_卖出(万元)": stat_row["卖出(万元)"],
        })

holding_compare_df = pd.DataFrame(holding_compare_rows)




overview_path = os.path.join(OUTPUT_DIR, "compare_overview.csv")
summary_path = os.path.join(OUTPUT_DIR, "compare_yearly_summary.csv")
holding_path = os.path.join(OUTPUT_DIR, "compare_holdings.csv")
yearly_compare_path = os.path.join(OUTPUT_DIR, "compare_yearly_diff.csv")
holding_compare_path = os.path.join(OUTPUT_DIR, "compare_holding_diff.csv")

overview_df.to_csv(overview_path, index=False, encoding="utf-8-sig")
summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
holding_df.to_csv(holding_path, index=False, encoding="utf-8-sig")
yearly_compare_df.to_csv(yearly_compare_path, index=False, encoding="utf-8-sig")
holding_compare_df.to_csv(holding_compare_path, index=False, encoding="utf-8-sig")



print("\n" + "=" * 80)
print("总览对比表")
print("=" * 80)
print(overview_df.to_string(index=False))

print("\n" + "=" * 80)
print("逐年表现对比表")
print("=" * 80)
print(yearly_compare_df.to_string(index=False))

print("\n" + "=" * 80)
print("逐年逐股票持仓对比表")
print("=" * 80)
print(holding_compare_df.to_string(index=False))

print("\n" + "=" * 80)
print("Rolling 模型逐年汇总")
print("=" * 80)
print(summary_roll.to_string(index=False))

print("\n" + "=" * 80)
print("Static 模型逐年汇总")
print("=" * 80)
print(summary_static.to_string(index=False))

print("\n" + "=" * 80)
print("结果文件已保存")
print("=" * 80)
print(f"参数对比表: {param_path}")
print(f"总览对比表: {overview_path}")
print(f"逐年汇总表: {summary_path}")
print(f"逐年差异表: {yearly_compare_path}")
print(f"持仓明细表: {holding_path}")
print(f"持仓差异表: {holding_compare_path}")
print("=" * 80)