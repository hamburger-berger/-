import os
import re
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import pulp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import font_manager as fm
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata


CN_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"
EN_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
fm.fontManager.addfont(CN_FONT_PATH)
fm.fontManager.addfont(EN_FONT_PATH)
CN_FONT = fm.FontProperties(fname=CN_FONT_PATH)
EN_FONT = fm.FontProperties(fname=EN_FONT_PATH, weight="bold")

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"]          = 12
plt.rcParams["font.family"]        = EN_FONT.get_name()
plt.rcParams["font.weight"]        = "bold"
plt.rcParams["axes.labelweight"]   = "bold"
plt.rcParams["axes.titleweight"]   = "bold"

def contains_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', str(text))) if text else False

def beautify(cn_font, en_font):
    for obj in plt.gcf().findobj(match=lambda x: hasattr(x, "get_text")):
        try:
            txt = obj.get_text()
            if not txt:
                continue
            if contains_chinese(txt):
                obj.set_fontproperties(cn_font)
                obj.set_fontweight("normal")
            else:
                obj.set_fontproperties(en_font)
                obj.set_fontweight("bold")
        except Exception:
            pass


PALETTE4 = {
    "A": "#364F6B",
    "B": "#3FC1C9",
    "C": "#F5A623",
    "D": "#FC5185",
}

PALETTE5 = {
    "A": "#7B9EA6",
    "B": "#C17C74",
    "C": "#8DAB8E",
    "D": "#C9A87C",
}

STOCK_NAMES = {
    "A": "贵州茅台",
    "B": "牧原股份",
    "C": "比亚迪",
    "D": "中远海特",
}
STOCKS = ["A", "B", "C", "D"]
YEARS  = [2021, 2022, 2023, 2024, 2025]

WINDOWS = {
    "A": [1, 1, 1, 1, 1],
    "B": [1, 1, 0, 0, 0],
    "C": [0, 1, 1, 1, 1],
    "D": [0, 0, 1, 1, 1],
}

# 基准参数
BASE = dict(
    LAMBDA  = 2.0,
    RF      = 0.02,
    CB      = 0.001,
    CS      = 0.001,
    U       = {"A": 0.60, "B": 0.40, "C": 0.40, "D": 0.30},
    U_CYC   = 0.70,
    R_MAX   = 0.50,
    Q_MAX   = 0.60,
    C_MAX   = 0.40,
    W0      = 1_000_000.0,
)

OUTPUT_BASE = "/root/autodl-tmp/yunchou"
OUTPUT_DIR  = os.path.join(OUTPUT_BASE, "figures")
OUTPUT_DIR2 = os.path.join(OUTPUT_BASE, "figure2")
os.makedirs(OUTPUT_DIR2, exist_ok=True)


param_path = os.path.join(OUTPUT_DIR, "annual_params.csv")
param_table = pd.read_csv(param_path)

real_return = {}
FILES = {
    "A": os.path.join(OUTPUT_BASE, "yunchou/maotai.csv"),
    "B": os.path.join(OUTPUT_BASE, "yunchou/muyuangufen.csv"),
    "C": os.path.join(OUTPUT_BASE, "yunchou/biyadi.csv"),
    "D": os.path.join(OUTPUT_BASE, "yunchou/zhongyuanhaite.csv"),
}
# 修正路径
FILES = {
    "A": "/root/autodl-tmp/yunchou/maotai.csv",
    "B": "/root/autodl-tmp/yunchou/muyuangufen.csv",
    "C": "/root/autodl-tmp/yunchou/biyadi.csv",
    "D": "/root/autodl-tmp/yunchou/zhongyuanhaite.csv",
}
for k, path in FILES.items():
    df = pd.read_csv(path, parse_dates=["日期"]).sort_values("日期")
    df["日收益率"] = df["收盘"].pct_change().fillna(0)
    real_return[k] = {}
    for y in YEARS:
        window = df[df["日期"].dt.year == y]["日收益率"]
        real_return[k][y] = float(252 * window.mean())

def get_param(k, y, col):
    row = param_table[(param_table["股票类别"] == k) & (param_table["年份"] == y)]
    return float(row[col].values[0])


def solve_portfolio(LAMBDA, RF, CB, CS, U, U_CYC, R_MAX, Q_MAX, C_MAX, W0):
    """五年滚动求解，返回累计收益率和每年总资产列表"""
    W     = W0
    h_old = {k: 0.0 for k in STOCKS}
    W_list       = []
    yearly_return = []

    for yi, y in enumerate(YEARS):
        r_hat = {k: get_param(k, y, "年化预测收益率") for k in STOCKS}
        rho   = {k: get_param(k, y, "综合风险rho")    for k in STOCKS}
        delta = {k: WINDOWS[k][yi]                     for k in STOCKS}

        prob = pulp.LpProblem(f"p_{y}", pulp.LpMaximize)
        x = {k: pulp.LpVariable(f"x_{k}", lowBound=0) for k in STOCKS}
        b = {k: pulp.LpVariable(f"b_{k}", lowBound=0) for k in STOCKS}
        s = {k: pulp.LpVariable(f"s_{k}", lowBound=0) for k in STOCKS}
        c = pulp.LpVariable("c", lowBound=0)

        cost = CB * pulp.lpSum(b[k] for k in STOCKS) + CS * pulp.lpSum(s[k] for k in STOCKS)

        prob += (pulp.lpSum((r_hat[k] - LAMBDA * rho[k]) * x[k] for k in STOCKS) + RF * c)
        prob += pulp.lpSum(x[k] for k in STOCKS) + c + cost == W
        for k in STOCKS:
            prob += x[k] == h_old[k] + b[k] - s[k]
            prob += x[k] <= delta[k] * U[k] * W
        prob += x["B"] + x["C"] + x["D"] <= U_CYC * W
        prob += pulp.lpSum(rho[k] * x[k] for k in STOCKS) <= R_MAX * W
        prob += pulp.lpSum(b[k] + s[k] for k in STOCKS) <= Q_MAX * W
        prob += c <= C_MAX * W

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if pulp.LpStatus[prob.status] != "Optimal":
            return None, None, None

        x_val = {k: pulp.value(x[k]) for k in STOCKS}
        c_val = pulp.value(c)

        W_prev = W
        h_new  = {k: x_val[k] * (1 + real_return[k][y]) for k in STOCKS}
        c_new  = c_val * (1 + RF)
        W      = sum(h_new[k] for k in STOCKS) + c_new
        h_old  = h_new

        W_list.append(W)
        yearly_return.append((W - W_prev) / W_prev)

    total_return = (W - W0) / W0 * 100
    return total_return, W_list, yearly_return


# 基准结果
print("计算基准结果...")
base_ret, base_W, base_yr = solve_portfolio(**BASE)
print(f"基准五年累计收益率: {base_ret:.2f}%")


print("\n── λ 灵敏度分析 ──")
lambda_vals = np.round(np.arange(0.5, 5.5, 0.5), 2)
lambda_results = []

for lam in lambda_vals:
    p = dict(BASE)
    p["LAMBDA"] = lam
    ret, W_list, yr = solve_portfolio(**p)
    avg_risk = sum(
        get_param(k, y, "综合风险rho") for k in STOCKS for y in YEARS
    ) / (len(STOCKS) * len(YEARS))
    lambda_results.append({
        "λ": lam,
        "五年累计收益率(%)": round(ret, 4) if ret is not None else None,
        "年均收益率(%)": round(ret / 5, 4) if ret is not None else None,
        "最终资产(万元)": round(W_list[-1] / 1e4, 4) if W_list is not None else None,
        "最佳年收益率(%)": round(max(yr) * 100, 4) if yr is not None else None,
        "最差年收益率(%)": round(min(yr) * 100, 4) if yr is not None else None,
    })
    print(f"  λ={lam:.1f}  累计收益={ret:.2f}%")

lambda_df = pd.DataFrame(lambda_results)
lambda_df.to_csv(os.path.join(OUTPUT_DIR2, "sensitivity_lambda.csv"),
                 index=False, encoding="utf-8-sig")
print("✅ sensitivity_lambda.csv")


print("\n── U_k 灵敏度分析 ──")
U_scan = np.round(np.arange(0.10, 0.81, 0.05), 2)
Uk_results = []

for k in STOCKS:
    for u_val in U_scan:
        p = dict(BASE)
        p["U"] = dict(BASE["U"])
        p["U"][k] = u_val
        ret, W_list, yr = solve_portfolio(**p)
        Uk_results.append({
            "股票类别": k,
            "股票名称": STOCK_NAMES[k],
            f"U_{k}": u_val,
            "五年累计收益率(%)": round(ret, 4) if ret is not None else None,
            "最终资产(万元)": round(W_list[-1] / 1e4, 4) if W_list is not None else None,
            "最佳年收益率(%)": round(max(yr) * 100, 4) if yr is not None else None,
            "最差年收益率(%)": round(min(yr) * 100, 4) if yr is not None else None,
        })
    print(f"  U_{k} 扫描完成")

Uk_df = pd.DataFrame(Uk_results)
Uk_df.to_csv(os.path.join(OUTPUT_DIR2, "sensitivity_Uk.csv"),
             index=False, encoding="utf-8-sig")
print("✅ sensitivity_Uk.csv")


print("\n── R_max 灵敏度分析 ──")
Rmax_vals = np.round(np.arange(0.10, 0.91, 0.05), 2)
Rmax_results = []

for r_max in Rmax_vals:
    p = dict(BASE)
    p["R_MAX"] = r_max
    ret, W_list, yr = solve_portfolio(**p)
    Rmax_results.append({
        "R_max": r_max,
        "五年累计收益率(%)": round(ret, 4) if ret is not None else None,
        "最终资产(万元)": round(W_list[-1] / 1e4, 4) if W_list is not None else None,
        "最佳年收益率(%)": round(max(yr) * 100, 4) if yr is not None else None,
        "最差年收益率(%)": round(min(yr) * 100, 4) if yr is not None else None,
        "年收益波动范围(%)": round((max(yr) - min(yr)) * 100, 4) if yr is not None else None,
    })
    print(f"  R_max={r_max:.2f}  累计收益={ret:.2f}%" if ret is not None else f"  R_max={r_max:.2f}  无解")

Rmax_df = pd.DataFrame(Rmax_results).dropna()
Rmax_df.to_csv(os.path.join(OUTPUT_DIR2, "sensitivity_Rmax.csv"),
               index=False, encoding="utf-8-sig")
print("✅ sensitivity_Rmax.csv")


print("\n── λ × R_max 二维扫描 ──")
lam_grid  = np.round(np.arange(0.5, 5.5, 0.5), 2)
rmax_grid = np.round(np.arange(0.10, 0.91, 0.05), 2)

grid_results = []
for lam in lam_grid:
    for r_max in rmax_grid:
        p = dict(BASE)
        p["LAMBDA"] = lam
        p["R_MAX"]  = r_max
        ret, W_list, yr = solve_portfolio(**p)
        grid_results.append({
            "λ": lam,
            "R_max": r_max,
            "五年累计收益率(%)": round(ret, 4) if ret is not None else np.nan,
        })
    print(f"  λ={lam:.1f} 行完成")

grid_df = pd.DataFrame(grid_results)
grid_df.to_csv(os.path.join(OUTPUT_DIR2, "sensitivity_lambda_Rmax_grid.csv"),
               index=False, encoding="utf-8-sig")
print("✅ sensitivity_lambda_Rmax_grid.csv")




def save(fname):
    beautify(CN_FONT, EN_FONT)
    plt.savefig(os.path.join(OUTPUT_DIR2, fname), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ {fname}")


for scheme_name, PALETTE in [("s4", PALETTE4), ("s5", PALETTE5)]:

    COLOR_LIST = [PALETTE["A"], PALETTE["B"], PALETTE["C"], PALETTE["D"]]
    LINE_COLOR = PALETTE["A"]
    ACCENT     = PALETTE["D"]

    # ── 图1：λ 灵敏度双轴折线图 ──
    lam_x   = lambda_df["λ"].values
    lam_ret = lambda_df["五年累计收益率(%)"].values
    lam_best = lambda_df["最佳年收益率(%)"].values
    lam_worst = lambda_df["最差年收益率(%)"].values

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    ax1.plot(lam_x, lam_ret, color=LINE_COLOR, linewidth=2.5,
             marker="o", markersize=7, markerfacecolor="white",
             markeredgewidth=2, label="五年累计收益率", zorder=3)
    ax1.fill_between(lam_x, lam_worst, lam_best,
                     color=LINE_COLOR, alpha=0.12, label="年收益率区间")

    # 标注基准点
    base_idx = np.argmin(np.abs(lam_x - BASE["LAMBDA"]))
    ax1.axvline(BASE["LAMBDA"], color="gray", linewidth=1.2,
                linestyle="--", alpha=0.6, label=f"基准 λ={BASE['LAMBDA']}")
    ax1.scatter([lam_x[base_idx]], [lam_ret[base_idx]],
                color=ACCENT, s=100, zorder=5)

    ax2.plot(lam_x, lam_ret / lam_x, color=ACCENT, linewidth=1.8,
             linestyle="--", marker="s", markersize=5,
             markerfacecolor="white", markeredgewidth=1.5,
             label="收益/风险厌恶比")
    ax2.set_ylabel("收益 / λ 比值", color=ACCENT)
    ax2.tick_params(axis="y", labelcolor=ACCENT)

    ax1.set_xlabel("风险厌恶系数 λ")
    ax1.set_ylabel("五年累计收益率（%）")
    ax1.set_title("风险厌恶系数 λ 灵敏度分析", fontsize=14)
    ax1.grid(axis="y", linestyle="--", alpha=0.3, color="gray")
    ax1.spines["top"].set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               prop=CN_FONT, loc="upper right", framealpha=0.85)
    save(f"fig1_lambda_{scheme_name}.png")

    # ── 图2：U_k 灵敏度分组折线图 ──
    fig, ax = plt.subplots(figsize=(9, 5))
    for ki, k in enumerate(STOCKS):
        sub = Uk_df[Uk_df["股票类别"] == k].dropna(subset=["五年累计收益率(%)"])
        u_col = f"U_{k}"
        ax.plot(sub[u_col], sub["五年累计收益率(%)"],
                color=COLOR_LIST[ki], linewidth=2.2,
                marker="o", markersize=6, markerfacecolor="white",
                markeredgewidth=2, label=STOCK_NAMES[k])
        # 标基准点
        base_u = BASE["U"][k]
        base_row = sub[np.abs(sub[u_col] - base_u) < 0.01]
        if len(base_row) > 0:
            ax.scatter([base_u], [base_row["五年累计收益率(%)"].values[0]],
                       color=COLOR_LIST[ki], s=80, zorder=5, edgecolors="white", linewidths=1.5)

    ax.axvline(x=-1, color="gray", linewidth=1, linestyle="--", alpha=0)  # dummy
    ax.set_xlabel("最大投资比例 U_k")
    ax.set_ylabel("五年累计收益率（%）")
    ax.set_title("各股票最大投资比例 U_k 灵敏度分析", fontsize=14)
    ax.legend(prop=CN_FONT, framealpha=0.85)
    ax.grid(linestyle="--", alpha=0.3, color="gray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(f"fig2_Uk_{scheme_name}.png")

    # ── 图3：R_max 灵敏度折线 + 波动阴影 ──
    rmax_x    = Rmax_df["R_max"].values
    rmax_ret  = Rmax_df["五年累计收益率(%)"].values
    rmax_best = Rmax_df["最佳年收益率(%)"].values
    rmax_worst= Rmax_df["最差年收益率(%)"].values
    rmax_range= Rmax_df["年收益波动范围(%)"].values

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    ax1.plot(rmax_x, rmax_ret, color=LINE_COLOR, linewidth=2.5,
             marker="o", markersize=7, markerfacecolor="white",
             markeredgewidth=2, label="五年累计收益率", zorder=3)
    ax1.fill_between(rmax_x, rmax_worst, rmax_best,
                     color=LINE_COLOR, alpha=0.13, label="年收益率区间")
    ax1.axvline(BASE["R_MAX"], color="gray", linewidth=1.2,
                linestyle="--", alpha=0.6, label=f"基准 R_max={BASE['R_MAX']}")

    ax2.plot(rmax_x, rmax_range, color=ACCENT, linewidth=1.8,
             linestyle="--", marker="s", markersize=5,
             markerfacecolor="white", markeredgewidth=1.5,
             label="年收益率波动范围")
    ax2.set_ylabel("年收益率波动范围（%）", color=ACCENT)
    ax2.tick_params(axis="y", labelcolor=ACCENT)

    ax1.set_xlabel("组合风险上限 R_max")
    ax1.set_ylabel("五年累计收益率（%）")
    ax1.set_title("组合风险上限 R_max 灵敏度分析", fontsize=14)
    ax1.grid(axis="y", linestyle="--", alpha=0.3, color="gray")
    ax1.spines["top"].set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               prop=CN_FONT, loc="lower right", framealpha=0.85)
    save(f"fig3_Rmax_{scheme_name}.png")

    # ── 图4：λ × R_max 热力图（viridis）──
    pivot = grid_df.pivot(index="R_max", columns="λ", values="五年累计收益率(%)")
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto",
                   origin="lower", interpolation="bilinear")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.1f}" for v in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.2f}" for v in pivot.index])
    ax.set_xlabel("风险厌恶系数 λ")
    ax.set_ylabel("组合风险上限 R_max")
    ax.set_title("λ × R_max 双参数灵敏度热力图\n（颜色=五年累计收益率%）", fontsize=13)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("五年累计收益率（%）")

    # 标注基准点
    base_lam_idx  = list(pivot.columns).index(
        min(pivot.columns, key=lambda v: abs(v - BASE["LAMBDA"])))
    base_rmax_idx = list(pivot.index).index(
        min(pivot.index, key=lambda v: abs(v - BASE["R_MAX"])))
    ax.scatter([base_lam_idx], [base_rmax_idx], color="white",
               s=120, zorder=5, marker="*", label="基准点")
    ax.legend(prop=CN_FONT, loc="upper right")
    save(f"fig4_heatmap_{scheme_name}.png")

    # ── 图5：λ × R_max 3D曲面图 ──
    lam_u  = np.sort(grid_df["λ"].unique())
    rmax_u = np.sort(grid_df["R_max"].unique())
    LAM_M, RMAX_M = np.meshgrid(lam_u, rmax_u)

    Z = pivot.values.copy()
    from numpy import nanmean
    nan_mask = np.isnan(Z)
    if nan_mask.any():
        Z[nan_mask] = np.nanmean(Z)

    fig = plt.figure(figsize=(11, 7))
    ax3d = fig.add_subplot(111, projection="3d")

    surf = ax3d.plot_surface(LAM_M, RMAX_M, Z,
                             cmap="viridis", edgecolor="none",
                             alpha=0.92, antialiased=True)


    base_z = grid_df[
        (np.abs(grid_df["λ"] - BASE["LAMBDA"]) < 0.01) &
        (np.abs(grid_df["R_max"] - BASE["R_MAX"]) < 0.01)
    ]["五年累计收益率(%)"].values
    if len(base_z) > 0:
        ax3d.scatter([BASE["LAMBDA"]], [BASE["R_MAX"]], [base_z[0]],
                     color="white", s=120, zorder=5, marker="*",
                     edgecolors="black", linewidths=0.8)

    ax3d.set_xlabel("风险厌恶系数 λ", labelpad=10)
    ax3d.set_ylabel("组合风险上限 R_max", labelpad=10)
    ax3d.set_zlabel("累计收益率（%）", labelpad=8)
    ax3d.set_title("λ × R_max 双参数灵敏度三维曲面", fontsize=13, pad=15)
    ax3d.view_init(elev=28, azim=225)

    cbar3d = fig.colorbar(surf, ax=ax3d, shrink=0.5, aspect=12, pad=0.08)
    cbar3d.set_label("五年累计收益率（%）")

    beautify(CN_FONT, EN_FONT)
    plt.savefig(os.path.join(OUTPUT_DIR2, f"fig5_3d_{scheme_name}.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ fig5_3d_{scheme_name}.png")


summary_rows = []

# λ 影响
lam_range = lambda_df["五年累计收益率(%)"].max() - lambda_df["五年累计收益率(%)"].min()
summary_rows.append({"参数": "λ（风险厌恶系数）",
                     "扫描范围": "0.5 ~ 5.0",
                     "基准值": BASE["LAMBDA"],
                     "基准累计收益率(%)": round(base_ret, 4),
                     "最大收益率(%)": round(lambda_df["五年累计收益率(%)"].max(), 4),
                     "最小收益率(%)": round(lambda_df["五年累计收益率(%)"].min(), 4),
                     "收益率变动范围(%)": round(lam_range, 4)})

# U_k 影响
for k in STOCKS:
    sub = Uk_df[Uk_df["股票类别"] == k].dropna(subset=["五年累计收益率(%)"])
    u_range = sub["五年累计收益率(%)"].max() - sub["五年累计收益率(%)"].min()
    summary_rows.append({"参数": f"U_{k}（{STOCK_NAMES[k]}最大比例）",
                         "扫描范围": "0.10 ~ 0.80",
                         "基准值": BASE["U"][k],
                         "基准累计收益率(%)": round(base_ret, 4),
                         "最大收益率(%)": round(sub["五年累计收益率(%)"].max(), 4),
                         "最小收益率(%)": round(sub["五年累计收益率(%)"].min(), 4),
                         "收益率变动范围(%)": round(u_range, 4)})

# R_max 影响
rmax_range_val = Rmax_df["五年累计收益率(%)"].max() - Rmax_df["五年累计收益率(%)"].min()
summary_rows.append({"参数": "R_max（组合风险上限）",
                     "扫描范围": "0.10 ~ 0.90",
                     "基准值": BASE["R_MAX"],
                     "基准累计收益率(%)": round(base_ret, 4),
                     "最大收益率(%)": round(Rmax_df["五年累计收益率(%)"].max(), 4),
                     "最小收益率(%)": round(Rmax_df["五年累计收益率(%)"].min(), 4),
                     "收益率变动范围(%)": round(rmax_range_val, 4)})

summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.sort_values("收益率变动范围(%)", ascending=False)
summary_df.to_csv(os.path.join(OUTPUT_DIR2, "sensitivity_summary.csv"),
                  index=False, encoding="utf-8-sig")
print("✅ sensitivity_summary.csv")

print("\n" + "=" * 60)
print("全部完成！figure2 目录内容：")
for f in sorted(os.listdir(OUTPUT_DIR2)):
    print(f"  {f}")
print("=" * 60)
