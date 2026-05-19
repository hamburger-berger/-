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
from matplotlib.patches import Patch



CN_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"
EN_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"

fm.fontManager.addfont(CN_FONT_PATH)
fm.fontManager.addfont(EN_FONT_PATH)
CN_FONT = fm.FontProperties(fname=CN_FONT_PATH)
EN_FONT = fm.FontProperties(fname=EN_FONT_PATH, weight="bold")

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = EN_FONT.get_name()
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"


def contains_chinese(text):
    if text is None:
        return False
    return bool(re.search(r'[\u4e00-\u9fff]', str(text)))


def beautify(cn_font, en_font):
    fig = plt.gcf()
    for obj in fig.findobj(match=lambda x: hasattr(x, "get_text")):
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




COLORS = {
    "A": "#2E86AB",
    "B": "#E84855",
    "C": "#3BB273",
    "D": "#F4A261",
}

STOCK_NAMES = {
    "A": "贵州茅台（A类）",
    "B": "牧原股份（B类）",
    "C": "比亚迪（C类）",
    "D": "中远海特（D类）",
}

STOCKS = ["A", "B", "C", "D"]
YEARS  = [2021, 2022, 2023, 2024, 2025]

# 投资时间窗口
WINDOWS = {
    "A": [1, 1, 1, 1, 1],
    "B": [1, 1, 0, 0, 0],
    "C": [0, 1, 1, 1, 1],
    "D": [0, 0, 1, 1, 1],
}

# 模型参数
LAMBDA   = 2.0    # 风险厌恶系数
RF       = 0.02   # 无风险收益率（年化）
CB       = 0.001  # 买入成本率
CS       = 0.001  # 卖出成本率
U        = {"A": 0.60, "B": 0.40, "C": 0.40, "D": 0.30}  # 各类最大比例
U_CYC    = 0.70   # 周期股总仓位上限
R_MAX    = 0.50   # 组合风险上限
Q_MAX    = 0.60   # 年度最大交易规模比例
C_MAX    = 0.40   # 最大现金比例

W0       = 1_000_000.0   # 初始总资产（元）

OUTPUT_DIR = "/root/autodl-tmp/yunchou/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)



param_path = os.path.join(OUTPUT_DIR, "annual_params.csv")

if os.path.exists(param_path):
    param_table = pd.read_csv(param_path)
    print(f"✅ 读取年度参数表: {param_path}")
else:

    print("⚠️  未找到 annual_params.csv，重新计算年度参数...")

    FILES = {
        "A": "/root/autodl-tmp/yunchou/maotai.csv",
        "B": "/root/autodl-tmp/yunchou/muyuangufen.csv",
        "C": "/root/autodl-tmp/yunchou/biyadi.csv",
        "D": "/root/autodl-tmp/yunchou/zhongyuanhaite.csv",
    }

    def calc_max_drawdown(prices):
        cummax = prices.cummax()
        drawdown = (prices - cummax) / cummax
        return float(-drawdown.min())

    rows = []
    for k, path in FILES.items():
        df = pd.read_csv(path, parse_dates=["日期"]).sort_values("日期")
        df["日收益率"] = df["收盘"].pct_change().fillna(0)
        for yi, y in enumerate(YEARS):
            window = df[df["日期"].dt.year == (2021 if y == 2021 else y - 1)]
            r   = window["日收益率"]
            pr  = window["收盘"]
            ann_r   = float(252 * r.mean())
            vol     = float(np.sqrt(252) * r.std())
            mdd     = calc_max_drawdown(pr)
            rho     = 0.5 * vol + 0.5 * mdd
            rows.append({
                "股票类别": k, "股票名称": STOCK_NAMES[k],
                "年份": y,
                "年化预测收益率": ann_r,
                "年化波动率": vol,
                "最大回撤": mdd,
                "综合风险rho": rho,
                "投资窗口delta": WINDOWS[k][yi],
            })
    param_table = pd.DataFrame(rows)
    param_table.to_csv(param_path, index=False, encoding="utf-8-sig")
    print(f"✅ 年度参数表已保存: {param_path}")


def get_param(k, y, col):
    row = param_table[(param_table["股票类别"] == k) & (param_table["年份"] == y)]
    return float(row[col].values[0])




FILES = {
    "A": "/root/autodl-tmp/yunchou/maotai.csv",
    "B": "/root/autodl-tmp/yunchou/muyuangufen.csv",
    "C": "/root/autodl-tmp/yunchou/biyadi.csv",
    "D": "/root/autodl-tmp/yunchou/zhongyuanhaite.csv",
}

real_return = {}  # real_return[k][y] = 当年实际年化收益率
for k, path in FILES.items():
    df = pd.read_csv(path, parse_dates=["日期"]).sort_values("日期")
    df["日收益率"] = df["收盘"].pct_change().fillna(0)
    real_return[k] = {}
    for y in YEARS:
        window = df[df["日期"].dt.year == y]["日收益率"]
        real_return[k][y] = float(252 * window.mean())

print("\n实际年化收益率（%）：")
for k in STOCKS:
    vals = [f"{real_return[k][y]*100:.2f}%" for y in YEARS]
    print(f"  {STOCK_NAMES[k]}: {dict(zip(YEARS, vals))}")




print("\n" + "=" * 60)
print("开始五年滚动优化求解")
print("=" * 60)

# 初始状态
W      = W0
h_old  = {k: 0.0 for k in STOCKS}   # 调仓前持仓市值
c_old  = 0.0                          # 上一年现金（滚动用）

# 存储每年结果
results = []

for yi, y in enumerate(YEARS):

    print(f"\n{'─'*50}")
    print(f"第 {yi+1} 年（{y}年）优化，总资产 W = {W:,.2f} 元")

    # 取当年参数
    r_hat = {k: get_param(k, y, "年化预测收益率") for k in STOCKS}
    rho   = {k: get_param(k, y, "综合风险rho")    for k in STOCKS}
    delta = {k: WINDOWS[k][yi]                     for k in STOCKS}


    prob = pulp.LpProblem(f"portfolio_{y}", pulp.LpMaximize)

    # 决策变量
    x = {k: pulp.LpVariable(f"x_{k}", lowBound=0) for k in STOCKS}
    b = {k: pulp.LpVariable(f"b_{k}", lowBound=0) for k in STOCKS}
    s = {k: pulp.LpVariable(f"s_{k}", lowBound=0) for k in STOCKS}
    c = pulp.LpVariable("c", lowBound=0)

    # 交易成本
    cost = CB * pulp.lpSum(b[k] for k in STOCKS) + CS * pulp.lpSum(s[k] for k in STOCKS)

    # 目标函数：最大化风险调整收益
    prob += (
        pulp.lpSum((r_hat[k] - LAMBDA * rho[k]) * x[k] for k in STOCKS)
        + RF * c
    )

    # 约束1：资金平衡
    prob += pulp.lpSum(x[k] for k in STOCKS) + c + cost == W

    # 约束2：调仓关系
    for k in STOCKS:
        prob += x[k] == h_old[k] + b[k] - s[k]

    # 约束3：投资时间窗口
    for k in STOCKS:
        prob += x[k] <= delta[k] * U[k] * W
        # 若窗口外，强制为0（delta=0时上界为0）

    # 约束4：周期风险约束
    prob += x["B"] + x["C"] + x["D"] <= U_CYC * W

    # 约束5：组合风险约束
    prob += pulp.lpSum(rho[k] * x[k] for k in STOCKS) <= R_MAX * W

    # 约束6：交易规模约束
    prob += pulp.lpSum(b[k] + s[k] for k in STOCKS) <= Q_MAX * W

    # 约束7：现金约束
    prob += c <= C_MAX * W


    solver = pulp.PULP_CBC_CMD(msg=0)
    status = prob.solve(solver)

    print(f"  求解状态: {pulp.LpStatus[prob.status]}")

    if pulp.LpStatus[prob.status] != "Optimal":
        print("  ⚠️  未找到最优解，跳过本年")
        continue

    # 提取结果
    x_val = {k: pulp.value(x[k]) for k in STOCKS}
    b_val = {k: pulp.value(b[k]) for k in STOCKS}
    s_val = {k: pulp.value(s[k]) for k in STOCKS}
    c_val = pulp.value(c)
    obj   = pulp.value(prob.objective)

    print(f"  目标函数值（风险调整收益）: {obj:,.4f}")
    print(f"  现金保留: {c_val:,.2f} 元")
    for k in STOCKS:
        print(f"  {STOCK_NAMES[k]}: 持仓={x_val[k]:,.2f}  买入={b_val[k]:,.2f}  卖出={s_val[k]:,.2f}  δ={delta[k]}")

    # 存储本年结果
    for k in STOCKS:
        results.append({
            "年份": y,
            "股票类别": k,
            "股票名称": STOCK_NAMES[k],
            "是否允许投资δ": delta[k],
            "年化预测收益率": round(r_hat[k], 6),
            "综合风险ρ": round(rho[k], 6),
            "风险调整收益": round(r_hat[k] - LAMBDA * rho[k], 6),
            "调仓前持仓": round(h_old[k], 2),
            "目标持仓x": round(x_val[k], 2),
            "买入b": round(b_val[k], 2),
            "卖出s": round(s_val[k], 2),
            "保留现金c": round(c_val, 2) if k == "A" else "",
            "总资产W": round(W, 2),
        })

    h_new = {}
    for k in STOCKS:
        r_real = real_return[k][y]
        h_new[k] = x_val[k] * (1 + r_real)

    c_new = c_val * (1 + RF)
    W_new = sum(h_new[k] for k in STOCKS) + c_new

    print(f"\n  【年末结算】实际收益率:")
    for k in STOCKS:
        print(f"    {STOCK_NAMES[k]}: 实际年化={real_return[k][y]*100:.2f}%，年末持仓={h_new[k]:,.2f} 元")
    print(f"  年末现金: {c_new:,.2f} 元")
    print(f"  ➜ 下一年初总资产 W = {W_new:,.2f} 元（变动 {(W_new/W-1)*100:+.2f}%）")

    # 更新状态
    h_old = h_new
    c_old = c_new
    W     = W_new



result_df = pd.DataFrame(results)
csv_out = os.path.join(OUTPUT_DIR, "optimization_results.csv")
result_df.to_csv(csv_out, index=False, encoding="utf-8-sig")
print(f"\n✅ 优化结果已保存: {csv_out}")

print("\n" + "=" * 60)
print("五年投资方案汇总")
print("=" * 60)
print(result_df[["年份","股票名称","是否允许投资δ","目标持仓x","买入b","卖出s","保留现金c","总资产W"]].to_string(index=False))



# 提取每年各股票持仓金额
hold_df = result_df.pivot(index="年份", columns="股票类别", values="目标持仓x").fillna(0)
cash_by_year = (
    result_df[result_df["保留现金c"] != ""]
    .set_index("年份")["保留现金c"]
    .astype(float)
)
W_by_year = (
    result_df.groupby("年份")["总资产W"].first()
)

# ── 图A：各年持仓结构堆叠柱状图 ──
fig, ax = plt.subplots(figsize=(10, 6))
bottom = np.zeros(len(YEARS))
year_labels = [str(y) for y in YEARS]

for k in STOCKS:
    vals = [hold_df.loc[y, k] if y in hold_df.index else 0 for y in YEARS]
    ax.bar(year_labels, vals, bottom=bottom, color=COLORS[k],
           label=STOCK_NAMES[k], width=0.5, edgecolor="white")
    bottom += np.array(vals)

# 现金部分
cash_vals = [cash_by_year.get(y, 0) for y in YEARS]
ax.bar(year_labels, cash_vals, bottom=bottom, color="#AAAAAA",
       label="保留现金", width=0.5, edgecolor="white")

ax.set_title("五年各年度持仓结构（堆叠柱状图）", fontsize=14)
ax.set_xlabel("年份")
ax.set_ylabel("金额（元）")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e4:.0f}万"))
ax.legend(prop=CN_FONT, loc="upper left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
beautify(CN_FONT, EN_FONT)
fname = os.path.join(OUTPUT_DIR, "figA_portfolio_stack.png")
plt.savefig(fname, dpi=300, bbox_inches="tight")
plt.close()
print(f"\n✅ {fname}")

# ── 图B：各年持仓比例饼图（每年一张）──
for yi, y in enumerate(YEARS):
    sub = result_df[result_df["年份"] == y]
    labels, sizes, colors = [], [], []
    for k in STOCKS:
        v = float(sub[sub["股票类别"] == k]["目标持仓x"].values[0])
        if v > 1:
            labels.append(STOCK_NAMES[k])
            sizes.append(v)
            colors.append(COLORS[k])
    cv = cash_by_year.get(y, 0)
    if cv > 1:
        labels.append("保留现金")
        sizes.append(cv)
        colors.append("#AAAAAA")

    if not sizes:
        continue

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=140,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
        textprops={"fontsize": 11}
    )
    for txt in texts:
        txt.set_fontproperties(CN_FONT)
    for at in autotexts:
        at.set_fontproperties(EN_FONT)
        at.set_fontweight("bold")

    W_val = W_by_year.get(y, 0)
    ax.set_title(f"{y} 年持仓配置比例\n总资产：{W_val/1e4:.2f} 万元", fontsize=13)
    beautify(CN_FONT, EN_FONT)
    fname = os.path.join(OUTPUT_DIR, f"figB_{y}_pie.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ {fname}")

# ── 图C：总资产增长曲线 ──
fig, ax = plt.subplots(figsize=(9, 5))
w_vals = [W_by_year.get(y, np.nan) for y in YEARS]
ax.plot(year_labels, w_vals, color="#2E86AB", marker="o", linewidth=2.5,
        markersize=8, markerfacecolor="white", markeredgewidth=2.5)
for i, (yl, v) in enumerate(zip(year_labels, w_vals)):
    ax.annotate(f"{v/1e4:.1f}万", (yl, v),
                textcoords="offset points", xytext=(0, 12),
                ha="center", fontproperties=CN_FONT, fontsize=11)
ax.set_title("五年总资产增长曲线", fontsize=14)
ax.set_xlabel("年份")
ax.set_ylabel("总资产（元）")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e4:.0f}万"))
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
beautify(CN_FONT, EN_FONT)
fname = os.path.join(OUTPUT_DIR, "figC_wealth_curve.png")
plt.savefig(fname, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ {fname}")

# ── 图D：各股票每年买入/卖出金额（每只股票一张）──
for k in STOCKS:
    sub = result_df[result_df["股票类别"] == k].set_index("年份")
    buys  = [sub.loc[y, "买入b"]  if y in sub.index else 0 for y in YEARS]
    sells = [sub.loc[y, "卖出s"]  if y in sub.index else 0 for y in YEARS]

    x_pos = np.arange(len(YEARS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x_pos - width/2, buys,  width, label="买入", color=COLORS[k], alpha=0.9, edgecolor="white")
    ax.bar(x_pos + width/2, sells, width, label="卖出", color=COLORS[k], alpha=0.45, edgecolor="white")

    for i, (b_v, s_v) in enumerate(zip(buys, sells)):
        if b_v > 0:
            ax.text(i - width/2, b_v + W0*0.003, f"{b_v/1e4:.1f}万",
                    ha="center", fontsize=9, fontproperties=CN_FONT)
        if s_v > 0:
            ax.text(i + width/2, s_v + W0*0.003, f"{s_v/1e4:.1f}万",
                    ha="center", fontsize=9, fontproperties=CN_FONT)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(year_labels)
    ax.set_title(f"{STOCK_NAMES[k]} 各年买入/卖出金额", fontsize=13)
    ax.set_ylabel("金额（元）")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e4:.0f}万"))
    ax.legend(prop=CN_FONT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    beautify(CN_FONT, EN_FONT)
    fname = os.path.join(OUTPUT_DIR, f"figD_{k}_trade.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ {fname}")

# ── 图E：风险调整收益雷达图（每年一张）──
from matplotlib.patches import FancyArrowPatch

for yi, y in enumerate(YEARS):
    sub = result_df[result_df["年份"] == y]
    rac = []
    for k in STOCKS:
        v = float(sub[sub["股票类别"] == k]["风险调整收益"].values[0])
        rac.append(v)

    categories = [STOCK_NAMES[k] for k in STOCKS]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    rac_plot = rac + rac[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    for label in ax.get_xticklabels():
        label.set_fontproperties(CN_FONT)
        label.set_fontsize(11)

    ax.plot(angles, rac_plot, color="#2E86AB", linewidth=2)
    ax.fill(angles, rac_plot, color="#2E86AB", alpha=0.25)
    ax.set_title(f"{y} 年各股票风险调整收益雷达图", fontsize=13, pad=20)
    beautify(CN_FONT, EN_FONT)
    fname = os.path.join(OUTPUT_DIR, f"figE_{y}_radar.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ {fname}")

print("\n" + "=" * 60)
print("全部完成！")
final_W = W
total_return = (final_W / W0 - 1) * 100
print(f"初始资产: {W0/1e4:.2f} 万元")
print(f"五年后资产: {final_W/1e4:.2f} 万元")
print(f"五年累计收益率: {total_return:+.2f}%")
print("=" * 60)
