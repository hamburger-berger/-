import os
import re
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import font_manager as fm
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

#

CN_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"
EN_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"

if not os.path.exists(CN_FONT_PATH):
    raise FileNotFoundError(f"中文字体文件不存在: {CN_FONT_PATH}")
if not os.path.exists(EN_FONT_PATH):
    raise FileNotFoundError(f"英文字体文件不存在: {EN_FONT_PATH}")

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
    "A": "#2E86AB",   # 蓝  贵州茅台
    "B": "#E84855",   # 红  牧原股份
    "C": "#3BB273",   # 绿  比亚迪
    "D": "#F4A261",   # 橙  中远海特
}

STOCK_NAMES = {
    "A": "贵州茅台（A类）",
    "B": "牧原股份（B类）",
    "C": "比亚迪（C类）",
    "D": "中远海特（D类）",
}

# 投资窗口 delta_{k,y}  y=1..5 对应 2021..2025
WINDOWS = {
    "A": [1, 1, 1, 1, 1],
    "B": [1, 1, 0, 0, 0],
    "C": [0, 1, 1, 1, 1],
    "D": [0, 0, 1, 1, 1],
}

OUTPUT_DIR = "/root/autodl-tmp/yunchou/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)



FILES = {
    "A": "/root/autodl-tmp/yunchou/maotai.csv",
    "B": "/root/autodl-tmp/yunchou/muyuangufen.csv",
    "C": "/root/autodl-tmp/yunchou/biyadi.csv",
    "D": "/root/autodl-tmp/yunchou/zhongyuanhaite.csv",
}

dfs = {}
for k, path in FILES.items():
    df = pd.read_csv(path, parse_dates=["日期"])
    df = df.sort_values("日期").reset_index(drop=True)
    dfs[k] = df

print("数据读取完毕：")
for k, df in dfs.items():
    print(f"  {STOCK_NAMES[k]}: {df['日期'].min().date()} ~ {df['日期'].max().date()}，共 {len(df)} 条")



YEARS = [2021, 2022, 2023, 2024, 2025]

def calc_daily_return(prices):
    """用收盘价计算日收益率"""
    return prices.pct_change().fillna(0)

def calc_max_drawdown(prices):
    """计算最大回撤"""
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    return float(-drawdown.min())

# 存储年度参数
params = {}

for k, df in dfs.items():
    df["日收益率"] = calc_daily_return(df["收盘"])
    params[k] = {}
    for y in YEARS:
        # 用上一年数据估参数
        if y == 2021:
            window = df[df["日期"].dt.year == 2021]
        else:
            window = df[df["日期"].dt.year == y - 1]

        r = window["日收益率"]
        prices = window["收盘"]

        annual_return = float(252 * r.mean())
        volatility = float(np.sqrt(252) * r.std())
        mdd = calc_max_drawdown(prices)
        rho = 0.5 * volatility + 0.5 * mdd

        params[k][y] = {
            "r":   annual_return,
            "vol": volatility,
            "mdd": mdd,
            "rho": rho,
        }


param_df = {}
for metric in ["r", "vol", "mdd", "rho"]:
    param_df[metric] = pd.DataFrame(
        {k: [params[k][y][metric] for y in YEARS] for k in "ABCD"},
        index=YEARS
    )

print("\n年度预测收益率矩阵（%）：")
print((param_df["r"] * 100).round(2))
print("\n综合风险矩阵：")
print(param_df["rho"].round(4))



for k, df in dfs.items():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["日期"], df["收盘"], color=COLORS[k], linewidth=1.5)
    ax.set_title(f"{STOCK_NAMES[k]} 收盘价走势（2021-2025）", fontsize=15)
    ax.set_xlabel("日期")
    ax.set_ylabel("收盘价（元）")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(8))
    plt.xticks(rotation=30)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    beautify(CN_FONT, EN_FONT)
    fname = os.path.join(OUTPUT_DIR, f"fig1_{k}_price.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ {fname}")



fig, ax = plt.subplots(figsize=(8, 4))
data = param_df["r"] * 100  # 转成百分比
cmap = LinearSegmentedColormap.from_list("rg", ["#E84855", "#ffffff", "#2E86AB"])
im = ax.imshow(data.T.values, cmap=cmap, aspect="auto", vmin=-data.abs().max().max(), vmax=data.abs().max().max())
ax.set_xticks(range(len(YEARS)))
ax.set_xticklabels([str(y) for y in YEARS])
ax.set_yticks(range(4))
ax.set_yticklabels([STOCK_NAMES[k] for k in "ABCD"])
ax.set_title("各股票年度预测收益率热力图（%）", fontsize=14)
for i in range(len(YEARS)):
    for j, k in enumerate("ABCD"):
        val = data.loc[YEARS[i], k]
        ax.text(i, j, f"{val:.1f}%", ha="center", va="center", fontsize=11,
                color="white" if abs(val) > data.abs().max().max() * 0.5 else "black")
plt.colorbar(im, ax=ax, label="年化收益率（%）")
beautify(CN_FONT, EN_FONT)
fname = os.path.join(OUTPUT_DIR, "fig2_return_heatmap.png")
plt.savefig(fname, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ {fname}")



fig, ax = plt.subplots(figsize=(8, 4))
data = param_df["rho"]
cmap2 = LinearSegmentedColormap.from_list("wr", ["#ffffff", "#F4A261", "#E84855"])
im = ax.imshow(data.T.values, cmap=cmap2, aspect="auto")
ax.set_xticks(range(len(YEARS)))
ax.set_xticklabels([str(y) for y in YEARS])
ax.set_yticks(range(4))
ax.set_yticklabels([STOCK_NAMES[k] for k in "ABCD"])
ax.set_title("各股票年度综合风险热力图（波动率+最大回撤）", fontsize=14)
for i in range(len(YEARS)):
    for j, k in enumerate("ABCD"):
        val = data.loc[YEARS[i], k]
        ax.text(i, j, f"{val:.3f}", ha="center", va="center", fontsize=11,
                color="white" if val > data.values.max() * 0.65 else "black")
plt.colorbar(im, ax=ax, label="综合风险值")
beautify(CN_FONT, EN_FONT)
fname = os.path.join(OUTPUT_DIR, "fig3_risk_heatmap.png")
plt.savefig(fname, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ {fname}")



for k, df in dfs.items():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["日期"], df["收盘"], color=COLORS[k], linewidth=1.2, label="收盘价", zorder=2)

    for idx, y in enumerate(YEARS):
        if WINDOWS[k][idx] == 1:
            start = pd.Timestamp(f"{y}-01-01")
            end   = pd.Timestamp(f"{y}-12-31")
            ax.axvspan(start, end, alpha=0.18, color=COLORS[k], zorder=1)

    # 图例
    window_patch = Patch(color=COLORS[k], alpha=0.4, label="可投资窗口")
    ax.legend(handles=[
        plt.Line2D([0], [0], color=COLORS[k], linewidth=2, label="收盘价"),
        window_patch
    ], prop=CN_FONT, loc="upper right")

    ax.set_title(f"{STOCK_NAMES[k]} 投资时间窗口标注", fontsize=14)
    ax.set_xlabel("日期")
    ax.set_ylabel("收盘价（元）")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(8))
    plt.xticks(rotation=30)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    beautify(CN_FONT, EN_FONT)
    fname = os.path.join(OUTPUT_DIR, f"fig4_{k}_window.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ {fname}")



for y in YEARS:
    fig, ax = plt.subplots(figsize=(7, 6))
    for k in "ABCD":
        r   = params[k][y]["r"] * 100
        rho = params[k][y]["rho"]
        delta = WINDOWS[k][YEARS.index(y)]
        marker = "o" if delta == 1 else "X"
        size   = 180 if delta == 1 else 120
        ax.scatter(rho, r, color=COLORS[k], s=size, marker=marker,
                   zorder=3, edgecolors="white", linewidths=1.2)
        ax.annotate(STOCK_NAMES[k], (rho, r),
                    textcoords="offset points", xytext=(8, 4),
                    fontproperties=CN_FONT, fontsize=10)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("综合风险值 ρ")
    ax.set_ylabel("年化预测收益率（%）")
    ax.set_title(f"{y} 年各股票风险-收益分布图", fontsize=14)

    legend_handles = [
        plt.scatter([], [], marker="o", color="gray", s=120, label="窗口内（可投资）"),
        plt.scatter([], [], marker="X", color="gray", s=100, label="窗口外（禁止投资）"),
    ]
    ax.legend(handles=legend_handles, prop=CN_FONT)
    ax.grid(linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    beautify(CN_FONT, EN_FONT)
    fname = os.path.join(OUTPUT_DIR, f"fig5_{y}_risk_return.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ {fname}")



for y in YEARS:
    ret_dict = {}
    for k, df in dfs.items():
        window = df[df["日期"].dt.year == y][["日期", "日收益率"]].set_index("日期")
        ret_dict[STOCK_NAMES[k]] = window["日收益率"]

    corr_df = pd.DataFrame(ret_dict).corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap3 = LinearSegmentedColormap.from_list("corr", ["#E84855", "#ffffff", "#2E86AB"])
    im = ax.imshow(corr_df.values, cmap=cmap3, vmin=-1, vmax=1, aspect="auto")
    labels = list(corr_df.columns)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticklabels(labels)
    for i in range(4):
        for j in range(4):
            val = corr_df.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=11,
                    color="white" if abs(val) > 0.5 else "black")
    plt.colorbar(im, ax=ax, label="相关系数")
    ax.set_title(f"{y} 年日收益率相关性热力图", fontsize=14)
    beautify(CN_FONT, EN_FONT)
    fname = os.path.join(OUTPUT_DIR, f"fig6_{y}_corr.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ {fname}")



for k in "ABCD":
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    year_labels = [str(y) for y in YEARS]
    vols = [params[k][y]["vol"] for y in YEARS]
    mdds = [params[k][y]["mdd"] for y in YEARS]

    # 波动率
    bars1 = axes[0].bar(year_labels, vols, color=COLORS[k], width=0.5, edgecolor="white")
    axes[0].set_title(f"{STOCK_NAMES[k]}\n年化波动率", fontsize=13)
    axes[0].set_ylabel("年化波动率")
    axes[0].set_ylim(0, max(vols) * 1.3)
    for bar, v in zip(bars1, vols):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # 最大回撤
    bars2 = axes[1].bar(year_labels, mdds, color=COLORS[k], width=0.5, alpha=0.75, edgecolor="white")
    axes[1].set_title(f"{STOCK_NAMES[k]}\n最大回撤", fontsize=13)
    axes[1].set_ylabel("最大回撤")
    axes[1].set_ylim(0, max(mdds) * 1.3)
    for bar, v in zip(bars2, mdds):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.suptitle(f"{STOCK_NAMES[k]} 年度风险指标", fontsize=14, y=1.01)
    plt.tight_layout()
    beautify(CN_FONT, EN_FONT)
    fname = os.path.join(OUTPUT_DIR, f"fig7_{k}_risk_bar.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ {fname}")



rows = []
for k in "ABCD":
    for y in YEARS:
        rows.append({
            "股票类别": k,
            "股票名称": STOCK_NAMES[k],
            "年份": y,
            "年化预测收益率": round(params[k][y]["r"], 6),
            "年化波动率": round(params[k][y]["vol"], 6),
            "最大回撤": round(params[k][y]["mdd"], 6),
            "综合风险rho": round(params[k][y]["rho"], 6),
            "投资窗口delta": WINDOWS[k][YEARS.index(y)],
        })

param_table = pd.DataFrame(rows)
csv_path = os.path.join(OUTPUT_DIR, "annual_params.csv")
param_table.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"\n✅ 年度参数表已保存: {csv_path}")
print(param_table.to_string(index=False))

print("\n全部完成！")