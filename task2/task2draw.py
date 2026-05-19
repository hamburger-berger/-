import os
import re
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap


# ============================================================
# 0. 路径设置
# ============================================================
CLUSTER_DIR = "cluster_results"
SOLVE_DIR = "solve_results"
OUTPUT_DIR = "conference_figures_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. 股票中文名
# ============================================================
STOCK_NAME_MAP = {
    "601398": "工商银行",
    "000333": "美的集团",
    "000895": "双汇发展",
    "601899": "紫金矿业",
}


# ============================================================
# 2. 字体设置
# ============================================================
def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def find_font_by_pattern(patterns):
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        if len(files) > 0:
            return files[0]
    return None


CN_FONT_PATH = first_existing([
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/System/Library/Fonts/Songti.ttc",
    "C:/Windows/Fonts/simsun.ttc",
])

if CN_FONT_PATH is None:
    CN_FONT_PATH = find_font_by_pattern([
        "/usr/share/fonts/**/*CJK*.ttc",
        "/usr/share/fonts/**/*CJK*.otf",
        "/usr/share/fonts/**/*Noto*.ttc",
        "/usr/share/fonts/**/*Noto*.otf",
    ])

EN_FONT_PATH = first_existing([
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "C:/Windows/Fonts/times.ttf",
])

if EN_FONT_PATH is None:
    EN_FONT_PATH = find_font_by_pattern([
        "/usr/share/fonts/**/*DejaVuSerif*.ttf",
        "/usr/share/fonts/**/*LiberationSerif*.ttf",
    ])

if CN_FONT_PATH is None:
    raise FileNotFoundError("没有找到中文字体，请安装 fonts-noto-cjk")

if EN_FONT_PATH is None:
    raise FileNotFoundError("没有找到英文字体")

print("中文字体路径:", CN_FONT_PATH)
print("英文字体路径:", EN_FONT_PATH)

fm.fontManager.addfont(CN_FONT_PATH)
fm.fontManager.addfont(EN_FONT_PATH)

CN_FONT = fm.FontProperties(fname=CN_FONT_PATH, weight="bold")
EN_FONT = fm.FontProperties(fname=EN_FONT_PATH, weight="bold")

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = EN_FONT.get_name()
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"


def contains_chinese(text):
    return bool(re.search(r"[\u4e00-\u9fff]", str(text))) if text else False


def beautify_text(fig):
    for obj in fig.findobj(match=lambda x: hasattr(x, "get_text")):
        try:
            txt = obj.get_text()
            if txt is None or txt == "":
                continue

            if contains_chinese(txt):
                obj.set_fontproperties(CN_FONT)
                obj.set_fontweight("bold")
            else:
                obj.set_fontproperties(EN_FONT)
                obj.set_fontweight("bold")

        except Exception:
            pass


# ============================================================
# 3. 配色
# ============================================================
BG = "#FAF8F2"
PANEL = "#F6F1E8"
GRID = "#D8D2C7"
TEXT = "#2F3437"
SUBTEXT = "#6B6F72"

PALETTE = {
    "blue": "#6F88A6",
    "cyan": "#8EB6B0",
    "green": "#AFC8A7",
    "sage": "#C8D7BD",
    "sand": "#D8B89D",
    "rose": "#C9A6A0",
    "purple": "#B0A3BD",
    "gray": "#BFC2BD",
    "dark": "#4D5F68",
}

STOCK_PALETTE = [
    "#6F88A6",
    "#8EB6B0",
    "#AFC8A7",
    "#D8B89D",
]

CORR_CMAP = LinearSegmentedColormap.from_list(
    "muted_corr",
    [
        "#F7F3EA",
        "#E4EADF",
        "#C9DCCB",
        "#9DBEC0",
        "#7893AA",
    ]
)

FRONTIER_COLOR = "#6F88A6"


def style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.grid(True, color=GRID, linestyle="--", linewidth=0.8, alpha=0.55)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#C9C4B8")
        ax.spines[spine].set_linewidth(0.9)

    ax.tick_params(axis="both", colors=TEXT, labelsize=10)


def save_png(fig, filename):
    beautify_text(fig)

    png_path = os.path.join(OUTPUT_DIR, filename + ".png")

    fig.savefig(
        png_path,
        dpi=320,
        bbox_inches="tight",
        facecolor=BG
    )

    plt.close(fig)
    print(f"✅ {png_path}")


# ============================================================
# 4. 读取数据
# ============================================================
selected_corr = pd.read_csv(
    os.path.join(CLUSTER_DIR, "selected_4_corr_matrix.csv"),
    index_col=0
)

selected_corr.index = selected_corr.index.astype(str).str.zfill(6)
selected_corr.columns = selected_corr.columns.astype(str).str.zfill(6)
selected_corr = selected_corr.astype(float)

candidate_corr = pd.read_csv(
    os.path.join(CLUSTER_DIR, "candidate_corr_matrix.csv"),
    index_col=0
)

candidate_corr.index = candidate_corr.index.astype(str).str.zfill(6)
candidate_corr.columns = candidate_corr.columns.astype(str).str.zfill(6)
candidate_corr = candidate_corr.astype(float)

cluster_metrics = pd.read_csv(
    os.path.join(CLUSTER_DIR, "cluster_metrics.csv")
)

frontier_df = pd.read_csv(
    os.path.join(SOLVE_DIR, "efficient_frontier.csv")
)

portfolio_df = pd.read_csv(
    os.path.join(SOLVE_DIR, "portfolio_comparison.csv")
)

weight_cols = [c for c in portfolio_df.columns if c.startswith("w_")]
stock_codes = [c.replace("w_", "") for c in weight_cols]
stock_labels = [STOCK_NAME_MAP.get(code, code) for code in stock_codes]


# ============================================================
# 图1：最终4只股票相关性热力图
# ============================================================
fig, ax = plt.subplots(figsize=(6.8, 5.9), facecolor=BG)
ax.set_facecolor(PANEL)

mat = selected_corr.values
im = ax.imshow(mat, cmap=CORR_CMAP, vmin=0, vmax=1)

x_labels = [STOCK_NAME_MAP.get(c, c) for c in selected_corr.columns]
y_labels = [STOCK_NAME_MAP.get(c, c) for c in selected_corr.index]

ax.set_xticks(np.arange(len(x_labels)))
ax.set_yticks(np.arange(len(y_labels)))

ax.set_xticklabels(x_labels, rotation=0, fontsize=11, fontproperties=CN_FONT)
ax.set_yticklabels(y_labels, fontsize=11, fontproperties=CN_FONT)

for i in range(len(selected_corr.index) + 1):
    ax.axhline(i - 0.5, color="#FFFFFF", linewidth=1.2)
    ax.axvline(i - 0.5, color="#FFFFFF", linewidth=1.2)

for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        val = mat[i, j]
        txt_color = "#FFFFFF" if val > 0.72 else TEXT
        ax.text(
            j,
            i,
            f"{val:.2f}",
            ha="center",
            va="center",
            color=txt_color,
            fontsize=12,
            fontproperties=EN_FONT
        )

ax.set_title(
    "最终入选四只股票相关性矩阵",
    fontsize=17,
    color=TEXT,
    pad=14,
    fontproperties=CN_FONT
)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.outline.set_visible(False)
cbar.ax.tick_params(labelsize=9, colors=SUBTEXT)
cbar.set_label("相关系数", fontsize=10, color=TEXT, labelpad=8)
cbar.ax.yaxis.label.set_fontproperties(CN_FONT)

save_png(fig, "selected_4_corr_heatmap")


# ============================================================
# 图2：有效前沿图
# ============================================================
fig, ax = plt.subplots(figsize=(8.5, 6.3), facecolor=BG)
style_ax(ax)

frontier_plot = (
    frontier_df
    .sort_values(["annual_volatility", "annual_return"])
    .drop_duplicates(subset=["annual_volatility", "annual_return"])
)

ax.plot(
    frontier_plot["annual_volatility"],
    frontier_plot["annual_return"],
    color=FRONTIER_COLOR,
    linewidth=2.8,
    alpha=0.95,
    label="Efficient Frontier"
)

scheme_color = {
    "等权重组合": PALETTE["gray"],
    "最小风险组合": PALETTE["blue"],
    "目标收益约束最小风险组合": PALETTE["cyan"],
    "最优切点组合_最大夏普": PALETTE["sand"],
}

scheme_short = {
    "等权重组合": "等权重",
    "最小风险组合": "最小风险 / 目标收益约束",
    "目标收益约束最小风险组合": "最小风险 / 目标收益约束",
    "最优切点组合_最大夏普": "最大夏普",
}

# 合并完全重合的点，避免“最小风险”和“目标收益约束”标签重叠
plot_points = []

used_duplicate = False
for _, row in portfolio_df.iterrows():
    name = row["组合方案"]

    if pd.isna(row["annual_return"]) or pd.isna(row["annual_volatility"]):
        continue

    if name == "目标收益约束最小风险组合":
        min_row = portfolio_df[portfolio_df["组合方案"] == "最小风险组合"]
        if len(min_row) > 0:
            same_x = abs(float(row["annual_volatility"]) - float(min_row.iloc[0]["annual_volatility"])) < 1e-8
            same_y = abs(float(row["annual_return"]) - float(min_row.iloc[0]["annual_return"])) < 1e-8

            if same_x and same_y:
                used_duplicate = True
                continue

    plot_points.append(row)

# 点
for _, row in pd.DataFrame(plot_points).iterrows():
    name = row["组合方案"]
    color = scheme_color.get(name, PALETTE["purple"])

    ax.scatter(
        row["annual_volatility"],
        row["annual_return"],
        s=105,
        color=color,
        edgecolor="#FFFFFF",
        linewidth=1.2,
        zorder=5
    )

# 标签位置：避免挤在坐标轴附近
offsets = {
    "等权重组合": (16, 14),
    "最小风险组合": (18, -30),
    "目标收益约束最小风险组合": (18, -30),
    "最优切点组合_最大夏普": (14, 16),
}

for _, row in pd.DataFrame(plot_points).iterrows():
    name = row["组合方案"]

    label = scheme_short.get(name, name)
    dx, dy = offsets.get(name, (8, 8))

    ax.annotate(
        label,
        xy=(row["annual_volatility"], row["annual_return"]),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=9.6,
        color=TEXT,
        fontproperties=CN_FONT,
        bbox=dict(
            boxstyle="round,pad=0.24",
            fc="#FFFFFF",
            ec="#DDD6C8",
            lw=0.8,
            alpha=0.90
        ),
        arrowprops=dict(
            arrowstyle="-",
            color="#BEB7AA",
            lw=0.8,
            alpha=0.75
        )
    )

# 坐标范围加 margin，避免标签压到边框
x_min = min(frontier_plot["annual_volatility"].min(), portfolio_df["annual_volatility"].min())
x_max = max(frontier_plot["annual_volatility"].max(), portfolio_df["annual_volatility"].max())
y_min = min(frontier_plot["annual_return"].min(), portfolio_df["annual_return"].min())
y_max = max(frontier_plot["annual_return"].max(), portfolio_df["annual_return"].max())

ax.set_xlim(x_min - 0.012, x_max + 0.018)
ax.set_ylim(y_min - 0.018, y_max + 0.018)

ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

ax.set_xlabel("年化波动率", fontsize=12, color=TEXT, labelpad=8, fontproperties=CN_FONT)
ax.set_ylabel("年化收益率", fontsize=12, color=TEXT, labelpad=8, fontproperties=CN_FONT)
ax.set_title("有效前沿与组合方案对比", fontsize=17, color=TEXT, pad=14, fontproperties=CN_FONT)

ax.legend(
    loc="lower right",
    frameon=True,
    framealpha=0.88,
    facecolor="#FFFFFF",
    edgecolor="#DDD6C8",
    fontsize=10,
    prop=EN_FONT
)

save_png(fig, "efficient_frontier")


# ============================================================
# 图3：组合权重横向堆叠柱状图
# ============================================================
plot_df = portfolio_df[["组合方案"] + weight_cols].copy()
plot_df = plot_df.dropna(subset=weight_cols)

scheme_order = [
    "等权重组合",
    "最小风险组合",
    "目标收益约束最小风险组合",
    "最优切点组合_最大夏普",
]

plot_df["组合方案"] = pd.Categorical(
    plot_df["组合方案"],
    categories=scheme_order,
    ordered=True
)

plot_df = plot_df.sort_values("组合方案").reset_index(drop=True)

# 图加高，给底部 legend 留空间
fig, ax = plt.subplots(figsize=(10.2, 6.8), facecolor=BG)
style_ax(ax)

y_pos = np.arange(len(plot_df))
lefts = np.zeros(len(plot_df))
bar_height = 0.58

for idx, col in enumerate(weight_cols):
    code = col.replace("w_", "")
    label = STOCK_NAME_MAP.get(code, code)
    vals = plot_df[col].values

    ax.barh(
        y_pos,
        vals,
        left=lefts,
        height=bar_height,
        color=STOCK_PALETTE[idx],
        edgecolor="#FFFFFF",
        linewidth=1.1,
        label=label
    )

    for i, (v, l) in enumerate(zip(vals, lefts)):
        if v >= 0.07:
            ax.text(
                l + v / 2,
                i,
                f"{v:.1%}",
                ha="center",
                va="center",
                fontsize=11,
                color="#FFFFFF" if idx in [0, 1] else TEXT,
                fontproperties=EN_FONT
            )

    lefts += vals

label_map = {
    "等权重组合": "等权重组合",
    "最小风险组合": "最小风险组合",
    "目标收益约束最小风险组合": "目标收益约束",
    "最优切点组合_最大夏普": "最大夏普组合",
}

ax.set_yticks(y_pos)
ax.set_yticklabels(
    [label_map.get(str(x), str(x)) for x in plot_df["组合方案"]],
    fontsize=13,
    fontproperties=CN_FONT
)

ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

ax.tick_params(axis="x", labelsize=12)

ax.set_xlabel(
    "投资权重",
    fontsize=14,
    color=TEXT,
    labelpad=14,
    fontproperties=CN_FONT
)

ax.set_title(
    "不同组合方案的持仓权重结构",
    fontsize=22,
    color=TEXT,
    pad=22,
    fontproperties=CN_FONT
)

# legend 放到更下方，避免挡住 x 轴标题
legend = ax.legend(
    title="入选股票",
    prop=CN_FONT,
    title_fontproperties=CN_FONT,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.34),
    ncol=4,
    frameon=True,
    framealpha=0.92,
    facecolor="#FFFFFF",
    edgecolor="#DDD6C8",
    fontsize=12,
    columnspacing=1.8,
    handlelength=1.4,
    borderpad=0.75
)

# 关键：手动留出底部空间
fig.subplots_adjust(
    left=0.17,
    right=0.98,
    top=0.86,
    bottom=0.30
)

save_png(fig, "portfolio_weights")

# ============================================================
# 图4：聚类评价指标图
# ============================================================
fig, ax1 = plt.subplots(figsize=(8.4, 5.8), facecolor=BG)
style_ax(ax1)

ax1.plot(
    cluster_metrics["k"],
    cluster_metrics["silhouette"],
    marker="o",
    markersize=7,
    linewidth=2.4,
    color=PALETTE["blue"],
    markeredgecolor="#FFFFFF",
    markeredgewidth=1.1,
    label="Silhouette"
)

ax1.set_xlabel("聚类数 k", fontsize=12, color=TEXT, labelpad=8, fontproperties=CN_FONT)
ax1.set_ylabel("轮廓系数", fontsize=12, color=TEXT, labelpad=8, fontproperties=CN_FONT)
ax1.tick_params(axis="y", colors=PALETTE["blue"])

ax2 = ax1.twinx()

ax2.plot(
    cluster_metrics["k"],
    cluster_metrics["CH"],
    marker="s",
    markersize=7,
    linewidth=2.4,
    linestyle="--",
    color=PALETTE["sand"],
    markeredgecolor="#FFFFFF",
    markeredgewidth=1.1,
    label="CH Index"
)

ax2.set_ylabel("CH 指标", fontsize=12, color=TEXT, labelpad=8, fontproperties=CN_FONT)
ax2.tick_params(axis="y", colors=PALETTE["sand"])
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_color("#C9C4B8")

best_k = int(cluster_metrics.loc[cluster_metrics["CH"].idxmax(), "k"])

ax1.axvline(
    best_k,
    color=PALETTE["gray"],
    linestyle=":",
    linewidth=1.6,
    alpha=0.9
)

ax1.text(
    best_k + 0.08,
    ax1.get_ylim()[1] * 0.96,
    f"k = {best_k}",
    fontsize=10.5,
    color=TEXT,
    fontproperties=EN_FONT,
    bbox=dict(
        boxstyle="round,pad=0.25",
        fc="#FFFFFF",
        ec="#DDD6C8",
        alpha=0.88
    )
)

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines_1 + lines_2,
    labels_1 + labels_2,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.13),
    ncol=2,
    frameon=True,
    framealpha=0.9,
    facecolor="#FFFFFF",
    edgecolor="#DDD6C8",
    prop=EN_FONT
)

ax1.set_title("聚类数选择：轮廓系数与 CH 指标", fontsize=17, color=TEXT, pad=14, fontproperties=CN_FONT)

save_png(fig, "cluster_metrics")


# ============================================================
# 图5：50只股票相关性热力图
# ============================================================
fig, ax = plt.subplots(figsize=(11.5, 9.5), facecolor=BG)
ax.set_facecolor(PANEL)

im = ax.imshow(
    candidate_corr.values,
    cmap=CORR_CMAP,
    vmin=-0.2,
    vmax=1.0,
    aspect="auto"
)

ax.set_xticks(np.arange(len(candidate_corr.columns)))
ax.set_yticks(np.arange(len(candidate_corr.index)))

ax.set_xticklabels(candidate_corr.columns, rotation=90, fontsize=6.5, fontproperties=EN_FONT)
ax.set_yticklabels(candidate_corr.index, fontsize=6.5, fontproperties=EN_FONT)

ax.tick_params(axis="both", length=0, colors=TEXT)
ax.set_title("50只候选股票收益率相关性热力图", fontsize=17, color=TEXT, pad=14, fontproperties=CN_FONT)

cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
cbar.outline.set_visible(False)
cbar.ax.tick_params(labelsize=9, colors=SUBTEXT)
cbar.set_label("相关系数", fontsize=10, color=TEXT, labelpad=8)
cbar.ax.yaxis.label.set_fontproperties(CN_FONT)

for spine in ax.spines.values():
    spine.set_visible(False)

save_png(fig, "candidate_50_corr_heatmap")


print("\n全部重画完成！输出目录：conference_figures_final")