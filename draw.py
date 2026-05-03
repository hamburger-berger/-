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


# A 茅台  B 牧原  C 比亚迪  D 中远海特  现金
PALETTE = {
    "A":    "#3D6B8E",   # 深蓝紫（对应 Exon）
    "B":    "#6BBFCF",   # 浅蓝（对应 Intron）
    "C":    "#A8CDAB",   # 浅绿（对应 Downstream）
    "D":    "#3DA876",   # 深绿（对应 Upstream）
    "cash": "#D6D6D6",   # 浅灰（现金）
}

STOCK_NAMES = {
    "A": "贵州茅台",
    "B": "牧原股份",
    "C": "比亚迪",
    "D": "中远海特",
}

STOCKS = ["A", "B", "C", "D"]
YEARS  = [2021, 2022, 2023, 2024, 2025]

OUTPUT_DIR = "/root/autodl-tmp/yunchou/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


result_path = os.path.join(OUTPUT_DIR, "optimization_results.csv")
result_df = pd.read_csv(result_path)

# 持仓金额透视
hold_df = result_df.pivot(index="年份", columns="股票类别", values="目标持仓x").fillna(0)

cash_by_year = {}
for y in YEARS:
    sub = result_df[(result_df["年份"] == y) & (result_df["保留现金c"] != "")]
    if len(sub) > 0:
        cash_by_year[y] = float(sub["保留现金c"].values[0])
    else:
        cash_by_year[y] = 0.0

W_by_year = result_df.groupby("年份")["总资产W"].first()

# 风险调整收益
rac_df = result_df.pivot(index="年份", columns="股票类别", values="风险调整收益").fillna(0)


fig, ax = plt.subplots(figsize=(11, 6))

year_labels = [str(y) for y in YEARS]
y_pos = np.arange(len(YEARS))
bar_height = 0.55

lefts = np.zeros(len(YEARS))

for k in STOCKS:
    vals = np.array([hold_df.loc[y, k] if y in hold_df.index else 0 for y in YEARS])
    bars = ax.barh(y_pos, vals, bar_height, left=lefts,
                   color=PALETTE[k], label=STOCK_NAMES[k], edgecolor="white", linewidth=0.8)
    # 标注
    for i, (v, l) in enumerate(zip(vals, lefts)):
        if v > 5000:
            ax.text(l + v / 2, i, f"{v/1e4:.1f}万",
                    ha="center", va="center", fontsize=9,
                    fontproperties=CN_FONT, color="white")
    lefts += vals

# 现金
cash_vals = np.array([cash_by_year.get(y, 0) for y in YEARS])
ax.barh(y_pos, cash_vals, bar_height, left=lefts,
        color=PALETTE["cash"], label="保留现金", edgecolor="white", linewidth=0.8)
for i, (v, l) in enumerate(zip(cash_vals, lefts)):
    if v > 5000:
        ax.text(l + v / 2, i, f"{v/1e4:.1f}万",
                ha="center", va="center", fontsize=9,
                fontproperties=CN_FONT, color="#555555")

# 总资产标注在最右侧
total_lefts = lefts + cash_vals
for i, y in enumerate(YEARS):
    W = W_by_year.get(y, 0)
    ax.text(total_lefts[i] + W * 0.01, i, f"共 {W/1e4:.1f}万",
            va="center", fontsize=9.5, fontproperties=CN_FONT, color="#333333")

ax.set_yticks(y_pos)
ax.set_yticklabels(year_labels)
ax.set_xlabel("持仓金额（元）")
ax.set_title("五年各年度持仓结构（横向堆叠柱状图）", fontsize=14)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e4:.0f}万"))
ax.legend(prop=CN_FONT, loc="lower right", framealpha=0.85)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", linestyle="--", alpha=0.35, color="gray")

beautify(CN_FONT, EN_FONT)
fname = os.path.join(OUTPUT_DIR, "figB_horizontal_bar.png")
plt.savefig(fname, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ {fname}")



categories = [STOCK_NAMES[k] for k in STOCKS]
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

YEAR_COLORS = ["#2B4F8C", "#3D7BBF", "#5BAAD4", "#3DA876", "#1E7A52"]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

ax.set_facecolor("white")
ax.yaxis.grid(True, color="#CCCCCC", linestyle="--", linewidth=0.8, alpha=0.7)
ax.xaxis.grid(True, color="#CCCCCC", linestyle="--", linewidth=0.8, alpha=0.7)
ax.spines["polar"].set_color("#CCCCCC")
ax.spines["polar"].set_linewidth(0.8)

# 轴标签
ax.set_thetagrids(np.degrees(angles[:-1]), categories)
for label in ax.get_xticklabels():
    label.set_fontproperties(CN_FONT)
    label.set_fontsize(12)

# 画五年
all_vals = []
for yi, y in enumerate(YEARS):
    rac = [float(rac_df.loc[y, k]) if y in rac_df.index else 0 for k in STOCKS]
    all_vals.extend(rac)
    rac_plot = rac + rac[:1]
    color = YEAR_COLORS[yi]
    ax.plot(angles, rac_plot, color=color, linewidth=2, linestyle="-", label=str(y) + "年")
    ax.fill(angles, rac_plot, color=color, alpha=0.08)
    # 节点标记
    ax.scatter(angles[:-1], rac, color=color, s=40, zorder=5)

# y轴刻度字体
for label in ax.get_yticklabels():
    label.set_fontproperties(EN_FONT)
    label.set_fontsize(8)
    label.set_color("#888888")

ax.set_title("五年各股票风险调整收益雷达图", fontsize=14, pad=25)

legend = ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12),
                   prop=CN_FONT, framealpha=0.85, edgecolor="#CCCCCC")

beautify(CN_FONT, EN_FONT)
# 图例年份标签单独设置
for text in legend.get_texts():
    text.set_fontproperties(CN_FONT)

fname = os.path.join(OUTPUT_DIR, "figE_radar_5years.png")
plt.savefig(fname, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ {fname}")

print("\n全部完成！")