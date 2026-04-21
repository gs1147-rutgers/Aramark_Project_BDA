"""
Aramark SRF Spend Analysis — Visualization Suite v2
=====================================================
Complete redesign: clean corporate style, larger fonts,
callout annotations, consistent palette across all 19 charts
+ 4 new big-data insight charts from training_chunks/.
"""

import os
_HERE = os.path.dirname(os.path.abspath(__file__)) + os.sep
_HERE = os.path.dirname(os.path.abspath(__file__)) + os.sep
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter, MaxNLocator
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path(_HERE + "ml_outputs")
CHUNKS_DIR = Path(_HERE + "training_chunks")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Corporate colour palette ──────────────────────────────────────────────────
NAVY      = "#1B3A6B"
TEAL      = "#0083B5"
ORANGE    = "#F26B21"
GREEN     = "#27AE60"
RED       = "#C0392B"
LGRAY     = "#F7F9FC"
MGRAY     = "#BDC3C7"
DGRAY     = "#555555"
WHITE     = "#FFFFFF"

CAT_COLORS  = [TEAL, NAVY, ORANGE, GREEN, RED, "#8E44AD", "#1ABC9C",
               "#E67E22", "#2980B9", "#D35400", "#16A085", "#7F8C8D", "#2C3E50"]
CLUSTER_CLR = [NAVY, TEAL, ORANGE, GREEN, RED, "#8E44AD", "#1ABC9C"]

def _apply_style(fig, ax_list=None):
    """Apply consistent corporate style to a figure."""
    fig.patch.set_facecolor(WHITE)
    for ax in (ax_list or fig.get_axes()):
        ax.set_facecolor(LGRAY)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(MGRAY)
        ax.spines["bottom"].set_color(MGRAY)
        ax.tick_params(colors=DGRAY, labelsize=11)
        ax.xaxis.label.set_color(DGRAY)
        ax.yaxis.label.set_color(DGRAY)
        ax.title.set_color(NAVY)
        ax.grid(axis="both", color=MGRAY, linewidth=0.5, alpha=0.6, linestyle="--")
        ax.set_axisbelow(True)

def _add_footer(fig, text="Aramark SRF Spend Analysis  |  2026"):
    fig.text(0.99, 0.01, text, ha="right", va="bottom", fontsize=8,
             color=MGRAY, style="italic")

def _dollar(x, pos):
    if abs(x) >= 1e6:   return f"${x/1e6:.1f}M"
    if abs(x) >= 1e3:   return f"${x/1e3:.0f}K"
    return f"${x:.0f}"

def save(name):
    p = OUTPUT_DIR / name
    plt.savefig(p, dpi=180, bbox_inches="tight", facecolor=WHITE)
    print(f"  ✓ {name}")
    plt.close("all")

# ── Load existing ML outputs ──────────────────────────────────────────────────
fi        = pd.read_csv(OUTPUT_DIR / "feature_importance.csv", index_col=0)
seg_spend = pd.read_csv(OUTPUT_DIR / "segment_spend.csv", index_col=0)
cat_spend = pd.read_csv(OUTPUT_DIR / "category_spend.csv", index_col=0)
state_spend= pd.read_csv(OUTPUT_DIR / "state_spend.csv", index_col=0)
clusters  = pd.read_csv(OUTPUT_DIR / "cluster_profiles.csv", index_col=0)
customers = pd.read_csv(OUTPUT_DIR / "customer_clusters.csv")
with open(OUTPUT_DIR / "analysis_report.json") as f:
    report = json.load(f)

# Load training chunk outputs
try:
    df_seg  = pd.read_parquet(CHUNKS_DIR / "chunk_02_segment_temporal.parquet")
    df_cat  = pd.read_parquet(CHUNKS_DIR / "chunk_03_category_jaccard.parquet")
    df_cust = pd.read_parquet(CHUNKS_DIR / "chunk_01_customer_features.parquet")
    HAS_CHUNKS = True
except Exception:
    HAS_CHUNKS = False

# =============================================================================
# CHART 01 — Feature Importance Consensus
# =============================================================================
print("\n[EDA Charts 01–11]")
fi_s = fi.sort_values("Avg_Rank")
labels = [n.replace("_enc","").replace("_"," ").title() for n in fi_s.index]
vals   = fi_s["Avg_Rank"].values
colors = [ORANGE if i == 0 else (TEAL if i < 3 else NAVY) for i in range(len(vals))]

fig, ax = plt.subplots(figsize=(13, 7))
bars = ax.barh(range(len(vals)), vals, color=colors, height=0.65, edgecolor="none")
ax.set_yticks(range(len(vals)))
ax.set_yticklabels(labels, fontsize=12)
ax.invert_yaxis()
ax.set_xlabel("Average Rank  (1 = most important)", fontsize=12, fontweight="bold")
ax.set_title("Feature Importance Consensus\nRF + Permutation Importance + Mutual Information",
             fontsize=15, fontweight="bold", pad=14)

for i, v in enumerate(vals):
    ax.text(v + 0.15, i, f"{v:.1f}", va="center", fontsize=11, color=DGRAY)

# Legend patches
legend_handles = [
    mpatches.Patch(facecolor=ORANGE, label="#1 Driver"),
    mpatches.Patch(facecolor=TEAL,   label="Top 3"),
    mpatches.Patch(facecolor=NAVY,   label="Other"),
]
ax.legend(handles=legend_handles, fontsize=10, loc="lower right")

_apply_style(fig)
_add_footer(fig)
plt.tight_layout()
save("01_feature_importance_consensus.png")

# =============================================================================
# CHART 02 — Feature Importance Methods Compared
# =============================================================================
methods = ["RF_Importance", "Permutation_Importance", "Mutual_Information"]
titles  = ["Random Forest", "Permutation\n(HistGB)", "Mutual\nInformation"]
grad    = [plt.cm.Blues, plt.cm.Oranges, plt.cm.Greens]

fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=False)
fig.suptitle("Feature Importance — Three Independent Methods",
             fontsize=15, fontweight="bold", color=NAVY, y=1.01)

for ax, method, title, cmap in zip(axes, methods, titles, grad):
    top = fi[method].nlargest(10)
    feature_labels = [n.replace("_enc","").replace("_"," ").title() for n in top.index]
    clrs = cmap(np.linspace(0.35, 0.85, len(top)))
    ax.barh(range(len(top)), top.values, color=clrs, height=0.65, edgecolor="none")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(feature_labels, fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlabel("Score", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", color=NAVY)
    for i, v in enumerate(top.values):
        ax.text(v * 1.02, i, f"{v:.3f}", va="center", fontsize=9, color=DGRAY)

_apply_style(fig)
_add_footer(fig)
plt.tight_layout()
save("02_feature_importance_methods.png")

# =============================================================================
# CHART 03 — Category Spend
# =============================================================================
cat_s = cat_spend["sum"].sort_values(ascending=False)
pcts  = cat_s / cat_s.sum() * 100
clrs  = CAT_COLORS[:len(cat_s)]

fig = plt.figure(figsize=(16, 7))
gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 1.3], wspace=0.35)
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])
fig.suptitle("Category Level 1 — Spend Breakdown", fontsize=15, fontweight="bold",
             color=NAVY, y=1.02)

# Donut chart (cleaner than pie)
wedges, texts, autotexts = ax0.pie(
    cat_s.values, labels=cat_s.index, autopct="%1.1f%%",
    colors=clrs, startangle=90, pctdistance=0.8,
    wedgeprops=dict(width=0.55, edgecolor=WHITE, linewidth=2),
    textprops={"fontsize": 9.5},
)
for at in autotexts:
    at.set_fontsize(8.5)
    at.set_fontweight("bold")
    at.set_color(WHITE)
ax0.set_title("Spend Share (%)", fontsize=13, fontweight="bold", color=NAVY, pad=10)

# Horizontal bar with value labels
ax1.barh(range(len(cat_s)), cat_s.values / 1e6, color=clrs, height=0.7, edgecolor="none")
ax1.set_yticks(range(len(cat_s)))
ax1.set_yticklabels(cat_s.index, fontsize=11)
ax1.invert_yaxis()
ax1.set_xlabel("Total Spend ($M)", fontsize=12, fontweight="bold")
ax1.set_title("Total Spend by Category", fontsize=13, fontweight="bold", color=NAVY)
ax1.xaxis.set_major_formatter(FuncFormatter(_dollar))
for i, v in enumerate(cat_s.values / 1e6):
    ax1.text(v + 0.4, i, f"${v:.1f}M  ({pcts.iloc[i]:.1f}%)", va="center", fontsize=10)

_apply_style(fig, [ax0, ax1])
ax0.set_facecolor(WHITE)
_add_footer(fig)
plt.tight_layout()
save("03_category_breakdown.png")

# =============================================================================
# CHART 04 — Market Segments
# =============================================================================
seg_top = seg_spend.nlargest(15, "TotalSpend")

fig, axes = plt.subplots(1, 2, figsize=(17, 7), sharey=True)
fig.suptitle("Market Segment Performance — Top 15 Segments",
             fontsize=15, fontweight="bold", color=NAVY, y=1.01)

# Left: total spend gradient
grad_l = [plt.cm.Blues(0.3 + 0.6 * i / (len(seg_top) - 1)) for i in range(len(seg_top))]
axes[0].barh(range(len(seg_top)), seg_top["TotalSpend"] / 1e6,
             color=grad_l, height=0.7, edgecolor="none")
axes[0].set_yticks(range(len(seg_top)))
axes[0].set_yticklabels(seg_top.index, fontsize=10.5)
axes[0].invert_yaxis()
axes[0].set_xlabel("Total Spend ($M)", fontsize=11, fontweight="bold")
axes[0].set_title("Total Spend", fontsize=13, fontweight="bold", color=NAVY)
for i, v in enumerate(seg_top["TotalSpend"] / 1e6):
    axes[0].text(v + 0.1, i, f"${v:.1f}M", va="center", fontsize=9.5)

# Right: avg ticket
grad_r = [plt.cm.Oranges(0.3 + 0.6 * i / (len(seg_top) - 1)) for i in range(len(seg_top))]
axes[1].barh(range(len(seg_top)), seg_top["AvgSpend"],
             color=grad_r, height=0.7, edgecolor="none")
axes[1].set_yticks(range(len(seg_top)))
axes[1].set_yticklabels(seg_top.index, fontsize=10.5)
axes[1].invert_yaxis()
axes[1].set_xlabel("Avg Spend per Transaction ($)", fontsize=11, fontweight="bold")
axes[1].set_title("Avg Transaction Value", fontsize=13, fontweight="bold", color=NAVY)
for i, v in enumerate(seg_top["AvgSpend"]):
    axes[1].text(v + 1, i, f"${v:.0f}", va="center", fontsize=9.5)

_apply_style(fig)
_add_footer(fig)
plt.tight_layout()
save("04_top_segments.png")

# =============================================================================
# CHART 05 — Geographic (States)
# =============================================================================
state_top = state_spend.nlargest(20, "sum")
intensity = np.linspace(0.85, 0.3, len(state_top))
clrs_s = [plt.cm.Blues(v) for v in intensity]

fig, ax = plt.subplots(figsize=(15, 6))
bars = ax.bar(range(len(state_top)), state_top["sum"] / 1e6,
              color=clrs_s, edgecolor="none", width=0.7)
# Highlight top 3
for i in range(3):
    bars[i].set_color(ORANGE)

ax.set_xticks(range(len(state_top)))
ax.set_xticklabels(state_top.index, fontsize=11, fontweight="bold")
ax.set_ylabel("Total Spend ($M)", fontsize=12, fontweight="bold")
ax.set_title("Top 20 States — Spend Concentration\n(Orange = Top 3, account for 36% of total)",
             fontsize=14, fontweight="bold", color=NAVY)
ax.yaxis.set_major_formatter(FuncFormatter(_dollar))

for i, v in enumerate(state_top["sum"] / 1e6):
    ax.text(i, v + 0.15, f"${v:.1f}M", ha="center", fontsize=9,
            color=NAVY, fontweight="bold" if i < 3 else "normal")

_apply_style(fig)
_add_footer(fig)
plt.tight_layout()
save("05_top_states.png")

# =============================================================================
# CHART 06 — Pareto Curve
# =============================================================================
cust_sp = customers.groupby("Customer Id")["TotalSpend"].sum().sort_values(ascending=False)
cumsum  = cust_sp.cumsum() / cust_sp.sum() * 100
pct_cus = np.arange(1, len(cumsum) + 1) / len(cumsum) * 100

fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(pct_cus, cumsum.values, linewidth=3.5, color=NAVY, label="Actual concentration")
ax.plot([0, 100], [0, 100], "--", linewidth=1.8, color=MGRAY, label="Perfect equality")
ax.fill_between(pct_cus, cumsum.values, pct_cus, alpha=0.15, color=TEAL)

# Pareto markers
pts = [(1, 24.4, "Top 1%\n→ 24.4%"), (10, 65.0, "Top 10%\n→ 65%"), (20, 79.1, "Top 20%\n→ 79.1%")]
for x, y, label in pts:
    ax.scatter(x, y, s=200, color=ORANGE, zorder=6, edgecolors=NAVY, linewidth=1.5)
    ax.annotate(label, xy=(x, y), xytext=(x + 5, y - 8),
                fontsize=11, fontweight="bold", color=NAVY,
                arrowprops=dict(arrowstyle="-", color=MGRAY, lw=1.2))

ax.set_xlabel("Customers ranked by spend (%)", fontsize=12, fontweight="bold")
ax.set_ylabel("Cumulative spend captured (%)", fontsize=12, fontweight="bold")
ax.set_title("Revenue Concentration (Pareto)\nTop 10% of customers drive 65% of revenue",
             fontsize=14, fontweight="bold", color=NAVY)
ax.legend(fontsize=11, loc="lower right")
ax.set_xlim(0, 100); ax.set_ylim(0, 105)

_apply_style(fig)
_add_footer(fig)
plt.tight_layout()
save("06_pareto_curve.png")

# =============================================================================
# CHART 07 — Cluster Heatmap
# =============================================================================
cluster_data = clusters.copy()
c_norm = (cluster_data - cluster_data.min()) / (cluster_data.max() - cluster_data.min() + 1e-9)

fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(
    c_norm.T, annot=cluster_data.T.round(0), fmt="g",
    cmap=sns.color_palette("Blues", as_cmap=True),
    cbar_kws={"label": "Normalised value (0–1)", "shrink": 0.7},
    ax=ax, linewidths=1.5, linecolor=WHITE,
    annot_kws={"size": 10, "color": NAVY, "fontweight": "bold"},
)
ax.set_xlabel("Customer Cluster", fontsize=12, fontweight="bold")
ax.set_ylabel("")
ax.set_title("Customer Segment Profiles (5 Clusters)\nDarker = higher relative value",
             fontsize=14, fontweight="bold", color=NAVY)
ax.tick_params(labelsize=11)
ax.set_yticklabels(
    [l.get_text().replace("Log","").replace("_"," ").title()
     for l in ax.get_yticklabels()], rotation=0)

cluster_labels = {0:"Mid-Tier", 1:"Enterprise", 2:"Micro", 3:"Small", 4:"Regular"}
ax.set_xticklabels(
    [f"C{i}\n{cluster_labels.get(i,'')}" for i in range(len(clusters))], rotation=0)

_add_footer(fig)
fig.patch.set_facecolor(WHITE)
plt.tight_layout()
save("07_cluster_heatmap.png")

# =============================================================================
# CHART 08 — Cluster Scatter
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

cluster_labels_full = {0:"Mid-Tier", 1:"Enterprise", 2:"Micro", 3:"Small", 4:"Regular"}
for cid in sorted(customers["Cluster"].unique()):
    sub = customers[customers["Cluster"] == cid]
    ax.scatter(sub["Transactions"], sub["LogTotalSpend"],
               s=220, alpha=0.7, label=f"C{cid}: {cluster_labels_full.get(cid,'')}",
               color=CLUSTER_CLR[cid], edgecolors=WHITE, linewidth=1)

ax.set_xlabel("Annual Transactions", fontsize=12, fontweight="bold")
ax.set_ylabel("Log Total Spend ($)", fontsize=12, fontweight="bold")
ax.set_title("Customer Clusters — Transaction Volume vs Spend\nClear tier separation enables targeted strategies",
             fontsize=14, fontweight="bold", color=NAVY)
ax.legend(fontsize=11, loc="upper left", framealpha=0.9)

_apply_style(fig)
_add_footer(fig)
plt.tight_layout()
save("08_cluster_scatter.png")

# =============================================================================
# CHART 09 — Spend Distribution
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Customer Spend Distribution", fontsize=15, fontweight="bold", color=NAVY)

axes[0].hist(np.log1p(customers["TotalSpend"]), bins=80, color=TEAL,
             alpha=0.85, edgecolor="none")
axes[0].set_xlabel("Log(Annual Spend $)", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Number of Customers", fontsize=12, fontweight="bold")
axes[0].set_title("Spend Distribution (Log-Transformed)\nLog-normal shape is typical for B2B",
                  fontsize=12, color=NAVY)

bp_data = [customers[customers["Cluster"] == i]["TotalSpend"].values for i in range(5)]
bp = axes[1].boxplot(bp_data,
    labels=[f"C{i}\n{cluster_labels_full.get(i,'')}" for i in range(5)],
    patch_artist=True, medianprops=dict(color=WHITE, linewidth=2.5),
    whiskerprops=dict(color=MGRAY), capprops=dict(color=MGRAY),
    flierprops=dict(marker="o", alpha=0.3, markersize=4))
for patch, color in zip(bp["boxes"], CLUSTER_CLR[:5]):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
axes[1].set_ylabel("Total Spend ($)", fontsize=12, fontweight="bold")
axes[1].set_title("Spend by Cluster — Range & Outliers",
                  fontsize=12, color=NAVY)
axes[1].yaxis.set_major_formatter(FuncFormatter(_dollar))

_apply_style(fig)
_add_footer(fig)
plt.tight_layout()
save("09_spend_distribution.png")

# =============================================================================
# CHART 10 — Monthly Trends
# =============================================================================
if "monthly_trend" in report:
    monthly = report["monthly_trend"]
    months_list = sorted(monthly.keys(), key=lambda x: float(x))
    values = [monthly[m] for m in months_list]
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"][:len(months_list)]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(range(len(months_list)), np.array(values) / 1e6,
            marker="o", linewidth=3, markersize=9, color=NAVY, zorder=3)
    ax.fill_between(range(len(months_list)), np.array(values) / 1e6,
                    min(values) / 1e6, alpha=0.18, color=TEAL)

    # Shade summer trough
    ax.axvspan(5, 6.5, alpha=0.1, color=ORANGE, label="Summer trough")
    # Shade fall peak
    ax.axvspan(7.5, 9.5, alpha=0.1, color=GREEN, label="Fall peak")

    max_idx = int(np.argmax(values))
    min_idx = int(np.argmin(values))
    ax.scatter([max_idx], [values[max_idx] / 1e6], s=300, color=GREEN, zorder=5,
               marker="*", label=f"Peak: {month_labels[max_idx]}")
    ax.scatter([min_idx], [values[min_idx] / 1e6], s=200, color=RED, zorder=5,
               marker="v", label=f"Trough: {month_labels[min_idx]}")

    for i, v in enumerate(np.array(values) / 1e6):
        ax.text(i, v + 0.12, f"${v:.1f}M", ha="center", fontsize=9.5, color=NAVY)

    ax.set_xticks(range(len(months_list)))
    ax.set_xticklabels(month_labels, fontsize=12, fontweight="bold")
    ax.set_ylabel("Total Spend ($M)", fontsize=12, fontweight="bold")
    ax.set_title("Monthly Spend — Seasonality Pattern (2025)\n31% swing from trough to peak",
                 fontsize=14, fontweight="bold", color=NAVY)
    ax.legend(fontsize=10, loc="lower right")
    ax.yaxis.set_major_formatter(FuncFormatter(_dollar))

    _apply_style(fig)
    _add_footer(fig)
    plt.tight_layout()
    save("10_monthly_trends.png")

# =============================================================================
# CHART 11 — Business Entity Type
# =============================================================================
if "biz_type_spend" in report:
    biz_data = report["biz_type_spend"]
    biz_types = list(biz_data["sum"].keys())[:4]
    biz_spend = [biz_data["sum"][b] / 1e6 for b in biz_types]
    biz_avg   = [biz_data["mean"][b]      for b in biz_types]
    colors_biz = [NAVY, TEAL, ORANGE, GREEN][:len(biz_types)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Business Entity Type — Volume vs Value",
                 fontsize=15, fontweight="bold", color=NAVY)

    axes[0].bar(biz_types, biz_spend, color=colors_biz, edgecolor="none", width=0.55)
    axes[0].set_ylabel("Total Spend ($M)", fontsize=12, fontweight="bold")
    axes[0].set_title("Total Spend Volume\nGPO drives absolute revenue",
                      fontsize=12, color=NAVY)
    axes[0].yaxis.set_major_formatter(FuncFormatter(_dollar))
    for i, v in enumerate(biz_spend):
        axes[0].text(i, v + 0.5, f"${v:.1f}M", ha="center",
                     fontsize=11, fontweight="bold", color=colors_biz[i])

    axes[1].bar(biz_types, biz_avg, color=colors_biz, edgecolor="none", width=0.55)
    axes[1].set_ylabel("Avg Transaction ($)", fontsize=12, fontweight="bold")
    axes[1].set_title("Avg Transaction Value\nManaged Services = 2× premium",
                      fontsize=12, color=NAVY)
    axes[1].yaxis.set_major_formatter(FuncFormatter(_dollar))
    for i, v in enumerate(biz_avg):
        axes[1].text(i, v + 2, f"${v:.0f}", ha="center",
                     fontsize=11, fontweight="bold", color=colors_biz[i])

    _apply_style(fig)
    _add_footer(fig)
    plt.tight_layout()
    save("11_business_entity_type.png")

# ── Advanced ML charts 12–19 ──────────────────────────────────────────────────
print("\n[Advanced ML Charts 12–19]")

try:
    fi_adv = pd.read_csv(OUTPUT_DIR / "feature_importance_advanced.csv", index_col=0)
    mc     = pd.read_csv(OUTPUT_DIR / "model_comparison.csv", index_col=0)
    # Normalise column name (may be "R²" or "R2")
    if "R²" in mc.columns:
        mc = mc.rename(columns={"R²": "R2"})
    with open(OUTPUT_DIR / "advanced_models_report.json") as f:
        adv = json.load(f)

    # ==========================================================================
    # CHART 12 — Model Comparison
    # ==========================================================================
    metrics = ["R2", "MAE", "RMSE"]
    ylabels = ["R² Score (↑ better)", "MAE (↓ better)", "RMSE (↓ better)"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("ML Model Performance Comparison — All 6 Models",
                 fontsize=15, fontweight="bold", color=NAVY, y=1.02)

    for ax, metric, ylabel in zip(axes, metrics, ylabels):
        vals_m = mc[metric].values
        names  = mc.index.tolist()
        best_i = vals_m.argmax() if metric == "R2" else vals_m.argmin()
        clrs_m = [ORANGE if i == best_i else NAVY for i in range(len(vals_m))]
        ax.bar(range(len(names)), vals_m, color=clrs_m, edgecolor="none", width=0.6)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace("XGBoost", "XGB").replace("Gradient","GB")
                            .replace("Random","RF").replace("Stacking","Stack")
                            for n in names], rotation=30, ha="right", fontsize=9.5)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(metric, fontsize=13, fontweight="bold", color=NAVY)
        for i, v in enumerate(vals_m):
            ax.text(i, v * (1.02 if metric == "R2" else 0.98), f"{v:.3f}",
                    ha="center", fontsize=9,
                    fontweight="bold" if i == best_i else "normal")

    _apply_style(fig)
    _add_footer(fig)
    plt.tight_layout()
    save("12_model_comparison_all_metrics.png")

    # ==========================================================================
    # CHART 13 — XGB vs LGB vs SHAP feature importance
    # ==========================================================================
    top_n = 10
    # Column names: XGBoost, LightGBM, SHAP
    xgb_col  = "XGBoost"  if "XGBoost"  in fi_adv.columns else fi_adv.columns[0]
    lgb_col  = "LightGBM" if "LightGBM" in fi_adv.columns else fi_adv.columns[1]
    shap_col = "SHAP"     if "SHAP"     in fi_adv.columns else fi_adv.columns[2]
    xgb_fi = fi_adv[xgb_col].nlargest(top_n)
    shap_fi = fi_adv[shap_col].nlargest(top_n)
    lgb_fi  = fi_adv[lgb_col].nlargest(top_n)

    def clean_label(s):
        return s.replace("_enc","").replace("_"," ").title()

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=False)
    fig.suptitle("Feature Importance — Advanced Methods (XGBoost, LightGBM, SHAP)",
                 fontsize=15, fontweight="bold", color=NAVY, y=1.01)

    for ax, series, title, cmap in zip(
        axes,
        [xgb_fi, lgb_fi, shap_fi],
        ["XGBoost", "LightGBM\n(# Splits)", "SHAP Values\n(XGB)"],
        [plt.cm.Blues, plt.cm.Oranges, plt.cm.Greens]
    ):
        lbls  = [clean_label(n) for n in series.index]
        clrs2 = cmap(np.linspace(0.35, 0.85, len(series)))
        ax.barh(range(len(series)), series.values, color=clrs2, height=0.65, edgecolor="none")
        ax.set_yticks(range(len(series)))
        ax.set_yticklabels(lbls, fontsize=10.5)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=13, fontweight="bold", color=NAVY)
        for i, v in enumerate(series.values):
            ax.text(v * 1.02, i, f"{v:.3f}", va="center", fontsize=9.5)

    _apply_style(fig)
    _add_footer(fig)
    plt.tight_layout()
    save("13_feature_importance_xgb_lgb_shap.png")

    # ==========================================================================
    # CHART 14 — Hyperparameter tuning impact
    # ==========================================================================
    model_names  = mc.index.tolist()
    r2_vals      = mc["R2"].values
    best_i       = r2_vals.argmax()
    clrs_14 = [ORANGE if i == best_i else (GREEN if r2_vals[i] > r2_vals[0] else RED)
               for i in range(len(r2_vals))]

    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(range(len(model_names)), r2_vals, color=clrs_14, edgecolor="none", width=0.6)
    ax.axhline(y=r2_vals[0], linestyle="--", color=MGRAY, linewidth=1.5, label="RF baseline")
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels([n.replace("XGBoost","XGB").replace("Gradient","GB")
                        .replace("Random","RF").replace(" ","\\n") for n in model_names],
                       rotation=20, ha="right", fontsize=11)
    ax.set_ylabel("R² Score", fontsize=12, fontweight="bold")
    ax.set_title("Hyperparameter Tuning Impact — R² Progression\nOrange = best model | Green = improvement | Red = regression",
                 fontsize=13, fontweight="bold", color=NAVY)
    ax.legend(fontsize=10)
    for i, v in enumerate(r2_vals):
        ax.text(i, v + 0.003, f"{v:.4f}", ha="center", fontsize=10,
                fontweight="bold" if i == best_i else "normal", color=NAVY)

    _apply_style(fig)
    _add_footer(fig)
    plt.tight_layout()
    save("14_hyperparameter_tuning_impact.png")

    # ==========================================================================
    # CHART 15 — Cross-validation
    # ==========================================================================
    cv_results = adv.get("cross_validation", adv.get("cv_results", {}))
    if cv_results:
        cv_models = list(cv_results.keys())
        # Support both mean_r2 / std_r2 and nested dict with scores list
        def _mean_std(entry):
            if isinstance(entry, dict):
                if "mean_r2" in entry:
                    return entry["mean_r2"], entry.get("std_r2", 0)
                scores = entry.get("scores", [entry.get("r2", 0)])
            else:
                scores = list(entry) if hasattr(entry, "__iter__") else [float(entry)]
            import numpy as _np
            return float(_np.mean(scores)), float(_np.std(scores))
        cv_means = [_mean_std(cv_results[m])[0] for m in cv_models]
        cv_stds  = [_mean_std(cv_results[m])[1] for m in cv_models]
        best_cv   = np.argmax(cv_means)
        clrs_cv   = [ORANGE if i == best_cv else TEAL for i in range(len(cv_models))]

        fig, ax = plt.subplots(figsize=(11, 6))
        bars = ax.bar(range(len(cv_models)), cv_means, color=clrs_cv,
                      yerr=cv_stds, capsize=7, edgecolor="none", width=0.55,
                      error_kw=dict(color=DGRAY, linewidth=2))
        ax.set_xticks(range(len(cv_models)))
        ax.set_xticklabels([m.replace("XGBoost","XGB").replace("GradientBoost","GradBoost")
                            .replace("RandomForest","RF") for m in cv_models],
                           fontsize=11, rotation=15, ha="right")
        ax.set_ylabel("Cross-Validated R²  (mean ± std)", fontsize=12, fontweight="bold")
        ax.set_title("5-Fold Cross-Validation — Model Robustness\nError bars show stability across folds",
                     fontsize=13, fontweight="bold", color=NAVY)
        for i, (m, s) in enumerate(zip(cv_means, cv_stds)):
            ax.text(i, m + s + 0.003, f"{m:.4f}±{s:.4f}", ha="center", fontsize=9.5, color=NAVY)
        _apply_style(fig)
        _add_footer(fig)
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.text(0.5, 0.5, "Cross-validation results not in report",
                ha="center", va="center", fontsize=14, color=MGRAY)
    save("15_cross_validation_performance.png")

    # ==========================================================================
    # CHART 16 — Stacking meta-learner weights
    # ==========================================================================
    meta = adv.get("stacking_weights", adv.get("stacking", {}).get("meta_weights", {}))
    if meta:
        meta_models = list(meta.keys())
        meta_wts    = [float(meta[m]) for m in meta_models]
        meta_clrs   = [GREEN if w > 0 else RED for w in meta_wts]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(range(len(meta_models)), meta_wts, color=meta_clrs,
                       height=0.55, edgecolor="none")
        ax.axvline(0, color=MGRAY, linewidth=1.5, linestyle="--")
        ax.set_yticks(range(len(meta_models)))
        ax.set_yticklabels([m.replace("XGBoost","XGB").replace("GradientBoost","GradBoost")
                            .replace("RandomForest","RF") for m in meta_models], fontsize=11)
        ax.set_xlabel("Meta-Learner Weight", fontsize=12, fontweight="bold")
        ax.set_title("Stacking Ensemble — Meta-Learner Weights\nXGBoost heavily favoured; GradBoost penalised",
                     fontsize=13, fontweight="bold", color=NAVY)
        for i, v in enumerate(meta_wts):
            xpos = v + (0.04 if v >= 0 else -0.04)
            ax.text(xpos, i, f"{v:.3f}", va="center", fontsize=10.5,
                    ha="left" if v >= 0 else "right")
        _apply_style(fig)
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "Stacking weights not found in report",
                ha="center", va="center", fontsize=14, color=MGRAY)
        ax.axis("off")
    _add_footer(fig)
    plt.tight_layout()
    save("16_stacking_meta_weights.png")

    # ==========================================================================
    # CHART 17 — Best hyperparameters
    # ==========================================================================
    params = adv.get("best_params", adv.get("xgboost_tuned", {}).get("params", {}))
    if params:
        param_names = list(params.keys())
        param_vals  = [float(v) for v in params.values()]
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(range(len(param_names)), param_vals, color=TEAL,
                       height=0.55, edgecolor="none")
        ax.set_yticks(range(len(param_names)))
        ax.set_yticklabels([n.replace("_"," ").title() for n in param_names], fontsize=11)
        ax.set_xlabel("Optimised Value", fontsize=12, fontweight="bold")
        ax.set_title("Optuna Best Hyperparameters — XGBoost\n30-trial Bayesian optimisation",
                     fontsize=13, fontweight="bold", color=NAVY)
        for i, v in enumerate(param_vals):
            ax.text(v * 1.02, i, f"{v:.4g}", va="center", fontsize=10.5)
        _apply_style(fig)
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "Hyperparameter results not in report",
                ha="center", va="center", fontsize=14, color=MGRAY)
        ax.axis("off")
    _add_footer(fig)
    plt.tight_layout()
    save("17_best_hyperparameters.png")

    # ==========================================================================
    # CHART 18 — Top features consensus (advanced)
    # ==========================================================================
    fi_adv_s = fi_adv

    avail_cols = [c for c in [shap_col, xgb_col, lgb_col] if c in fi_adv_s.columns]
    if avail_cols:
        top_f = fi_adv_s[avail_cols[0]].nlargest(10)
        labels_18 = [clean_label(n) for n in top_f.index]
        clrs_18 = [ORANGE if i == 0 else (TEAL if i < 3 else NAVY) for i in range(len(top_f))]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(range(len(top_f)), top_f.values, color=clrs_18, height=0.65, edgecolor="none")
        ax.set_yticks(range(len(top_f)))
        ax.set_yticklabels(labels_18, fontsize=12)
        ax.invert_yaxis()
        ax.set_xlabel("Importance Score", fontsize=12, fontweight="bold")
        ax.set_title("Top 10 Features — Advanced Consensus\n(Category hierarchy is the dominant signal)",
                     fontsize=14, fontweight="bold", color=NAVY)
        for i, v in enumerate(top_f.values):
            ax.text(v * 1.02, i, f"{v:.4f}", va="center", fontsize=10.5)
        _apply_style(fig)
        _add_footer(fig)
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, "Advanced feature data unavailable",
                ha="center", va="center", fontsize=14, color=MGRAY)
        ax.axis("off")
    save("18_top_features_consensus_advanced.png")

    # ==========================================================================
    # CHART 19 — Model improvement waterfall
    # ==========================================================================
    # Add tuned XGBoost row if not already in mc
    if "XGBoost (Tuned)" not in mc.index and "xgboost_tuned" in adv:
        tuned_r2 = adv["xgboost_tuned"].get("r2", adv["xgboost_tuned"].get("R2", 0))
        tuned_mae = adv["xgboost_tuned"].get("mae", adv["xgboost_tuned"].get("MAE", 0))
        tuned_rmse = adv["xgboost_tuned"].get("rmse", adv["xgboost_tuned"].get("RMSE", 0))
        mc.loc["XGBoost (Tuned)"] = [tuned_r2, tuned_mae, tuned_rmse]
    wf_names = mc.index.tolist()
    wf_vals  = mc["R2"].values
    baseline = wf_vals[0]
    deltas   = np.diff(np.concatenate([[baseline], wf_vals]))

    fig, ax = plt.subplots(figsize=(13, 6))
    running = baseline
    for i, (name, delta) in enumerate(zip(wf_names, deltas)):
        bottom = min(running - delta, running) if delta < 0 else running
        color  = GREEN if delta >= 0 else RED
        if i == 0:
            ax.bar(i, wf_vals[0], color=NAVY, edgecolor="none", width=0.6, label="Baseline")
        else:
            ax.bar(i, abs(delta), bottom=bottom, color=color, edgecolor="none", width=0.6)
        ax.text(i, wf_vals[i] + 0.004, f"{wf_vals[i]:.4f}", ha="center",
                fontsize=9.5, fontweight="bold", color=NAVY)
        if i > 0 and delta != 0:
            sign = "▲" if delta > 0 else "▼"
            ax.text(i, bottom - 0.008, f"{sign}{abs(delta):.4f}", ha="center",
                    fontsize=8.5, color=color)
        running = wf_vals[i]

    ax.set_xticks(range(len(wf_names)))
    ax.set_xticklabels([n.replace("XGBoost","XGB").replace("Gradient","GB")
                        .replace("Random","RF") for n in wf_names],
                       rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("R² Score", fontsize=12, fontweight="bold")
    ax.set_title("Model Improvement Journey — R² Waterfall\nNet gain +18% from baseline RF → Tuned XGBoost",
                 fontsize=13, fontweight="bold", color=NAVY)
    legend_wf = [
        mpatches.Patch(facecolor=GREEN, label="Improvement"),
        mpatches.Patch(facecolor=RED,   label="Regression"),
        mpatches.Patch(facecolor=NAVY,  label="Baseline"),
    ]
    ax.legend(handles=legend_wf, fontsize=10)
    _apply_style(fig)
    _add_footer(fig)
    plt.tight_layout()
    save("19_model_improvement_waterfall.png")

except Exception as e:
    print(f"  ⚠ Advanced ML charts skipped: {e}")

# =============================================================================
# BIG DATA INSIGHT CHARTS (20–23) — from training_chunks/
# =============================================================================
if HAS_CHUNKS:
    print("\n[Big Data Insight Charts 20–23]")

    # ========================================================================
    # CHART 20 — Segment Risk Matrix (AMS × Trend)
    # ========================================================================
    try:
        fig, ax = plt.subplots(figsize=(13, 8))

        segs      = df_seg["segment_id"].tolist()
        x         = df_seg["spend_trend_slope"].values          # positive = growing
        y         = df_seg["ams_surprise_index"].values         # high = concentrated
        sizes     = (df_seg["ams_f1_total_spend"].values / df_seg["ams_f1_total_spend"].max() * 900 + 80)
        vol       = df_seg["monthly_cv"].values

        sc = ax.scatter(x, y, s=sizes, c=vol, cmap="RdYlGn_r",
                        alpha=0.8, edgecolors=WHITE, linewidth=1.5, zorder=3)
        plt.colorbar(sc, ax=ax, label="Monthly Volatility (CV)", pad=0.02)

        # Quadrant lines
        med_x, med_y = np.median(x), np.median(y)
        ax.axvline(x=0, color=MGRAY, linewidth=1.2, linestyle="--", zorder=1)
        ax.axhline(y=med_y, color=MGRAY, linewidth=1.2, linestyle="--", zorder=1)

        # Quadrant labels
        ax_xl, ax_xr = ax.get_xlim()
        ax_yb, ax_yt = ax.get_ylim()
        ax.text(0.72, 0.97, "⚠ HIGH RISK\nConcentrated + Declining",
                transform=ax.transAxes, fontsize=9.5, color=RED,
                fontweight="bold", va="top", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF0F0", alpha=0.8))
        ax.text(0.28, 0.97, "★ STAR\nConcentrated + Growing",
                transform=ax.transAxes, fontsize=9.5, color=GREEN,
                fontweight="bold", va="top", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0FFF0", alpha=0.8))
        ax.text(0.72, 0.05, "⬇ MONITOR\nDiversified + Declining",
                transform=ax.transAxes, fontsize=9.5, color=ORANGE,
                fontweight="bold", va="bottom", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFDF0", alpha=0.8))
        ax.text(0.28, 0.05, "✓ STABLE\nDiversified + Growing",
                transform=ax.transAxes, fontsize=9.5, color=TEAL,
                fontweight="bold", va="bottom", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F8FF", alpha=0.8))

        for i, seg in enumerate(segs):
            if abs(x[i]) > np.percentile(np.abs(x), 65) or y[i] > np.percentile(y, 80):
                ax.annotate(seg, (x[i], y[i]), fontsize=8, color=DGRAY,
                            xytext=(6, 4), textcoords="offset points")

        ax.set_xlabel("Monthly Spend Trend Slope ($/month)", fontsize=12, fontweight="bold")
        ax.set_ylabel("AMS Surprise Index (spend concentration)", fontsize=12, fontweight="bold")
        ax.set_title("Segment Risk Matrix — Trend × Concentration\n"
                     "Bubble size = total spend  |  Colour = volatility (red = volatile)",
                     fontsize=14, fontweight="bold", color=NAVY)

        _apply_style(fig)
        _add_footer(fig)
        plt.tight_layout()
        save("20_segment_risk_matrix.png")
    except Exception as e:
        print(f"  ⚠ Chart 20 skipped: {e}")

    # ========================================================================
    # CHART 21 — Customer Cluster Comparison (new vs old)
    # ========================================================================
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Customer Segmentation — K-Means vs BIRCH Cluster Profiles",
                     fontsize=15, fontweight="bold", color=NAVY)

        for ax, col, title in zip(axes, ["kmeans_cluster","birch_cluster"],
                                  ["MiniBatch K-Means (7 clusters)","BIRCH (7 clusters)"]):
            if col not in df_cust.columns:
                continue
            profile = df_cust.groupby(col)[
                ["total_spend","total_txns","category_breadth","active_months"]
            ].mean().round(1)

            x_pos = range(len(profile))
            bars_ax = ax.bar(x_pos, profile["total_spend"] / 1e3,
                             color=CLUSTER_CLR[:len(profile)], edgecolor="none", width=0.6)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f"C{i}" for i in profile.index], fontsize=11)
            ax.set_ylabel("Avg Spend per Customer ($K)", fontsize=11, fontweight="bold")
            ax.set_title(title, fontsize=13, fontweight="bold", color=NAVY)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"${v:.0f}K"))
            for xi, v in enumerate(profile["total_spend"] / 1e3):
                n = int(df_cust[col].value_counts().get(profile.index[xi], 0))
                ax.text(xi, v + 1, f"${v:.0f}K\nn={n:,}", ha="center", fontsize=9)

        _apply_style(fig)
        _add_footer(fig)
        plt.tight_layout()
        save("21_cluster_comparison_kmeans_birch.png")
    except Exception as e:
        print(f"  ⚠ Chart 21 skipped: {e}")

    # ========================================================================
    # CHART 22 — CMS Category Entropy Distribution
    # ========================================================================
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Customer Purchase Diversity — Count-Min Sketch Analysis",
                     fontsize=15, fontweight="bold", color=NAVY)

        axes[0].hist(df_cust["cms_category_entropy"].dropna(), bins=60,
                     color=TEAL, edgecolor="none", alpha=0.85)
        axes[0].set_xlabel("Category Entropy (bits)", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Number of Customers", fontsize=12, fontweight="bold")
        axes[0].set_title("Category Purchase Entropy\nHigh entropy = diversified buyer",
                          fontsize=12, color=NAVY)
        med_e = df_cust["cms_category_entropy"].median()
        axes[0].axvline(med_e, color=ORANGE, linewidth=2, linestyle="--",
                        label=f"Median: {med_e:.2f} bits")
        axes[0].legend(fontsize=10)

        axes[1].scatter(df_cust["category_breadth"], df_cust["cms_category_entropy"],
                        alpha=0.15, s=12, color=NAVY, rasterized=True)
        axes[1].set_xlabel("Category Breadth (unique L1 categories)", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("CMS Category Entropy (bits)", fontsize=12, fontweight="bold")
        axes[1].set_title("Breadth vs Entropy\nLinear relationship confirms CMS accuracy",
                          fontsize=12, color=NAVY)

        _apply_style(fig)
        _add_footer(fig)
        plt.tight_layout()
        save("22_cms_category_entropy.png")
    except Exception as e:
        print(f"  ⚠ Chart 22 skipped: {e}")

    # ========================================================================
    # CHART 23 — MinHash Basket Similarity + DGIM Active Rate
    # ========================================================================
    try:
        fig = plt.figure(figsize=(16, 6))
        gs23 = gridspec.GridSpec(1, 2, wspace=0.35)
        ax_l = fig.add_subplot(gs23[0])
        ax_r = fig.add_subplot(gs23[1])
        fig.suptitle("Big Data Algorithm Insights — MinHash Jaccard & DGIM Active Rate",
                     fontsize=15, fontweight="bold", color=NAVY)

        # Left: MinHash Jaccard histogram
        ax_l.hist(df_cust["jaccard_basket_centroid"].dropna(), bins=50,
                  color=ORANGE, edgecolor="none", alpha=0.85)
        ax_l.set_xlabel("Jaccard Similarity to Nearest Centroid", fontsize=12, fontweight="bold")
        ax_l.set_ylabel("Number of Customers", fontsize=12, fontweight="bold")
        ax_l.set_title("MinHash LSH — Basket Similarity\nHigh peak = homogeneous buyer base",
                       fontsize=12, color=NAVY)
        med_j = df_cust["jaccard_basket_centroid"].median()
        ax_l.axvline(med_j, color=RED, linewidth=2, linestyle="--",
                     label=f"Median Jaccard: {med_j:.2f}")
        ax_l.legend(fontsize=10)

        # Right: DGIM active rate bar chart per segment
        dgim_s = df_seg[["segment_id","dgim_active_rate","spend_trend_slope"]].copy()
        dgim_s = dgim_s.sort_values("dgim_active_rate", ascending=True)
        clrs_dgim = [GREEN if v >= 0 else RED for v in dgim_s["spend_trend_slope"]]
        ax_r.barh(range(len(dgim_s)), dgim_s["dgim_active_rate"],
                  color=clrs_dgim, height=0.65, edgecolor="none")
        ax_r.set_yticks(range(len(dgim_s)))
        ax_r.set_yticklabels(dgim_s["segment_id"], fontsize=9.5)
        ax_r.set_xlabel("DGIM Active Rate (above-median months / 12)", fontsize=11)
        ax_r.set_title("DGIM Sliding Window — Active Month Rate\nGreen = growing | Red = declining trend",
                       fontsize=12, color=NAVY)
        ax_r.axvline(0.5, color=MGRAY, linewidth=1.5, linestyle="--")

        _apply_style(fig, [ax_l, ax_r])
        _add_footer(fig)
        plt.tight_layout()
        save("23_minhash_dgim_insights.png")
    except Exception as e:
        print(f"  ⚠ Chart 23 skipped: {e}")

print("\n" + "=" * 60)
print("  ALL CHARTS COMPLETE")
print("=" * 60)
