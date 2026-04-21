import os
_HERE = os.path.dirname(os.path.abspath(__file__)) + os.sep
"""
Aramark SRF Spend Data - Visualization Suite
=============================================
Create publication-ready charts from ML analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from matplotlib.ticker import FuncFormatter

OUTPUT_DIR = _HERE + "ml_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
fi = pd.read_csv(f"{OUTPUT_DIR}/feature_importance.csv", index_col=0)
seg_spend = pd.read_csv(f"{OUTPUT_DIR}/segment_spend.csv", index_col=0)
cat_spend = pd.read_csv(f"{OUTPUT_DIR}/category_spend.csv", index_col=0)
state_spend = pd.read_csv(f"{OUTPUT_DIR}/state_spend.csv", index_col=0)
clusters = pd.read_csv(f"{OUTPUT_DIR}/cluster_profiles.csv", index_col=0)
customers = pd.read_csv(f"{OUTPUT_DIR}/customer_clusters.csv")

with open(f"{OUTPUT_DIR}/analysis_report.json", "r") as f:
    report = json.load(f)

# ─────────────────────────────────────────────
# 1. FEATURE IMPORTANCE — Combined ranking
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))
fi_sorted = fi.sort_values("Avg_Rank")
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(fi_sorted)))
bars = ax.barh(range(len(fi_sorted)), fi_sorted["Avg_Rank"], color=colors)
ax.set_yticks(range(len(fi_sorted)))
ax.set_yticklabels([name.replace("_enc", "").replace("_", " ") for name in fi_sorted.index], fontsize=10)
ax.set_xlabel("Average Rank (lower = more important)", fontsize=11, fontweight="bold")
ax.set_title("Feature Importance Consensus\n(RF + Permutation + Mutual Information)", fontsize=13, fontweight="bold")
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.3)

# Add values
for i, (idx, row) in enumerate(fi_sorted.iterrows()):
    ax.text(row["Avg_Rank"] + 0.2, i, f"{row['Avg_Rank']:.1f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_feature_importance_consensus.png", dpi=300, bbox_inches="tight")
print("✓ Saved: 01_feature_importance_consensus.png")
plt.close()

# ─────────────────────────────────────────────
# 2. FEATURE IMPORTANCE — Methods compared
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 8))

methods = ["RF_Importance", "Permutation_Importance", "Mutual_Information"]
titles = ["Random Forest", "Permutation (HGB)", "Mutual Information"]

for ax, method, title in zip(axes, methods, titles):
    top_features = fi[method].nlargest(10)
    colors_grad = plt.cm.viridis(np.linspace(0.2, 0.9, len(top_features)))
    ax.barh(range(len(top_features)), top_features.values, color=colors_grad)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels([name.replace("_enc", "").replace("_", " ") for name in top_features.index], fontsize=9)
    ax.set_xlabel("Importance Score", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_feature_importance_methods.png", dpi=300, bbox_inches="tight")
print("✓ Saved: 02_feature_importance_methods.png")
plt.close()

# ─────────────────────────────────────────────
# 3. CATEGORY SPEND — Pie & bar
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Pie
cat_pct = (cat_spend["sum"] / cat_spend["sum"].sum() * 100).sort_values(ascending=False)
colors_pie = plt.cm.Set3(np.linspace(0, 1, len(cat_pct)))
wedges, texts, autotexts = axes[0].pie(
    cat_pct, labels=cat_pct.index, autopct="%1.1f%%", colors=colors_pie,
    startangle=90, textprops={"fontsize": 9}
)
axes[0].set_title("Category Level 1 — Spend Share", fontsize=12, fontweight="bold")

# Bar
cat_spend_sorted = cat_spend["sum"].sort_values(ascending=False)
colors_bar = plt.cm.Spectral(np.linspace(0.2, 0.8, len(cat_spend_sorted)))
axes[1].bar(range(len(cat_spend_sorted)), cat_spend_sorted.values / 1e6, color=colors_bar)
axes[1].set_xticks(range(len(cat_spend_sorted)))
axes[1].set_xticklabels(cat_spend_sorted.index, rotation=45, ha="right", fontsize=9)
axes[1].set_ylabel("Total Spend ($M)", fontsize=10, fontweight="bold")
axes[1].set_title("Category Level 1 — Total Spend", fontsize=12, fontweight="bold")
axes[1].grid(axis="y", alpha=0.3)

# Add value labels
for i, v in enumerate(cat_spend_sorted.values / 1e6):
    axes[1].text(i, v + 1, f"${v:.1f}M", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_category_breakdown.png", dpi=300, bbox_inches="tight")
print("✓ Saved: 03_category_breakdown.png")
plt.close()

# ─────────────────────────────────────────────
# 4. TOP SEGMENTS — Spend & volume
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

seg_top = seg_spend.nlargest(15, "TotalSpend")

# Total spend
colors_grad = plt.cm.Greens(np.linspace(0.3, 0.9, len(seg_top)))
axes[0].barh(range(len(seg_top)), seg_top["TotalSpend"] / 1e6, color=colors_grad)
axes[0].set_yticks(range(len(seg_top)))
axes[0].set_yticklabels(seg_top.index, fontsize=9)
axes[0].set_xlabel("Total Spend ($M)", fontsize=10, fontweight="bold")
axes[0].set_title("Market Segments — Total Spend", fontsize=12, fontweight="bold")
axes[0].invert_yaxis()
axes[0].grid(axis="x", alpha=0.3)

# Avg spend per transaction
colors_grad2 = plt.cm.Blues(np.linspace(0.3, 0.9, len(seg_top)))
axes[1].barh(range(len(seg_top)), seg_top["AvgSpend"], color=colors_grad2)
axes[1].set_yticks(range(len(seg_top)))
axes[1].set_yticklabels(seg_top.index, fontsize=9)
axes[1].set_xlabel("Avg Spend per Transaction ($)", fontsize=10, fontweight="bold")
axes[1].set_title("Market Segments — Avg Transaction Value", fontsize=12, fontweight="bold")
axes[1].invert_yaxis()
axes[1].grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_top_segments.png", dpi=300, bbox_inches="tight")
print("✓ Saved: 04_top_segments.png")
plt.close()

# ─────────────────────────────────────────────
# 5. TOP STATES — Geographic distribution
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))

state_top = state_spend.nlargest(20, "sum")
colors_states = plt.cm.cool(np.linspace(0.2, 0.8, len(state_top)))
bars = ax.bar(range(len(state_top)), state_top["sum"] / 1e6, color=colors_states)
ax.set_xticks(range(len(state_top)))
ax.set_xticklabels(state_top.index, fontsize=10, fontweight="bold")
ax.set_ylabel("Total Spend ($M)", fontsize=11, fontweight="bold")
ax.set_title("Top 20 States by Spend", fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

# Add value labels
for i, (idx, row) in enumerate(state_top.iterrows()):
    ax.text(i, row["sum"]/1e6 + 0.2, f"${row['sum']/1e6:.1f}M", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_top_states.png", dpi=300, bbox_inches="tight")
print("✓ Saved: 05_top_states.png")
plt.close()

# ─────────────────────────────────────────────
# 6. PARETO CURVE — Customer concentration
# ─────────────────────────────────────────────
cust_spend = customers.groupby("Customer Id")["TotalSpend"].sum().sort_values(ascending=False)
cumsum = cust_spend.cumsum() / cust_spend.sum() * 100
pct_customers = np.arange(1, len(cumsum) + 1) / len(cumsum) * 100

fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(pct_customers, cumsum.values, linewidth=3, color="#2E86AB", label="Actual distribution")
ax.plot([0, 100], [0, 100], "k--", linewidth=2, alpha=0.5, label="Theoretical equality")
ax.fill_between(pct_customers, cumsum.values, pct_customers, alpha=0.2, color="#2E86AB")

# Mark key Pareto points
ax.scatter([1, 10, 20], [24.4, 65, 79.1], s=200, color="red", zorder=5, marker="o")
ax.text(1.5, 24.4 + 3, "Top 1%\n24.4% spend", fontsize=10, fontweight="bold")
ax.text(10.5, 65 + 3, "Top 10%\n65% spend", fontsize=10, fontweight="bold")
ax.text(20.5, 79.1 + 3, "Top 20%\n79.1% spend", fontsize=10, fontweight="bold")

ax.set_xlabel("% of Customers (ranked by spend)", fontsize=11, fontweight="bold")
ax.set_ylabel("% of Total Spend", fontsize=11, fontweight="bold")
ax.set_title("Pareto Distribution — Customer Concentration", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
ax.grid(alpha=0.3)
ax.set_xlim(0, 100)
ax.set_ylim(0, 105)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_pareto_curve.png", dpi=300, bbox_inches="tight")
print("✓ Saved: 06_pareto_curve.png")
plt.close()

# ─────────────────────────────────────────────
# 7. CUSTOMER CLUSTERS — Profile heatmap
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

cluster_data = clusters.copy()
# Normalize for heatmap
cluster_norm = (cluster_data - cluster_data.min()) / (cluster_data.max() - cluster_data.min())

sns.heatmap(
    cluster_norm.T, annot=cluster_data.T.round(0), fmt="g", cmap="YlOrRd",
    cbar_kws={"label": "Normalized Value"}, ax=ax, linewidths=1, linecolor="white"
)
ax.set_xlabel("Cluster", fontsize=11, fontweight="bold")
ax.set_ylabel("Metric", fontsize=11, fontweight="bold")
ax.set_title("Customer Cluster Profiles (5 segments)", fontsize=13, fontweight="bold")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_cluster_heatmap.png", dpi=300, bbox_inches="tight")
print("✓ Saved: 07_cluster_heatmap.png")
plt.close()

# ─────────────────────────────────────────────
# 8. CLUSTER SCATTER — Total Spend vs Transactions
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))

cluster_colors = plt.cm.Set1(np.linspace(0, 1, 5))
for cluster_id in sorted(customers["Cluster"].unique()):
    cluster_data = customers[customers["Cluster"] == cluster_id]
    ax.scatter(
        cluster_data["Transactions"], cluster_data["LogTotalSpend"],
        s=200, alpha=0.6, label=f"Cluster {cluster_id}", color=cluster_colors[cluster_id],
        edgecolors="black", linewidth=1.5
    )

ax.set_xlabel("Number of Transactions", fontsize=11, fontweight="bold")
ax.set_ylabel("Log Total Spend ($)", fontsize=11, fontweight="bold")
ax.set_title("Customer Clusters — Transaction Volume vs Spend", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="upper left")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_cluster_scatter.png", dpi=300, bbox_inches="tight")
print("✓ Saved: 08_cluster_scatter.png")
plt.close()

# ─────────────────────────────────────────────
# 9. SPEND DISTRIBUTION — Log-normal fit
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

spend_all = customers["TotalSpend"]

# Histogram (log scale)
axes[0].hist(np.log1p(spend_all), bins=100, color="#1f77b4", alpha=0.7, edgecolor="black")
axes[0].set_xlabel("Log(Spend)", fontsize=10, fontweight="bold")
axes[0].set_ylabel("Frequency", fontsize=10, fontweight="bold")
axes[0].set_title("Spend Distribution (Log-Transformed)", fontsize=11, fontweight="bold")
axes[0].grid(axis="y", alpha=0.3)

# Box plot by cluster
cluster_names = [f"C{i}" for i in range(5)]
box_data = [customers[customers["Cluster"] == i]["TotalSpend"].values for i in range(5)]
bp = axes[1].boxplot(box_data, labels=cluster_names, patch_artist=True)
for patch, color in zip(bp["boxes"], cluster_colors):
    patch.set_facecolor(color)
axes[1].set_ylabel("Total Spend ($)", fontsize=10, fontweight="bold")
axes[1].set_xlabel("Cluster", fontsize=10, fontweight="bold")
axes[1].set_title("Spend Distribution by Cluster", fontsize=11, fontweight="bold")
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_spend_distribution.png", dpi=300, bbox_inches="tight")
print("✓ Saved: 09_spend_distribution.png")
plt.close()

# ─────────────────────────────────────────────
# 10. MONTHLY TRENDS
# ─────────────────────────────────────────────
if "monthly_trend" in report:
    monthly = report["monthly_trend"]
    months_list = sorted(monthly.keys(), key=lambda x: float(x))
    values = [monthly[m] for m in months_list]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(months_list, np.array(values) / 1e6, marker="o", linewidth=3, markersize=10, color="#FF6B6B")
    ax.fill_between(range(len(months_list)), np.array(values) / 1e6, alpha=0.3, color="#FF6B6B")

    ax.set_xlabel("Year-Month", fontsize=11, fontweight="bold")
    ax.set_ylabel("Total Spend ($M)", fontsize=11, fontweight="bold")
    ax.set_title("Spend Trend — Monthly (2025)", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)

    # Annotate peaks
    max_idx = np.argmax(values)
    ax.scatter([max_idx], [values[max_idx] / 1e6], s=300, color="red", zorder=5, marker="*")
    ax.text(max_idx, values[max_idx] / 1e6 + 0.3, f"Peak: {months_list[max_idx]}", fontsize=10, fontweight="bold")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/10_monthly_trends.png", dpi=300, bbox_inches="tight")
    print("✓ Saved: 10_monthly_trends.png")
    plt.close()

# ─────────────────────────────────────────────
# 11. BUSINESS ENTITY TYPE COMPARISON
# ─────────────────────────────────────────────
if "biz_type_spend" in report:
    biz_data = report["biz_type_spend"]
    biz_types = list(biz_data["sum"].keys())[:8]
    biz_spend = [biz_data["sum"][b] / 1e6 for b in biz_types]
    biz_avg = [biz_data["mean"][b] for b in biz_types]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    colors_biz = plt.cm.Pastel1(np.linspace(0, 1, len(biz_types)))
    axes[0].bar(biz_types, biz_spend, color=colors_biz, edgecolor="black", linewidth=1.5)
    axes[0].set_ylabel("Total Spend ($M)", fontsize=10, fontweight="bold")
    axes[0].set_title("Business Entity Type — Total Spend", fontsize=11, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(biz_types, biz_avg, color=colors_biz, edgecolor="black", linewidth=1.5)
    axes[1].set_ylabel("Avg Spend per Transaction ($)", fontsize=10, fontweight="bold")
    axes[1].set_title("Business Entity Type — Avg Transaction", fontsize=11, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/11_business_entity_type.png", dpi=300, bbox_inches="tight")
    print("✓ Saved: 11_business_entity_type.png")
    plt.close()

print("\n" + "="*60)
print("✓ ALL VISUALIZATIONS COMPLETE")
print("="*60)
print(f"Location: {OUTPUT_DIR}/")
print("\nGenerated files:")
print("  01_feature_importance_consensus.png")
print("  02_feature_importance_methods.png")
print("  03_category_breakdown.png")
print("  04_top_segments.png")
print("  05_top_states.png")
print("  06_pareto_curve.png")
print("  07_cluster_heatmap.png")
print("  08_cluster_scatter.png")
print("  09_spend_distribution.png")
print("  10_monthly_trends.png")
print("  11_business_entity_type.png")
