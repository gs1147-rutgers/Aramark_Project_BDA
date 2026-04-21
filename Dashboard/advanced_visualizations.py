import os
_HERE = os.path.dirname(os.path.abspath(__file__)) + os.sep
"""
Advanced ML Results - Comprehensive Visualizations & Summary
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle

OUTPUT_DIR = _HERE + "ml_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# Load all results
comparison = pd.read_csv(f"{OUTPUT_DIR}/model_comparison.csv")
fi_advanced = pd.read_csv(f"{OUTPUT_DIR}/feature_importance_advanced.csv", index_col=0)

with open(f"{OUTPUT_DIR}/advanced_models_report.json", "r") as f:
    report = json.load(f)

print("Creating Advanced ML Visualizations...\n")

# ─────────────────────────────────────────────
# 1. MODEL COMPARISON — Performance across all models
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# R² comparison
colors_bar = plt.cm.Set2(np.linspace(0, 1, len(comparison)))
axes[0].bar(range(len(comparison)), comparison["R²"], color=colors_bar, edgecolor="black", linewidth=1.5)
axes[0].set_xticks(range(len(comparison)))
axes[0].set_xticklabels(comparison["Model"], rotation=45, ha="right", fontsize=9)
axes[0].set_ylabel("R² Score", fontsize=11, fontweight="bold")
axes[0].set_title("Model Performance — R² Score", fontsize=12, fontweight="bold")
axes[0].grid(axis="y", alpha=0.3)
axes[0].set_ylim(0, max(comparison["R²"]) * 1.15)

# Add value labels
for i, v in enumerate(comparison["R²"]):
    axes[0].text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=8, fontweight="bold")

# MAE comparison
axes[1].bar(range(len(comparison)), comparison["MAE"], color=colors_bar, edgecolor="black", linewidth=1.5)
axes[1].set_xticks(range(len(comparison)))
axes[1].set_xticklabels(comparison["Model"], rotation=45, ha="right", fontsize=9)
axes[1].set_ylabel("MAE (log-spend units)", fontsize=11, fontweight="bold")
axes[1].set_title("Model Performance — MAE", fontsize=12, fontweight="bold")
axes[1].grid(axis="y", alpha=0.3)
axes[1].set_ylim(min(comparison["MAE"]) * 0.9, max(comparison["MAE"]) * 1.1)

for i, v in enumerate(comparison["MAE"]):
    axes[1].text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=8, fontweight="bold")

# RMSE comparison
axes[2].bar(range(len(comparison)), comparison["RMSE"], color=colors_bar, edgecolor="black", linewidth=1.5)
axes[2].set_xticks(range(len(comparison)))
axes[2].set_xticklabels(comparison["Model"], rotation=45, ha="right", fontsize=9)
axes[2].set_ylabel("RMSE (log-spend units)", fontsize=11, fontweight="bold")
axes[2].set_title("Model Performance — RMSE", fontsize=12, fontweight="bold")
axes[2].grid(axis="y", alpha=0.3)

for i, v in enumerate(comparison["RMSE"]):
    axes[2].text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=8, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/12_model_comparison_all_metrics.png", dpi=300, bbox_inches="tight")
print("✓ 12_model_comparison_all_metrics.png")
plt.close()

# ─────────────────────────────────────────────
# 2. FEATURE IMPORTANCE — Advanced methods comparison
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 10))

fi_top = fi_advanced.iloc[:15].fillna(0)
x = np.arange(len(fi_top))
width = 0.25

bars1 = ax.barh(x - width, fi_top["XGBoost"], width, label="XGBoost", alpha=0.8)
bars2 = ax.barh(x, fi_top["LightGBM"], width, label="LightGBM", alpha=0.8)
bars3 = ax.barh(x + width, fi_top["SHAP"], width, label="SHAP", alpha=0.8)

ax.set_yticks(x)
ax.set_yticklabels([name.replace("_enc", "").replace("_", " ") for name in fi_top.index], fontsize=10)
ax.set_xlabel("Importance Score", fontsize=11, fontweight="bold")
ax.set_title("Feature Importance — Cross-Method Comparison", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/13_feature_importance_xgb_lgb_shap.png", dpi=300, bbox_inches="tight")
print("✓ 13_feature_importance_xgb_lgb_shap.png")
plt.close()

# ─────────────────────────────────────────────
# 3. HYPERPARAMETER TUNING IMPACT
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

models = ["Random Forest\n(Baseline)", "Hist Gradient\nBoost", "XGBoost\n(Default)", "LightGBM\n(Default)",
          "Stacking\nEnsemble", "XGBoost\n(Tuned)"]
r2_scores = [0.2501, 0.2818, 0.2653, 0.2720, 0.2511, 0.2951]
colors_gradient = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(models)))

bars = ax.bar(range(len(models)), r2_scores, color=colors_gradient, edgecolor="black", linewidth=2)
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, fontsize=10)
ax.set_ylabel("R² Score", fontsize=11, fontweight="bold")
ax.set_title("Model Evolution — Hyperparameter Tuning Impact", fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0.24, 0.31)

# Highlight best model
best_idx = np.argmax(r2_scores)
bars[best_idx].set_edgecolor("red")
bars[best_idx].set_linewidth(3)

# Add value labels with improvement
for i, v in enumerate(r2_scores):
    if i == best_idx:
        improvement = (v - 0.2501) / 0.2501 * 100
        ax.text(i, v + 0.005, f"{v:.4f}\n(+{improvement:.1f}%)", ha="center", fontsize=10,
                fontweight="bold", color="red")
    else:
        ax.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/14_hyperparameter_tuning_impact.png", dpi=300, bbox_inches="tight")
print("✓ 14_hyperparameter_tuning_impact.png")
plt.close()

# ─────────────────────────────────────────────
# 4. CROSS-VALIDATION PERFORMANCE
# ─────────────────────────────────────────────
cv_data = report["cross_validation"]
cv_df = pd.DataFrame({
    "Model": list(cv_data.keys()),
    "Mean R²": [cv_data[m]["mean_r2"] for m in cv_data.keys()],
    "Std R²": [cv_data[m]["std_r2"] for m in cv_data.keys()],
})

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(cv_df))
colors_cv = plt.cm.Set1(np.linspace(0, 1, len(cv_df)))

ax.bar(x, cv_df["Mean R²"], color=colors_cv, edgecolor="black", linewidth=1.5, alpha=0.7)
ax.errorbar(x, cv_df["Mean R²"], yerr=cv_df["Std R²"], fmt="none", color="black",
            capsize=5, capthick=2, linewidth=2, label="±1 Std Dev")

ax.set_xticks(x)
ax.set_xticklabels(cv_df["Model"], fontsize=10)
ax.set_ylabel("R² Score", fontsize=11, fontweight="bold")
ax.set_title("5-Fold Cross-Validation Performance (More Robust Estimate)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

for i, row in cv_df.iterrows():
    ax.text(i, row["Mean R²"] + row["Std R²"] + 0.002, f"{row['Mean R²']:.4f}",
            ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/15_cross_validation_performance.png", dpi=300, bbox_inches="tight")
print("✓ 15_cross_validation_performance.png")
plt.close()

# ─────────────────────────────────────────────
# 5. META-LEARNER WEIGHTS (Stacking)
# ─────────────────────────────────────────────
meta_weights = report["stacking"]["meta_weights"]
weights_df = pd.DataFrame({
    "Base Model": list(meta_weights.keys()),
    "Weight": list(meta_weights.values())
}).sort_values("Weight", ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))

colors_weights = ["green" if w > 0 else "red" for w in weights_df["Weight"]]
bars = ax.barh(range(len(weights_df)), weights_df["Weight"], color=colors_weights,
               edgecolor="black", linewidth=1.5, alpha=0.7)

ax.set_yticks(range(len(weights_df)))
ax.set_yticklabels(weights_df["Base Model"], fontsize=11, fontweight="bold")
ax.set_xlabel("Meta-Learner Weight", fontsize=11, fontweight="bold")
ax.set_title("Stacking Ensemble — Base Model Weights", fontsize=13, fontweight="bold")
ax.axvline(x=0, color="black", linestyle="-", linewidth=1)
ax.grid(axis="x", alpha=0.3)

for i, (idx, row) in enumerate(weights_df.iterrows()):
    weight = row["Weight"]
    ax.text(weight + 0.05 if weight > 0 else weight - 0.05, i, f"{weight:.4f}",
            va="center", ha="left" if weight > 0 else "right", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/16_stacking_meta_weights.png", dpi=300, bbox_inches="tight")
print("✓ 16_stacking_meta_weights.png")
plt.close()

# ─────────────────────────────────────────────
# 6. BEST HYPERPARAMETERS (Optuna Results)
# ─────────────────────────────────────────────
best_params = report["xgboost_tuned"]["best_params"]
param_names = list(best_params.keys())
param_values = list(best_params.values())

fig, ax = plt.subplots(figsize=(10, 6))

colors_params = plt.cm.cool(np.linspace(0.2, 0.8, len(param_names)))
bars = ax.bar(range(len(param_names)), param_values, color=colors_params, edgecolor="black", linewidth=1.5)

ax.set_xticks(range(len(param_names)))
ax.set_xticklabels(param_names, fontsize=10, fontweight="bold")
ax.set_ylabel("Parameter Value", fontsize=11, fontweight="bold")
ax.set_title("Optuna-Optimized XGBoost Hyperparameters", fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

for i, v in enumerate(param_values):
    ax.text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/17_best_hyperparameters.png", dpi=300, bbox_inches="tight")
print("✓ 17_best_hyperparameters.png")
plt.close()

# ─────────────────────────────────────────────
# 7. TOP FEATURES — Consensus across methods
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))

# Rank each feature in each method, then average rank
xgb_fi = report["xgboost"]["top_features"]
lgb_fi = report["lightgbm"]["top_features"]
shap_fi = report["shap_importance"]

all_features = set(xgb_fi.keys()) | set(lgb_fi.keys()) | set(shap_fi.keys())
feature_rankings = {}

for feat in all_features:
    xgb_rank = list(xgb_fi.keys()).index(feat) + 1 if feat in xgb_fi else 15
    lgb_rank = list(lgb_fi.keys()).index(feat) + 1 if feat in lgb_fi else 15
    shap_rank = list(shap_fi.keys()).index(feat) + 1 if feat in shap_fi else 15
    avg_rank = (xgb_rank + lgb_rank + shap_rank) / 3
    feature_rankings[feat] = avg_rank

top_features_consensus = dict(sorted(feature_rankings.items(), key=lambda x: x[1])[:10])

feat_names = [f.replace("_enc", "").replace("_", " ") for f in top_features_consensus.keys()]
feat_ranks = list(top_features_consensus.values())

colors_consensus = plt.cm.Spectral(np.linspace(0.2, 0.8, len(feat_names)))
ax.barh(range(len(feat_names)), feat_ranks, color=colors_consensus, edgecolor="black", linewidth=1.5)
ax.set_yticks(range(len(feat_names)))
ax.set_yticklabels(feat_names, fontsize=10)
ax.set_xlabel("Average Rank (lower = more important)", fontsize=11, fontweight="bold")
ax.set_title("Top Features — Consensus Across XGBoost, LightGBM, SHAP", fontsize=13, fontweight="bold")
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.3)

for i, v in enumerate(feat_ranks):
    ax.text(v + 0.2, i, f"{v:.2f}", va="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/18_top_features_consensus_advanced.png", dpi=300, bbox_inches="tight")
print("✓ 18_top_features_consensus_advanced.png")
plt.close()

# ─────────────────────────────────────────────
# 8. MODEL IMPROVEMENT WATERFALL
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

models_waterfall = ["Baseline\n(RF)", "Hist Grad\nBoost", "XGBoost", "LightGBM", "Stacking", "XGBoost\n(Tuned)"]
r2_waterfall = [0.2501, 0.2818, 0.2653, 0.2720, 0.2511, 0.2951]
improvements = [0, 0.0317, 0.0152, 0.0219, 0.0010, 0.0450]

cumulative = [0.2501]
for imp in improvements[1:]:
    cumulative.append(cumulative[-1] + imp)

x_pos = np.arange(len(models_waterfall))

# Plot bars
for i in range(len(models_waterfall)):
    if i == 0:
        ax.bar(i, r2_waterfall[i], color="steelblue", edgecolor="black", linewidth=1.5, width=0.6)
    else:
        if r2_waterfall[i] >= cumulative[i-1]:
            ax.bar(i, improvements[i], bottom=cumulative[i-1], color="lightgreen",
                   edgecolor="black", linewidth=1.5, width=0.6)
        else:
            ax.bar(i, improvements[i], bottom=r2_waterfall[i], color="lightcoral",
                   edgecolor="black", linewidth=1.5, width=0.6)

ax.set_xticks(x_pos)
ax.set_xticklabels(models_waterfall, fontsize=10)
ax.set_ylabel("R² Score", fontsize=11, fontweight="bold")
ax.set_title("Model Improvement Journey — Waterfall", fontsize=13, fontweight="bold")
ax.set_ylim(0.24, 0.31)
ax.grid(axis="y", alpha=0.3)

# Add value labels
for i, (model, r2) in enumerate(zip(models_waterfall, r2_waterfall)):
    ax.text(i, r2 + 0.003, f"{r2:.4f}", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/19_model_improvement_waterfall.png", dpi=300, bbox_inches="tight")
print("✓ 19_model_improvement_waterfall.png")
plt.close()

print("\n" + "="*70)
print("✓ ALL ADVANCED ML VISUALIZATIONS CREATED")
print("="*70)
print("\nGenerated visualizations:")
print("  12_model_comparison_all_metrics.png")
print("  13_feature_importance_xgb_lgb_shap.png")
print("  14_hyperparameter_tuning_impact.png")
print("  15_cross_validation_performance.png")
print("  16_stacking_meta_weights.png")
print("  17_best_hyperparameters.png")
print("  18_top_features_consensus_advanced.png")
print("  19_model_improvement_waterfall.png")
