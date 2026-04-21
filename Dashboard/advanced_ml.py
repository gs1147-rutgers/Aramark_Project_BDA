import os
_HERE = os.path.dirname(os.path.abspath(__file__)) + os.sep
"""
Aramark SRF Spend Data - Advanced ML Models
============================================
Gradient Boosting, Stacking, Neural Networks, Hyperparameter Tuning, SHAP
"""

import pandas as pd
import numpy as np
import warnings
import json
import pickle
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

OUTPUT_DIR = _HERE + "ml_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("ADVANCED ML MODELS — Aramark SRF Spend Analysis")
print("=" * 70)

# ─────────────────────────────────────────────
# LOAD PREPROCESSED DATA
# ─────────────────────────────────────────────
print("\n[SETUP] Loading preprocessed data...")

FILE = _HERE + "Andrew_Meszaros_SRF_2026-04-01-0936.csv"
SAMPLE_SIZE = 500_000
CHUNK_SIZE  = 200_000

chunks = []
total_rows = 0
for chunk in pd.read_csv(FILE, chunksize=CHUNK_SIZE, low_memory=False):
    total_rows += len(chunk)
    chunks.append(chunk)

all_data = pd.concat(chunks, ignore_index=True)
all_data["Spend"] = pd.to_numeric(all_data["Spend Random Factor"], errors="coerce")
all_data = all_data.dropna(subset=["Spend"])
all_data = all_data[all_data["Spend"] > 0]

strat_col = "Customer Market Segment Id"
segment_counts = all_data[strat_col].value_counts()
fracs = {seg: min(SAMPLE_SIZE / len(all_data), 1.0) for seg in segment_counts.index}
sampled = all_data.groupby(strat_col, group_keys=False).apply(
    lambda g: g.sample(frac=fracs.get(g.name, 1.0), random_state=42)
)
if len(sampled) > SAMPLE_SIZE:
    sampled = sampled.sample(SAMPLE_SIZE, random_state=42)

df = sampled.reset_index(drop=True)

# Feature engineering
cat_cols = [
    "Business Entity Type", "Customer Market Segment Id",
    "Category Level 1", "Category Level 2", "Category Level 3",
    "State", "Distributor Group", "Ecommerce Status"
]

df_enc = df.copy()
df_enc["Spend"] = df["Spend"]
df_enc["LogSpend"] = np.log1p(df["Spend"])

encoders = {}
for col in cat_cols:
    if col in df_enc.columns:
        df_enc[col + "_enc"] = df_enc[col].fillna("UNKNOWN")
        le = LabelEncoder()
        df_enc[col + "_enc"] = le.fit_transform(df_enc[col + "_enc"].astype(str))
        encoders[col] = le

df_enc["HasRooms"] = df_enc["Number of Rooms"].notna().astype(int)
df_enc["RoomCount"] = pd.to_numeric(df_enc["Number of Rooms"], errors="coerce").fillna(0)
df_enc["IsEcomActive"] = (df_enc.get("Ecommerce Status", "").astype(str).str.lower() == "active").astype(int)
df_enc["CategoryDepth"] = (
    df_enc["Category Level 1"].notna().astype(int) +
    df_enc["Category Level 2"].notna().astype(int) +
    df_enc["Category Level 3"].notna().astype(int) +
    (df_enc.get("Category Level 4", pd.Series(dtype=str)).notna()).astype(int)
)

if "Year Month" in df_enc.columns:
    df_enc["YearMonth"] = pd.to_numeric(df_enc["Year Month"], errors="coerce")
    df_enc["Month"] = df_enc["YearMonth"].astype(str).str[-2:].replace("", "0").astype(float)

feature_cols = [c for c in df_enc.columns if c.endswith("_enc")] + [
    "HasRooms", "RoomCount", "IsEcomActive", "CategoryDepth"
]
if "Month" in df_enc.columns:
    feature_cols.append("Month")

X = df_enc[feature_cols].fillna(0)
y = df_enc["LogSpend"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train2, X_val, y_train2, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print(f"  ✓ Data loaded: {X.shape[0]:,} samples, {X.shape[1]} features")
print(f"  ✓ Train: {X_train2.shape[0]:,} | Val: {X_val.shape[0]:,} | Test: {X_test.shape[0]:,}")

# ─────────────────────────────────────────────
# 1. XGBOOST — Gradient Boosting
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("[1] XGBoost — Gradient Boosting")
print("=" * 70)

import xgboost as xgb

xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squaredlogerror",
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=20,
    verbosity=0
)

xgb_model.fit(
    X_train2, y_train2,
    eval_set=[(X_val, y_val)],
    verbose=False
)

y_pred_xgb = xgb_model.predict(X_test)
xgb_r2 = r2_score(y_test, y_pred_xgb)
xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

print(f"  R²:   {xgb_r2:.4f}")
print(f"  MAE:  {xgb_mae:.4f} (log-spend units)")
print(f"  RMSE: {xgb_rmse:.4f}")

# Feature importance
xgb_fi = pd.Series(
    xgb_model.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

print("\n  Top 10 XGBoost Features:")
for i, (feat, val) in enumerate(xgb_fi.head(10).items(), 1):
    print(f"    {i}. {feat:30s}: {val:.4f}")

with open(f"{OUTPUT_DIR}/xgb_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)

# ─────────────────────────────────────────────
# 2. LIGHTGBM — LightGradient Boosting
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("[2] LightGBM — Light Gradient Boosting")
print("=" * 70)

import lightgbm as lgb

lgb_model = lgb.LGBMRegressor(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgb_model.fit(
    X_train2, y_train2,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(20)]
)

y_pred_lgb = lgb_model.predict(X_test)
lgb_r2 = r2_score(y_test, y_pred_lgb)
lgb_mae = mean_absolute_error(y_test, y_pred_lgb)
lgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lgb))

print(f"  R²:   {lgb_r2:.4f}")
print(f"  MAE:  {lgb_mae:.4f} (log-spend units)")
print(f"  RMSE: {lgb_rmse:.4f}")

# Feature importance
lgb_fi = pd.Series(
    lgb_model.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

print("\n  Top 10 LightGBM Features:")
for i, (feat, val) in enumerate(lgb_fi.head(10).items(), 1):
    print(f"    {i}. {feat:30s}: {val:.4f}")

with open(f"{OUTPUT_DIR}/lgb_model.pkl", "wb") as f:
    pickle.dump(lgb_model, f)

# ─────────────────────────────────────────────
# 3. STACKING ENSEMBLE — Combine predictions
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("[3] Stacking Ensemble — Meta-learner")
print("=" * 70)

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

# Base learners
base_models = [
    ("xgb", xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)),
    ("lgb", lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1)),
    ("rf", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
    ("gb", GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)),
]

# Generate meta-features
meta_features_train = np.zeros((X_train2.shape[0], len(base_models)))
meta_features_val = np.zeros((X_val.shape[0], len(base_models)))
meta_features_test = np.zeros((X_test.shape[0], len(base_models)))

print("  Training base learners...")
for i, (name, model) in enumerate(base_models):
    print(f"    {i+1}/4: {name}...", end=" ", flush=True)
    model.fit(X_train2, y_train2)
    meta_features_train[:, i] = model.predict(X_train2)
    meta_features_val[:, i] = model.predict(X_val)
    meta_features_test[:, i] = model.predict(X_test)
    print("✓")

# Meta-learner (Ridge regression on base predictions)
meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_features_train, y_train2)

y_pred_stack = meta_model.predict(meta_features_test)
stack_r2 = r2_score(y_test, y_pred_stack)
stack_mae = mean_absolute_error(y_test, y_pred_stack)
stack_rmse = np.sqrt(mean_squared_error(y_test, y_pred_stack))

print(f"\n  Stacking Ensemble Performance:")
print(f"  R²:   {stack_r2:.4f}")
print(f"  MAE:  {stack_mae:.4f} (log-spend units)")
print(f"  RMSE: {stack_rmse:.4f}")

meta_weights = pd.Series(meta_model.coef_, index=[n for n, _ in base_models])
print(f"\n  Meta-learner weights:")
for name, weight in meta_weights.items():
    print(f"    {name:6s}: {weight:.4f}")

with open(f"{OUTPUT_DIR}/stack_model.pkl", "wb") as f:
    pickle.dump((base_models, meta_model), f)

# ─────────────────────────────────────────────
# 4. CROSS-VALIDATION — Robust performance
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("[4] Cross-Validation (5-Fold) — Robust Performance")
print("=" * 70)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

models_cv = {
    "XGBoost": xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1),
    "LightGBM": lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1),
    "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    "GradBoost": GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42),
}

cv_results = {}
print("  Computing cross-validation scores...")
for name, model in models_cv.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
    cv_results[name] = scores
    print(f"    {name:15s}: {scores.mean():.4f} ± {scores.std():.4f}")

# ─────────────────────────────────────────────
# 5. SHAP VALUES — Model explainability
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("[5] SHAP Values — Model Explainability")
print("=" * 70)

import shap

print("  Computing SHAP values for XGBoost...")
explainer = shap.TreeExplainer(xgb_model)

# Use sample for speed
sample_idx = np.random.choice(X_test.shape[0], min(2000, X_test.shape[0]), replace=False)
X_sample = X_test.iloc[sample_idx]
shap_values = explainer.shap_values(X_sample)

# Get mean absolute SHAP
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_fi = pd.Series(mean_abs_shap, index=feature_cols).sort_values(ascending=False)

print("\n  Top 10 SHAP Features (mean |impact|):")
for i, (feat, val) in enumerate(shap_fi.head(10).items(), 1):
    print(f"    {i}. {feat:30s}: {val:.4f}")

# Save SHAP
shap_summary = {
    "feature": feature_cols,
    "shap_importance": mean_abs_shap.tolist()
}

# ─────────────────────────────────────────────
# 6. HYPERPARAMETER TUNING — Optuna
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("[6] Hyperparameter Tuning (Optuna) — XGBoost")
print("=" * 70)

import optuna
from optuna.pruners import MedianPruner

def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }

    model = xgb.XGBRegressor(
        n_estimators=100,
        objective="reg:squaredlogerror",
        random_state=42,
        n_jobs=-1,
        **params
    )

    scores = cross_val_score(model, X_train2, y_train2, cv=3, scoring="r2", n_jobs=-1)
    return scores.mean()

print("  Running 30 trials...")
study = optuna.create_study(direction="maximize", pruner=MedianPruner())
study.optimize(objective, n_trials=30, show_progress_bar=False)

best_params = study.best_params
best_score = study.best_value

print(f"  Best R² (CV): {best_score:.4f}")
print(f"  Best parameters:")
for k, v in best_params.items():
    print(f"    {k:20s}: {v}")

# Train best model
xgb_tuned = xgb.XGBRegressor(
    n_estimators=200,
    objective="reg:squaredlogerror",
    random_state=42,
    n_jobs=-1,
    **best_params
)
xgb_tuned.fit(X_train2, y_train2)
y_pred_tuned = xgb_tuned.predict(X_test)
tuned_r2 = r2_score(y_test, y_pred_tuned)
tuned_mae = mean_absolute_error(y_test, y_pred_tuned)

print(f"\n  Tuned XGBoost on Test Set:")
print(f"  R²:  {tuned_r2:.4f} (improvement: {tuned_r2 - xgb_r2:+.4f})")
print(f"  MAE: {tuned_mae:.4f} (improvement: {tuned_mae - xgb_mae:+.4f})")

# ─────────────────────────────────────────────
# 7. MODEL COMPARISON TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("[7] Model Comparison — All Methods")
print("=" * 70)

comparison = pd.DataFrame({
    "Model": ["Random Forest (baseline)", "Hist Gradient Boost", "XGBoost", "LightGBM", "Stacking Ensemble", "XGBoost (tuned)"],
    "R²": [0.2501, 0.2818, xgb_r2, lgb_r2, stack_r2, tuned_r2],
    "MAE": [0.8481, 0.8274, xgb_mae, lgb_mae, stack_mae, tuned_mae],
    "RMSE": [1.1627, 1.1355, xgb_rmse, lgb_rmse, np.sqrt(mean_squared_error(y_test, y_pred_stack)),
             np.sqrt(mean_squared_error(y_test, y_pred_tuned))]
})

print("\n" + comparison.to_string(index=False))
comparison.to_csv(f"{OUTPUT_DIR}/model_comparison.csv", index=False)

# ─────────────────────────────────────────────
# 8. FEATURE INTERACTION ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("[8] Feature Interactions — SHAP Dependence")
print("=" * 70)

top_features = xgb_fi.head(5).index.tolist()
print(f"  Top 5 features for interaction analysis:")
for i, f in enumerate(top_features, 1):
    print(f"    {i}. {f}")

# ─────────────────────────────────────────────
# 9. RESIDUAL ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("[9] Residual Analysis — XGBoost")
print("=" * 70)

residuals = y_test - y_pred_xgb
print(f"  Mean residual: {residuals.mean():.6f} (should be ~0)")
print(f"  Std residual:  {residuals.std():.4f}")
print(f"  Min residual:  {residuals.min():.4f}")
print(f"  Max residual:  {residuals.max():.4f}")
print(f"  Skewness:      {pd.Series(residuals).skew():.4f}")
print(f"  Kurtosis:      {pd.Series(residuals).kurtosis():.4f}")

# ─────────────────────────────────────────────
# 10. SAVE COMPREHENSIVE REPORT
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("[10] Saving Results")
print("=" * 70)

report_advanced = {
    "xgboost": {
        "r2": xgb_r2,
        "mae": xgb_mae,
        "rmse": xgb_rmse,
        "top_features": xgb_fi.head(15).to_dict()
    },
    "lightgbm": {
        "r2": lgb_r2,
        "mae": lgb_mae,
        "rmse": lgb_rmse,
        "top_features": lgb_fi.head(15).to_dict()
    },
    "stacking": {
        "r2": stack_r2,
        "mae": stack_mae,
        "rmse": stack_rmse,
        "meta_weights": meta_weights.to_dict()
    },
    "xgboost_tuned": {
        "r2": tuned_r2,
        "mae": tuned_mae,
        "best_params": {str(k): v for k, v in best_params.items()},
        "improvement": tuned_r2 - xgb_r2
    },
    "cross_validation": {
        model: {
            "mean_r2": scores.mean(),
            "std_r2": scores.std(),
            "scores": scores.tolist()
        }
        for model, scores in cv_results.items()
    },
    "shap_importance": shap_fi.head(15).to_dict(),
    "residual_analysis": {
        "mean": float(residuals.mean()),
        "std": float(residuals.std()),
        "min": float(residuals.min()),
        "max": float(residuals.max())
    }
}

# Save JSON
with open(f"{OUTPUT_DIR}/advanced_models_report.json", "w") as f:
    json.dump(report_advanced, f, indent=2)

# Save feature importance comparison
fi_comparison = pd.DataFrame({
    "XGBoost": xgb_fi,
    "LightGBM": lgb_fi,
    "SHAP": shap_fi
}).fillna(0)
fi_comparison.to_csv(f"{OUTPUT_DIR}/feature_importance_advanced.csv")

print(f"  ✓ advanced_models_report.json")
print(f"  ✓ model_comparison.csv")
print(f"  ✓ feature_importance_advanced.csv")
print(f"  ✓ xgb_model.pkl")
print(f"  ✓ lgb_model.pkl")
print(f"  ✓ stack_model.pkl")

print("\n" + "=" * 70)
print("✓ ADVANCED ML MODELS COMPLETE")
print("=" * 70)
print("\nKey Findings:")
print(f"  • Best model: XGBoost (tuned) R² = {tuned_r2:.4f}")
print(f"  • Improvement over baseline: {(tuned_r2 - 0.2501)*100:.1f}%")
print(f"  • Best performing ensemble: Stacking (R² = {stack_r2:.4f})")
print(f"  • Top predictive feature: {xgb_fi.index[0]}")
