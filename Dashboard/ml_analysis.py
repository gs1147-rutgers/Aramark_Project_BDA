import os
_HERE = os.path.dirname(os.path.abspath(__file__)) + os.sep
"""
Aramark SRF Spend Data - ML Feature Analysis
============================================
Objective: Discover key features, patterns, and drivers in spend data
Approach: Multi-algorithm analysis on stratified sample
"""

import pandas as pd
import numpy as np
import warnings
import json
from collections import Counter

warnings.filterwarnings("ignore")

FILE = _HERE + "Andrew_Meszaros_SRF_2026-04-01-0936.csv"
OUTPUT_DIR = _HERE + "ml_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. STRATIFIED SAMPLING  (500k rows)
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Stratified sampling (500k rows)")
print("=" * 60)

# reservoir-sampling style: read in chunks, keep every Nth row
SAMPLE_SIZE = 500_000
CHUNK_SIZE  = 200_000

chunks = []
total_rows = 0
for chunk in pd.read_csv(FILE, chunksize=CHUNK_SIZE, low_memory=False):
    total_rows += len(chunk)
    chunks.append(chunk)
    if len(chunks) % 10 == 0:
        print(f"  Read {total_rows:,} rows so far…")

print(f"  Total rows in file: {total_rows:,}")

# combine and stratified-sample by market segment
all_data = pd.concat(chunks, ignore_index=True)
print(f"  Full dataset shape: {all_data.shape}")

# clean spend column
all_data["Spend"] = pd.to_numeric(all_data["Spend Random Factor"], errors="coerce")
all_data = all_data.dropna(subset=["Spend"])
all_data = all_data[all_data["Spend"] > 0]

# stratified sample by market segment
strat_col = "Customer Market Segment Id"
if strat_col in all_data.columns:
    segment_counts = all_data[strat_col].value_counts()
    fracs = {seg: min(SAMPLE_SIZE / len(all_data), 1.0) for seg in segment_counts.index}
    sampled = all_data.groupby(strat_col, group_keys=False).apply(
        lambda g: g.sample(frac=fracs.get(g.name, 1.0), random_state=42)
    )
    if len(sampled) > SAMPLE_SIZE:
        sampled = sampled.sample(SAMPLE_SIZE, random_state=42)
else:
    sampled = all_data.sample(min(SAMPLE_SIZE, len(all_data)), random_state=42)

df = sampled.reset_index(drop=True)
print(f"  Sample shape: {df.shape}\n")

# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 2 — Exploratory Data Analysis")
print("=" * 60)

report = {}

# Basic stats
report["spend_stats"] = df["Spend"].describe().to_dict()
print("\nSpend distribution:")
for k, v in report["spend_stats"].items():
    print(f"  {k:>8s}: ${v:,.2f}")

# Log-transform spend (heavily right-skewed)
df["LogSpend"] = np.log1p(df["Spend"])

# Spend by market segment
seg_spend = df.groupby("Customer Market Segment Id")["Spend"].agg(["sum","mean","count"])
seg_spend.columns = ["TotalSpend","AvgSpend","Transactions"]
seg_spend = seg_spend.sort_values("TotalSpend", ascending=False)
report["segment_spend"] = seg_spend.head(20).to_dict()
print("\nTop 10 Market Segments by Total Spend:")
print(seg_spend.head(10).to_string())

# Spend by Category Level 1
cat_spend = df.groupby("Category Level 1")["Spend"].agg(["sum","mean","count"])
cat_spend = cat_spend.sort_values("sum", ascending=False)
report["category_l1_spend"] = cat_spend.to_dict()
print("\nCategory Level 1 Spend:")
print(cat_spend.to_string())

# Spend by Business Entity Type
biz_spend = df.groupby("Business Entity Type")["Spend"].agg(["sum","mean","count"])
biz_spend = biz_spend.sort_values("sum", ascending=False)
report["biz_type_spend"] = biz_spend.head(20).to_dict()
print("\nTop Business Entity Types:")
print(biz_spend.head(15).to_string())

# Monthly trend
if "Year Month" in df.columns:
    df["YearMonth"] = pd.to_numeric(df["Year Month"], errors="coerce")
    monthly = df.groupby("YearMonth")["Spend"].sum().sort_index()
    report["monthly_trend"] = monthly.to_dict()
    print("\nMonthly Spend Trend:")
    print(monthly.to_string())

# State analysis
state_spend = df.groupby("State")["Spend"].agg(["sum","mean","count"]).sort_values("sum", ascending=False)
report["top_states"] = state_spend.head(15).to_dict()
print("\nTop 10 States by Spend:")
print(state_spend.head(10).to_string())

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Feature Engineering")
print("=" * 60)

# Encode categoricals
from sklearn.preprocessing import LabelEncoder, StandardScaler

cat_cols = [
    "Business Entity Type", "Customer Market Segment Id",
    "Category Level 1", "Category Level 2", "Category Level 3",
    "State", "Distributor Group", "Ecommerce Status"
]

df_enc = df.copy()
encoders = {}
for col in cat_cols:
    if col in df_enc.columns:
        df_enc[col + "_enc"] = df_enc[col].fillna("UNKNOWN")
        le = LabelEncoder()
        df_enc[col + "_enc"] = le.fit_transform(df_enc[col + "_enc"].astype(str))
        encoders[col] = le

# Number of rooms (hospitality indicator)
df_enc["HasRooms"] = df_enc["Number of Rooms"].notna().astype(int)
df_enc["RoomCount"] = pd.to_numeric(df_enc["Number of Rooms"], errors="coerce").fillna(0)

# Ecommerce active flag
df_enc["IsEcomActive"] = (df_enc.get("Ecommerce Status", "").astype(str).str.lower() == "active").astype(int)

# Category depth (how specific is the purchase?)
df_enc["CategoryDepth"] = (
    df_enc["Category Level 1"].notna().astype(int) +
    df_enc["Category Level 2"].notna().astype(int) +
    df_enc["Category Level 3"].notna().astype(int) +
    (df_enc.get("Category Level 4", pd.Series(dtype=str)).notna()).astype(int)
)

# Month extracted
if "YearMonth" in df_enc.columns:
    df_enc["Month"] = df_enc["YearMonth"].astype(str).str[-2:].replace("", "0").astype(float)

feature_cols = [c for c in df_enc.columns if c.endswith("_enc")] + [
    "HasRooms", "RoomCount", "IsEcomActive", "CategoryDepth"
]
if "Month" in df_enc.columns:
    feature_cols.append("Month")

X = df_enc[feature_cols].fillna(0)
y = df_enc["LogSpend"]

print(f"Feature matrix: {X.shape}")
print(f"Features used: {feature_cols}")

# ─────────────────────────────────────────────
# 4. RANDOM FOREST — Feature Importance
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Random Forest Feature Importance")
print("=" * 60)

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

rf_r2  = r2_score(y_test, y_pred_rf)
rf_mae = mean_absolute_error(y_test, y_pred_rf)
print(f"  RF R²:  {rf_r2:.4f}")
print(f"  RF MAE: {rf_mae:.4f} (log-spend units)")

fi_rf = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nRandom Forest Feature Importances:")
print(fi_rf.to_string())
report["rf_feature_importance"] = fi_rf.to_dict()
report["rf_r2"] = rf_r2
report["rf_mae"] = rf_mae

# ─────────────────────────────────────────────
# 5. GRADIENT BOOSTING — cross-validate importance
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — Gradient Boosting (XGB-style) Cross-check")
print("=" * 60)

from sklearn.ensemble import HistGradientBoostingRegressor
hgb = HistGradientBoostingRegressor(max_iter=200, max_depth=8, random_state=42)
hgb.fit(X_train, y_train)
y_pred_hgb = hgb.predict(X_test)
hgb_r2  = r2_score(y_test, y_pred_hgb)
hgb_mae = mean_absolute_error(y_test, y_pred_hgb)
print(f"  HGB R²:  {hgb_r2:.4f}")
print(f"  HGB MAE: {hgb_mae:.4f} (log-spend units)")
report["hgb_r2"] = hgb_r2
report["hgb_mae"] = hgb_mae

# Permutation importance (model-agnostic)
from sklearn.inspection import permutation_importance
perm = permutation_importance(hgb, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
fi_perm = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False)
print("\nPermutation Importances (HGB):")
print(fi_perm.to_string())
report["permutation_importance"] = fi_perm.to_dict()

# ─────────────────────────────────────────────
# 6. K-MEANS CLUSTERING — Spend Segments
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — K-Means Customer Clustering")
print("=" * 60)

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Aggregate to customer level first
cust_agg = df.groupby("Customer Id").agg(
    TotalSpend=("Spend", "sum"),
    AvgSpend=("Spend", "mean"),
    Transactions=("Spend", "count"),
    UniqueCategories=("Category Level 1", "nunique"),
    UniqueDistributors=("Distributor ID", "nunique"),
    HasRooms=("Number of Rooms", lambda x: x.notna().any().astype(int))
).reset_index()

cust_agg["LogTotalSpend"] = np.log1p(cust_agg["TotalSpend"])
cust_agg["LogAvgSpend"]   = np.log1p(cust_agg["AvgSpend"])

cluster_features = ["LogTotalSpend","LogAvgSpend","Transactions","UniqueCategories","UniqueDistributors"]
Xc = cust_agg[cluster_features].fillna(0)
scaler = StandardScaler()
Xc_scaled = scaler.fit_transform(Xc)

# Elbow method (k=2..8)
inertias = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(Xc_scaled)
    inertias.append((k, km.inertia_))

print("  Elbow inertias:")
for k, inert in inertias:
    print(f"    k={k}: {inert:,.0f}")

# Use k=5
km5 = KMeans(n_clusters=5, random_state=42, n_init=10)
cust_agg["Cluster"] = km5.fit_predict(Xc_scaled)

cluster_summary = cust_agg.groupby("Cluster")[cluster_features + ["TotalSpend"]].mean().sort_values("TotalSpend", ascending=False)
print("\nCluster Profiles (mean values):")
print(cluster_summary.round(2).to_string())
report["cluster_summary"] = cluster_summary.to_dict()

# ─────────────────────────────────────────────
# 7. MUTUAL INFORMATION — Non-linear correlations
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 — Mutual Information Scores")
print("=" * 60)

from sklearn.feature_selection import mutual_info_regression
mi_scores = mutual_info_regression(X, y, random_state=42, n_neighbors=5)
mi_series = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)
print("Mutual Information (non-linear correlation with LogSpend):")
print(mi_series.to_string())
report["mutual_information"] = mi_series.to_dict()

# ─────────────────────────────────────────────
# 8. ANOMALY DETECTION — Isolation Forest
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8 — Anomaly Detection (Isolation Forest)")
print("=" * 60)

from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.02, random_state=42, n_jobs=-1)
iso_labels = iso.fit_predict(X)
n_anomalies = (iso_labels == -1).sum()
anomaly_rate = n_anomalies / len(X) * 100
print(f"  Anomalies detected: {n_anomalies:,} ({anomaly_rate:.1f}% of sample)")

df_enc["IsAnomaly"] = (iso_labels == -1).astype(int)
anomaly_spend = df_enc[df_enc["IsAnomaly"] == 1]["Spend"].describe()
normal_spend  = df_enc[df_enc["IsAnomaly"] == 0]["Spend"].describe()

print("\n  Anomalous transactions spend stats:")
print(f"    Mean:   ${anomaly_spend['mean']:,.0f}  |  Median: ${anomaly_spend['50%']:,.0f}")
print(f"    Max:    ${anomaly_spend['max']:,.0f}")
print("\n  Normal transactions spend stats:")
print(f"    Mean:   ${normal_spend['mean']:,.0f}  |  Median: ${normal_spend['50%']:,.0f}")
print(f"    Max:    ${normal_spend['max']:,.0f}")

# Top categories in anomalies
top_anom_cats = df_enc[df_enc["IsAnomaly"]==1]["Category Level 1"].value_counts().head(10)
print("\n  Top categories in anomalous transactions:")
print(top_anom_cats.to_string())
report["anomaly_rate"] = anomaly_rate
report["anomaly_top_cats"] = top_anom_cats.to_dict()

# ─────────────────────────────────────────────
# 9. CROSS-SEGMENT SPEND CONCENTRATION (Pareto)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9 — Pareto / Concentration Analysis")
print("=" * 60)

# Customer-level spend concentration
cust_spend = df.groupby("Customer Id")["Spend"].sum().sort_values(ascending=False)
total_spend = cust_spend.sum()
cumulative  = cust_spend.cumsum() / total_spend * 100

p20_idx = int(len(cust_spend) * 0.20)
p10_idx = int(len(cust_spend) * 0.10)
p1_idx  = int(len(cust_spend) * 0.01)

pct_top20 = cumulative.iloc[p20_idx]
pct_top10 = cumulative.iloc[p10_idx]
pct_top1  = cumulative.iloc[p1_idx]

print(f"  Top  1% of customers → {pct_top1:.1f}% of total spend")
print(f"  Top 10% of customers → {pct_top10:.1f}% of total spend")
print(f"  Top 20% of customers → {pct_top20:.1f}% of total spend")
report["pareto"] = {"top_1pct": pct_top1, "top_10pct": pct_top10, "top_20pct": pct_top20}

# Category concentration
cat_conc = df.groupby("Category Level 1")["Spend"].sum().sort_values(ascending=False)
cat_pct = cat_conc / cat_conc.sum() * 100
print("\n  Category Level 1 spend share:")
print(cat_pct.round(2).to_string())
report["category_concentration"] = cat_pct.to_dict()

# ─────────────────────────────────────────────
# 10. ECOMMERCE IMPACT ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10 — Ecommerce Status Impact")
print("=" * 60)

if "Ecommerce Status" in df.columns:
    eco_spend = df.groupby("Ecommerce Status")["Spend"].agg(["mean","median","count","sum"])
    print(eco_spend.to_string())
    report["ecommerce_impact"] = eco_spend.to_dict()

# ─────────────────────────────────────────────
# 11. SAVE RESULTS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 11 — Saving outputs")
print("=" * 60)

# Feature importance combined ranking
fi_combined = pd.DataFrame({
    "RF_Importance": fi_rf,
    "Permutation_Importance": fi_perm,
    "Mutual_Information": mi_series
})
fi_combined["Avg_Rank"] = fi_combined.rank(ascending=False).mean(axis=1)
fi_combined = fi_combined.sort_values("Avg_Rank")

print("\nFINAL COMBINED FEATURE RANKING:")
print(fi_combined.round(4).to_string())
report["combined_ranking"] = fi_combined.to_dict()

# Save CSVs
fi_combined.to_csv(f"{OUTPUT_DIR}/feature_importance.csv")
seg_spend.to_csv(f"{OUTPUT_DIR}/segment_spend.csv")
cat_spend.to_csv(f"{OUTPUT_DIR}/category_spend.csv")
state_spend.to_csv(f"{OUTPUT_DIR}/state_spend.csv")
cluster_summary.to_csv(f"{OUTPUT_DIR}/cluster_profiles.csv")
cust_agg.to_csv(f"{OUTPUT_DIR}/customer_clusters.csv", index=False)

# Save JSON report
def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

with open(f"{OUTPUT_DIR}/analysis_report.json", "w") as f:
    json.dump(make_serializable(report), f, indent=2)

print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
print("\nDone!")
