import os
_HERE = os.path.dirname(os.path.abspath(__file__)) + os.sep
"""
Feature Engineering — Board-Level Intelligence
===============================================
Reads customer_detail.parquet (2.1M rows, already aggregated from 43M).
Produces one rich feature set per aggregate dimension:

  seg_features.parquet      — 1 row per market segment  (strategic matrix)
  seg_monthly.parquet       — segment × month           (trend + forecast)
  cat_monthly.parquet       — category × month          (mix shift)
  state_monthly.parquet     — state × month             (geo opportunity)
  portfolio_monthly.parquet — total portfolio × month   (top-line forecast)
  concentration.parquet     — client concentration / Pareto
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")

DIR = _HERE

print("Loading customer_detail …")
df = pd.read_parquet(DIR + "customer_detail.parquet")
cs = pd.read_parquet(DIR + "customer_summary.parquet")

# Numeric safety
for c in ["spend","avendra_price","savings_vs_avendra"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
for c in ["total_spend","total_avendra_price","total_savings_vs_avendra"]:
    cs[c] = pd.to_numeric(cs[c], errors="coerce").fillna(0)

df["year_month"] = pd.to_numeric(df["year_month"], errors="coerce")
df = df[df["year_month"].between(202501, 202512)]

MONTHS = sorted(df["year_month"].unique())   # [202501 … 202512]
N = len(MONTHS)
print(f"  {len(df):,} rows | {N} months | {df['customer_id'].nunique():,} customers")

# ── Helper: compute trend features from a spend array ─────────────────────────
def trend_features(vals):
    vals = np.array(vals, dtype=float)
    n    = len(vals)
    if n < 2:
        return dict(slope=0, slope_norm=0, r2=0, momentum=0, volatility=0,
                    q4_avg=vals[-1] if n else 0, q1_avg=vals[0] if n else 0,
                    best_month=0, worst_month=0)
    x    = np.arange(n).reshape(-1,1)
    lr   = LinearRegression().fit(x, vals)
    preds = lr.predict(x)
    ss_res = np.sum((vals - preds)**2)
    ss_tot = np.sum((vals - vals.mean())**2)
    r2   = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    mean = vals.mean() or 1e-9
    q1   = vals[:3].mean() if n >= 3 else vals.mean()
    q4   = vals[-3:].mean() if n >= 3 else vals.mean()
    return dict(
        slope       = float(lr.coef_[0]),
        slope_norm  = float(lr.coef_[0]) / mean,       # % change per month
        r2          = float(r2),
        momentum    = float((q4 - q1) / (q1 or 1e-9)), # H2 vs H1 growth
        volatility  = float(vals.std() / mean),         # coefficient of variation
        q4_avg      = float(q4),
        q1_avg      = float(q1),
        best_month  = int(np.argmax(vals)),
        worst_month = int(np.argmin(vals)),
    )

# ══════════════════════════════════════════════════════════════════════════════
# 1. Portfolio-level monthly (top-line)
# ══════════════════════════════════════════════════════════════════════════════
port = df.groupby("year_month").agg(
    spend          = ("spend",              "sum"),
    avendra_price  = ("avendra_price",      "sum"),
    savings        = ("savings_vs_avendra", "sum"),
    locations      = ("customer_id",        "nunique"),
    clients        = ("client_id",          "nunique"),
    categories     = ("cat_l1",             "nunique"),
).reset_index()
port.to_parquet(DIR + "portfolio_monthly.parquet", index=False)
print(f"portfolio_monthly: {len(port)} rows")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Segment × Month
# ══════════════════════════════════════════════════════════════════════════════
seg_mon = df.groupby(["segment","year_month"]).agg(
    spend          = ("spend",              "sum"),
    avendra_price  = ("avendra_price",      "sum"),
    savings        = ("savings_vs_avendra", "sum"),
    locations      = ("customer_id",        "nunique"),
    clients        = ("client_id",          "nunique"),
    categories     = ("cat_l1",             "nunique"),
).reset_index()
seg_mon.to_parquet(DIR + "seg_monthly.parquet", index=False)
print(f"seg_monthly: {len(seg_mon)} rows")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Category × Month
# ══════════════════════════════════════════════════════════════════════════════
cat_mon = df.groupby(["cat_l1","year_month"]).agg(
    spend         = ("spend",         "sum"),
    avendra_price = ("avendra_price", "sum"),
    savings       = ("savings_vs_avendra","sum"),
    locations     = ("customer_id",   "nunique"),
    segments      = ("segment",       "nunique"),
).reset_index()
cat_mon.to_parquet(DIR + "cat_monthly.parquet", index=False)
print(f"cat_monthly: {len(cat_mon)} rows")

# ══════════════════════════════════════════════════════════════════════════════
# 4. State × Month
# ══════════════════════════════════════════════════════════════════════════════
US_STATES = {
    'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN',
    'IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV',
    'NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN',
    'TX','UT','VT','VA','WA','WV','WI','WY','DC'
}
state_mon = df[df["state"].isin(US_STATES)].groupby(["state","year_month"]).agg(
    spend     = ("spend",       "sum"),
    locations = ("customer_id", "nunique"),
    clients   = ("client_id",   "nunique"),
).reset_index()
state_mon.to_parquet(DIR + "state_monthly.parquet", index=False)
print(f"state_monthly: {len(state_mon)} rows")

# ══════════════════════════════════════════════════════════════════════════════
# 5. Segment feature matrix  (the strategic intelligence layer)
# ══════════════════════════════════════════════════════════════════════════════
# Ecomm penetration per segment
ecomm = (cs[cs["ecomm"].isin(["ACTIVE","Active","active"])]
         .groupby("segment")["customer_id"].count()
         .rename("ecomm_active"))
total_locs_seg = cs.groupby("segment")["customer_id"].count().rename("total_locs")
ecomm_pen = (ecomm / total_locs_seg).fillna(0).rename("ecomm_penetration")

seg_rows = []
for seg, grp in seg_mon.groupby("segment"):
    grp = grp.sort_values("year_month")
    full = (grp.set_index("year_month")
               .reindex(MONTHS, fill_value=0)
               .reset_index())
    vals = full["spend"].values

    tf = trend_features(vals)

    total_spend      = float(grp["spend"].sum())
    total_avendra    = float(grp["avendra_price"].sum())
    total_savings    = float(grp["savings"].sum())
    avg_locations    = float(grp["locations"].mean())
    max_locations    = float(grp["locations"].max())
    category_breadth = float(grp["categories"].max())
    avendra_adv_pct  = total_savings / (total_avendra + 1e-9) * 100

    seg_rows.append({
        "segment":            seg,
        "total_spend":        total_spend,
        "avg_monthly_spend":  total_spend / N,
        "avendra_advantage_pct": avendra_adv_pct,
        "total_savings_vs_avendra": total_savings,
        "avg_locations":      avg_locations,
        "max_locations":      max_locations,
        "category_breadth":   category_breadth,
        "ecomm_penetration":  float(ecomm_pen.get(seg, 0)),
        **{f"trend_{k}": v for k, v in tf.items()},
    })

seg_feat = pd.DataFrame(seg_rows)

# ── Strategic classification (2×2 quadrant logic) ─────────────────────────────
# Use median thresholds so classification is data-relative, not hard-coded
spend_med   = seg_feat["total_spend"].median()
mom_zero    = 0.0   # positive momentum = growing

def classify(row):
    big  = row["total_spend"]   >= spend_med
    grow = row["trend_momentum"] > mom_zero
    if big  and grow: return "Protect & Grow"
    if not big  and grow: return "Emerging Opportunity"
    if big  and not grow: return "Revenue at Risk"
    return "Monitor"

seg_feat["strategic_class"]   = seg_feat.apply(classify, axis=1)
seg_feat["growth_signal"]     = seg_feat["trend_momentum"].apply(
    lambda x: "Growing" if x > 0.05 else ("Declining" if x < -0.05 else "Stable"))

seg_feat.to_parquet(DIR + "seg_features.parquet", index=False)
print(f"seg_features: {len(seg_feat)} rows × {len(seg_feat.columns)} features")
print(seg_feat[["segment","total_spend","trend_momentum","strategic_class"]].to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 6. Client concentration (Pareto / Gini)
# ══════════════════════════════════════════════════════════════════════════════
cl = pd.read_parquet(DIR + "client_summary.parquet")
cl["total_spend"] = pd.to_numeric(cl["total_spend"], errors="coerce").fillna(0)
cl_sorted = cl.sort_values("total_spend", ascending=False).reset_index(drop=True)
cl_sorted["cum_spend"]    = cl_sorted["total_spend"].cumsum()
cl_sorted["cum_pct"]      = cl_sorted["cum_spend"] / cl_sorted["total_spend"].sum() * 100
cl_sorted["client_rank"]  = cl_sorted.index + 1
cl_sorted["client_pct"]   = cl_sorted["client_rank"] / len(cl_sorted) * 100

# Gini coefficient
n  = len(cl_sorted)
vals = cl_sorted["total_spend"].values
gini = (2 * np.sum((np.arange(1, n+1)) * np.sort(vals)) / (n * vals.sum()) - (n+1)/n)

cl_sorted["gini"] = gini
cl_sorted.to_parquet(DIR + "concentration.parquet", index=False)
print(f"concentration: {len(cl_sorted)} clients  |  Gini = {gini:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. Random Forest feature importance  (what drives spend magnitude?)
# ══════════════════════════════════════════════════════════════════════════════
feature_cols = [
    "avg_locations","category_breadth","ecomm_penetration",
    "trend_volatility","trend_r2","trend_slope_norm","trend_momentum"
]
target_col = "avg_monthly_spend"

feat_df = seg_feat[feature_cols + [target_col]].dropna()
if len(feat_df) >= 5:
    scaler = StandardScaler()
    X = scaler.fit_transform(feat_df[feature_cols])
    y = feat_df[target_col].values
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X, y)
    importance = pd.DataFrame({
        "feature":    feature_cols,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)
    importance.to_parquet(DIR + "feature_importance.parquet", index=False)
    print("\nFeature importances (what predicts high-spend segments):")
    print(importance.to_string(index=False))

print("\n✓ Feature engineering complete. Run:  python3 board_dashboard.py")
