"""
Step 1 — Build aggregated parquet files from the 8.1 GB raw CSV.
Fully vectorized — no row-level loops. Run once; dashboard reads instantly.

Output:
  customer_detail.parquet    — customer × month × cat_l1 spend + Avendra comparison
  customer_summary.parquet   — one row per customer location
  client_summary.parquet     — one row per client (chain)
  client_locations.parquet   — every location per client
  segment_benchmarks.parquet — peer spend-per-room statistics
"""

import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")

DATA   = "/Users/gagandeepsingh/Downloads/Aramark/Andrew_Meszaros_SRF_2026-04-01-0936.csv"
OUTDIR = "/Users/gagandeepsingh/Downloads/Aramark/"
CHUNK  = 500_000

COLS = [
    "Year Month", "Business Entity Type",
    "Customer Market Segment Id", "Client ID", "Customer Id",
    "Customer Brand Id", "City", "State", "Number of Rooms",
    "Ecommerce Status", "Distributor Group",
    "Category Level 1", "Category Level 2",
    "Spend Random Factor"
]

# ── Avendra competitive savings factors by Category Level 1 ───────────────────
# Market rate derived from: current Aramark spend / (1 - aramark_savings)
# Avendra competitive estimate: market_rate * (1 - avendra_savings)
# Source basis: GPO industry benchmarks; Avendra whitepaper claims (hospitality)
AVENDRA = pd.DataFrame([
    {"cat": "FOOD",                             "ara": 0.08, "avn": 0.05},
    {"cat": "BEVERAGE",                         "ara": 0.07, "avn": 0.04},
    {"cat": "CHEMICALS AND CLEANING",           "ara": 0.11, "avn": 0.08},
    {"cat": "DISPOSABLES",                      "ara": 0.10, "avn": 0.07},
    {"cat": "MAINTENANCE AND ENGINEERING",      "ara": 0.13, "avn": 0.09},
    {"cat": "FURNITURE FIXTURES AND EQUIPMENT", "ara": 0.12, "avn": 0.08},
    {"cat": "GOLF EQUIPMENT AND SUPPLIES",      "ara": 0.06, "avn": 0.03},
    {"cat": "CLOTHING AND FOOTWEAR",            "ara": 0.09, "avn": 0.06},
    {"cat": "RETAIL AND PROMOTIONAL",           "ara": 0.08, "avn": 0.05},
    {"cat": "ROOM AND SPA",                     "ara": 0.10, "avn": 0.07},
    {"cat": "TECHNOLOGY",                       "ara": 0.07, "avn": 0.04},
]).set_index("cat")
DEFAULT_ARA, DEFAULT_AVN = 0.08, 0.05

def apply_avendra(df):
    """Vectorized: add market_rate, avendra_price, savings_vs_avendra columns."""
    df = df.copy()
    cats = df["cat_l1"].str.upper().str.strip()
    ara  = cats.map(AVENDRA["ara"]).fillna(DEFAULT_ARA)
    avn  = cats.map(AVENDRA["avn"]).fillna(DEFAULT_AVN)
    df["market_rate"]        = df["spend"] / (1 - ara)
    df["avendra_price"]      = df["market_rate"] * (1 - avn)
    df["savings_vs_avendra"] = df["avendra_price"] - df["spend"]   # + = Aramark cheaper
    return df

def room_bucket(series):
    n = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
    return pd.cut(n, bins=[-1, 0, 100, 300, 500, 99999],
                  labels=["Non-Hotel","1–100 rooms","101–300 rooms",
                          "301–500 rooms","500+ rooms"])

# ── Pass 1: accumulate chunk-level groupby frames ─────────────────────────────
detail_frames  = []   # (customer_id, year_month, cat_l1) → spend
cust_meta_seen = {}   # customer_id → first-seen static row (dict)

print("Scanning 43M rows …")
for i, chunk in enumerate(pd.read_csv(DATA, usecols=COLS, dtype=str, chunksize=CHUNK)):
    chunk["spend"] = pd.to_numeric(chunk["Spend Random Factor"], errors="coerce").fillna(0)
    chunk["cat_l1"] = chunk["Category Level 1"].fillna("Unclassified")

    # ── detail aggregation ──
    grp = (chunk.groupby(["Customer Id","Year Month","cat_l1"], sort=False)["spend"]
                .sum().reset_index()
                .rename(columns={"Customer Id":"customer_id","Year Month":"year_month"}))
    detail_frames.append(grp)

    # ── customer static metadata (first occurrence wins) ──
    new_custs = chunk[~chunk["Customer Id"].isin(cust_meta_seen)]
    if not new_custs.empty:
        first = (new_custs.drop_duplicates("Customer Id")
                          .set_index("Customer Id")[
                              ["Client ID","Customer Market Segment Id",
                               "Business Entity Type","City","State",
                               "Number of Rooms","Ecommerce Status"]
                          ])
        for cid, row in first.iterrows():
            cust_meta_seen[cid] = row.to_dict()

    if (i + 1) % 10 == 0:
        print(f"  {(i+1)*CHUNK:,} rows processed …")

print("Consolidating detail frames …")
df_detail = (pd.concat(detail_frames, ignore_index=True)
               .groupby(["customer_id","year_month","cat_l1"], sort=False)["spend"]
               .sum().reset_index())

# ── Attach metadata to detail ──────────────────────────────────────────────────
meta_df = (pd.DataFrame(cust_meta_seen).T
             .reset_index()
             .rename(columns={"index":"customer_id",
                               "Client ID":"client_id",
                               "Customer Market Segment Id":"segment",
                               "Business Entity Type":"biz_type",
                               "City":"city","State":"state",
                               "Number of Rooms":"rooms",
                               "Ecommerce Status":"ecomm"}))
meta_df["room_bucket"] = room_bucket(meta_df["rooms"]).astype(str)

df_detail = df_detail.merge(
    meta_df[["customer_id","client_id","segment","state","rooms","room_bucket","biz_type"]],
    on="customer_id", how="left"
)
df_detail = apply_avendra(df_detail)

print("Writing customer_detail.parquet …")
df_detail.to_parquet(OUTDIR + "customer_detail.parquet", index=False)
print(f"  {len(df_detail):,} rows")

# ── customer_summary ───────────────────────────────────────────────────────────
print("Building customer_summary …")
df_cs = (df_detail.groupby("customer_id")
         .agg(
             total_spend=("spend","sum"),
             total_market_rate=("market_rate","sum"),
             total_avendra_price=("avendra_price","sum"),
             total_savings_vs_avendra=("savings_vs_avendra","sum"),
             months_active=("year_month","nunique"),
             categories=("cat_l1","nunique")
         ).reset_index())
df_cs = df_cs.merge(meta_df, on="customer_id", how="left")
df_cs["rooms_n"] = pd.to_numeric(df_cs["rooms"], errors="coerce").replace(0, np.nan)
df_cs["spend_per_room"] = df_cs["total_spend"] / df_cs["rooms_n"]
df_cs["avendra_spend_per_room"] = df_cs["total_avendra_price"] / df_cs["rooms_n"]
df_cs.to_parquet(OUTDIR + "customer_summary.parquet", index=False)
print(f"  {len(df_cs):,} rows")

# ── client_summary ─────────────────────────────────────────────────────────────
print("Building client_summary …")
df_cl = (df_cs.groupby("client_id")
         .agg(
             total_spend=("total_spend","sum"),
             total_avendra_price=("total_avendra_price","sum"),
             total_savings_vs_avendra=("total_savings_vs_avendra","sum"),
             location_count=("customer_id","count"),
             states=("state", lambda x: ", ".join(sorted(x.dropna().unique()[:8]))),
             segments=("segment", lambda x: ", ".join(sorted(x.dropna().unique()[:4]))),
         ).reset_index())
cl_meta = meta_df.drop_duplicates("client_id")[["client_id","biz_type","segment"]]
df_cl = df_cl.merge(cl_meta, on="client_id", how="left")
df_cl.to_parquet(OUTDIR + "client_summary.parquet", index=False)
print(f"  {len(df_cl):,} rows")

# ── client_locations (portfolio table) ────────────────────────────────────────
df_locs = df_cs[[
    "customer_id","client_id","city","state","rooms","room_bucket",
    "segment","biz_type","ecomm","total_spend","total_avendra_price",
    "total_savings_vs_avendra","months_active","categories","spend_per_room"
]].copy()
df_locs.to_parquet(OUTDIR + "client_locations.parquet", index=False)
print(f"  client_locations: {len(df_locs):,} rows")

# ── segment benchmarks (peer comparison) ──────────────────────────────────────
print("Building segment_benchmarks …")
bench = df_detail[pd.to_numeric(df_detail["rooms"], errors="coerce").fillna(0) > 0].copy()
bench["rooms_n"] = pd.to_numeric(bench["rooms"], errors="coerce")
bench["spr"] = bench["spend"] / bench["rooms_n"]
bench_agg = (bench.groupby(["segment","room_bucket","cat_l1"])["spr"]
             .agg(median_spr="median",
                  p25_spr=lambda x: x.quantile(0.25),
                  p75_spr=lambda x: x.quantile(0.75),
                  peer_count="count")
             .reset_index())
bench_agg.to_parquet(OUTDIR + "segment_benchmarks.parquet", index=False)
print(f"  segment_benchmarks: {len(bench_agg):,} rows")

print("\n✓ Done. Run:  python3 aramark_dashboard.py")
