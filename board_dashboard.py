"""
Aramark Board Intelligence Dashboard
═════════════════════════════════════
Treats the entire dataset as one business.
Four board-ready views:

  Overview   — portfolio health + ML forecast + feature drivers
  Strategy   — strategic segment matrix (BCG-style 2×2)
  Forecast   — per-segment ML forecasts (SVR / Ridge / ETS auto-selected)
  Risk       — revenue concentration, Gini, at-risk segments

Run:  python3 board_dashboard.py  →  http://127.0.0.1:8050
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

# ── Data ───────────────────────────────────────────────────────────────────────
DIR = "/Users/gagandeepsingh/Downloads/Aramark/"

print("Loading …")
port  = pd.read_parquet(DIR + "portfolio_monthly.parquet")
seg_m = pd.read_parquet(DIR + "seg_monthly.parquet")
seg_f = pd.read_parquet(DIR + "seg_features.parquet")
cat_m = pd.read_parquet(DIR + "cat_monthly.parquet")
st_m  = pd.read_parquet(DIR + "state_monthly.parquet")
conc  = pd.read_parquet(DIR + "concentration.parquet")
fi    = pd.read_parquet(DIR + "feature_importance.parquet")
print("  Ready.")

for df_ in [port, seg_m, seg_f, cat_m, st_m, conc]:
    for c in df_.select_dtypes("number").columns:
        df_[c] = pd.to_numeric(df_[c], errors="coerce").fillna(0)

# ── Segment readable labels ───────────────────────────────────────────────────
# Derived by inspecting dominant Category Level 1 per segment in the raw data
SEG_LABELS = {
    "MS-100000": "Facilities / Education",
    "MS-100001": "Healthcare Dining",
    "MS-100002": "Hospitality (Large)",
    "MS-100003": "Institutional Food Svc",
    "MS-100004": "Correctional / Justice",
    "MS-100005": "Sports & Entertainment",
    "MS-100006": "Business Dining",
    "MS-100007": "Managed Facilities (M&E)",
    "MS-100010": "Senior Living",
    "MS-100013": "Travel Centers",
    "MS-100015": "Specialty Venues",
    "MS-100021": "Hotel / Lodging (Large)",
    "MS-100022": "Hotel / Lodging (Mid)",
    "MS-100023": "Resort & Leisure",
    "MS-100024": "Convention & Conference",
    "MS-100025": "Golf & Turf Mgmt",
    "MS-100026": "Luxury Resort (HI)",
    "MS-100027": "Campus / University",
    "MS-100030": "Outdoor / Recreation",
    "MS-100033": "Retail & Concessions",
    "MS-100037": "Military / Government",
    "MS-100038": "Specialty Food",
    "MS-100043": "Emerging Markets",
    "MS-100046": "Micro-Markets",
    "MS-100047": "Contract Catering",
    "MS-100065": "AV & Technology Svcs",
    "MS-100113": "Vending & Automation",
    "MS-100115": "Environmental Svcs",
}

seg_f["label"] = seg_f["segment"].map(SEG_LABELS).fillna(seg_f["segment"])
seg_m["label"] = seg_m["segment"].map(SEG_LABELS).fillna(seg_m["segment"])

MONTHS     = sorted(port["year_month"].unique())
MONTH_LBLS = []
_M = {"01":"Jan","02":"Feb","03":"Apr","04":"Apr","05":"May","06":"Jun",
      "07":"Jul","08":"Aug","09":"Sep","10":"Oct","11":"Nov","12":"Dec"}
_M = {f"{i:02d}": m for i, m in enumerate(
      ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],1)}
for ym in MONTHS:
    s = str(int(ym))
    MONTH_LBLS.append(s[:4] + " " + _M.get(s[4:], s[4:]))

FORECAST_LBLS = ["2026 Jan", "2026 Feb", "2026 Mar"]
US_STATES = {'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL',
             'IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT',
             'NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI',
             'SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY','DC'}

# ── Colors ────────────────────────────────────────────────────────────────────
C_BLUE   = "#003087"
C_RED    = "#DA291C"
C_GREEN  = "#1b7f4f"
C_AMBER  = "#d97706"
C_PURPLE = "#6366f1"
C_BG     = "#f0f4f8"
C_CARD   = "#ffffff"
C_MUTED  = "#64748b"
C_TEXT   = "#1e293b"
C_BORDER = "#e2e8f0"

CLASS_COLORS = {
    "Protect & Grow":       C_GREEN,
    "Emerging Opportunity": C_BLUE,
    "Revenue at Risk":      C_RED,
    "Monitor":              C_AMBER,
}

def fmt(v, decimals=2):
    try:
        v = float(v)
        if abs(v) >= 1e9: return f"${v/1e9:.{decimals}f}B"
        if abs(v) >= 1e6: return f"${v/1e6:.{decimals}f}M"
        if abs(v) >= 1e3: return f"${v/1e3:.1f}K"
        return f"${v:,.0f}"
    except: return "—"

def pct(v):
    try: return f"{float(v)*100:+.1f}%"
    except: return "—"

# ── ML Forecasting ─────────────────────────────────────────────────────────────
def make_features_ml(idx, hist):
    return [idx,
            np.sin(2*np.pi*idx/12), np.cos(2*np.pi*idx/12),
            hist[-1], hist[-2] if len(hist)>1 else hist[-1],
            np.mean(hist[-3:])]

def forecast(vals, n_ahead=3):
    vals = np.array(vals, dtype=float)
    n    = len(vals)
    results = {}

    # ── ETS ─────────────────────────────────────────────
    try:
        hist = list(vals)
        preds = []
        for _ in range(n_ahead):
            m = SimpleExpSmoothing(np.array(hist), initialization_method="estimated").fit()
            p = max(float(m.forecast(1)[0]), 0)
            preds.append(p); hist.append(p)
        if n >= 6:
            m2 = SimpleExpSmoothing(vals[:-3], initialization_method="estimated").fit()
            rmse = np.sqrt(mean_squared_error(vals[-3:], m2.forecast(3)))
        else:
            rmse = np.std(vals) + 1
        results["Exp. Smoothing"] = (preds, rmse)
    except: pass

    # ── Ridge ───────────────────────────────────────────
    if n >= 5:
        try:
            X = [make_features_ml(i+1, list(vals[:i+1])) for i in range(2, n)]
            y = vals[2:]
            cut = max(2, len(y)-3)
            pipe = Pipeline([("s",StandardScaler()), ("m",Ridge(alpha=1.0))])
            pipe.fit(X[:cut], y[:cut])
            rmse = (np.sqrt(mean_squared_error(y[cut:], pipe.predict(X[cut:])))
                    if cut < len(y) else np.std(y)+1)
            hist = list(vals)
            preds = []
            for step in range(n_ahead):
                f = make_features_ml(n+step+1, hist)
                p = max(float(pipe.predict([f])[0]), 0)
                preds.append(p); hist.append(p)
            results["Ridge"] = (preds, rmse)
        except: pass

    # ── SVR ─────────────────────────────────────────────
    if n >= 5:
        try:
            X = [make_features_ml(i+1, list(vals[:i+1])) for i in range(2, n)]
            y = vals[2:]
            cut = max(2, len(y)-3)
            pipe = Pipeline([("s",StandardScaler()),
                              ("m",SVR(kernel="rbf", C=100, epsilon=0.05))])
            pipe.fit(X[:cut], y[:cut])
            rmse = (np.sqrt(mean_squared_error(y[cut:], pipe.predict(X[cut:])))
                    if cut < len(y) else np.std(y)+1)
            hist = list(vals)
            preds = []
            for step in range(n_ahead):
                f = make_features_ml(n+step+1, hist)
                p = max(float(pipe.predict([f])[0]), 0)
                preds.append(p); hist.append(p)
            results["SVR (RBF)"] = (preds, rmse)
        except: pass

    if not results:
        avg = float(vals.mean())
        return [avg]*n_ahead, [avg]*n_ahead, [avg]*n_ahead, "Fallback"

    best = min(results, key=lambda k: results[k][1])
    preds = [max(p,0) for p in results[best][0]]
    std   = results[best][1]

    # All models' central estimates for comparison
    all_preds = np.array([np.array([max(p,0) for p in v[0]])
                           for v in results.values()])
    lo = all_preds.min(axis=0).tolist()
    hi = all_preds.max(axis=0).tolist()

    return preds, lo, hi, best

# ── Pre-compute portfolio + segment forecasts ──────────────────────────────────
print("Running ML forecasts …")
port_vals  = port.sort_values("year_month")["spend"].values
p_preds, p_lo, p_hi, p_model = forecast(port_vals)

seg_forecasts = {}
for seg, grp in seg_m.groupby("segment"):
    grp = grp.sort_values("year_month")
    full = grp.set_index("year_month").reindex(MONTHS, fill_value=0)
    preds, lo, hi, model = forecast(full["spend"].values)
    seg_forecasts[seg] = {"preds":preds, "lo":lo, "hi":hi, "model":model}
print(f"  Portfolio best model: {p_model}")
print(f"  Q1 2026 portfolio forecast: {fmt(p_preds[0])} / {fmt(p_preds[1])} / {fmt(p_preds[2])}")

# ── Chart builders ─────────────────────────────────────────────────────────────
LAYOUT_BASE = dict(
    template="plotly_white",
    paper_bgcolor=C_CARD, plot_bgcolor="#f8fafc",
    font=dict(family="Inter, Segoe UI, sans-serif", color=C_TEXT, size=12),
    margin=dict(l=12, r=12, t=44, b=12),
)
AXIS_BASE = dict(gridcolor="#e2e8f0", zeroline=False, showline=False)

def dark_layout(**kwargs):
    d = {**LAYOUT_BASE}
    d.update(kwargs)
    return d

def build_portfolio_chart():
    fig = go.Figure()


    # Confidence band
    fig.add_trace(go.Scatter(
        x=FORECAST_LBLS + FORECAST_LBLS[::-1],
        y=p_hi + p_lo[::-1],
        fill="toself", fillcolor="rgba(218,41,28,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Forecast range", showlegend=True,
        hoverinfo="skip"
    ))
    # Actual
    fig.add_trace(go.Scatter(
        x=MONTH_LBLS, y=port["spend"].values,
        name="Actual Portfolio Spend",
        mode="lines+markers",
        line=dict(color=C_BLUE, width=3),
        marker=dict(size=7, color=C_BLUE),
        hovertemplate="%{x}: %{y:$,.0f}<extra></extra>"
    ))
    # Bridge
    fig.add_trace(go.Scatter(
        x=[MONTH_LBLS[-1], FORECAST_LBLS[0]],
        y=[float(port["spend"].iloc[-1]), p_preds[0]],
        mode="lines", line=dict(color=C_RED, dash="dot", width=2),
        showlegend=False, hoverinfo="skip"
    ))
    # Forecast
    fig.add_trace(go.Scatter(
        x=FORECAST_LBLS, y=p_preds,
        name=f"Forecast ({p_model})",
        mode="lines+markers",
        line=dict(color=C_RED, width=3, dash="dash"),
        marker=dict(size=10, symbol="diamond", color=C_RED),
        hovertemplate="%{x}: %{y:$,.0f}<extra></extra>"
    ))
    fig.add_vline(x=len(MONTH_LBLS)-0.5, line_dash="dot",
                  line_color=C_MUTED, line_width=1,
                  annotation_text=" Q1 2026 Forecast →",
                  annotation_font=dict(color=C_MUTED, size=11))
    ALL_X = MONTH_LBLS + FORECAST_LBLS   # enforce chronological order
    fig.update_layout(
        **dark_layout(
            title=dict(text="Total Portfolio Spend — 2025 Actuals + Q1 2026 ML Forecast",
                       font=dict(size=15), x=0.01),
            height=420,
            hovermode="x unified",
            legend=dict(orientation="h", y=1.08, font=dict(size=12)),
            yaxis=dict(**AXIS_BASE, tickformat="$,.0f", tickfont=dict(size=12)),
            xaxis=dict(**AXIS_BASE, tickfont=dict(size=12),
                       categoryorder="array", categoryarray=ALL_X),
            margin=dict(l=16, r=16, t=60, b=20),
        )
    )
    return fig

def build_strategic_matrix():
    df = seg_f.copy()
    df["spend_B"] = df["total_spend"] / 1e6
    df["mom_pct"] = df["trend_momentum"] * 100

    fig = go.Figure()

    for cls, color in CLASS_COLORS.items():
        sub = df[df["strategic_class"] == cls]
        fig.add_trace(go.Scatter(
            x=sub["spend_B"], y=sub["mom_pct"],
            mode="markers+text",
            name=cls,
            marker=dict(
                size=np.sqrt(sub["avg_locations"].clip(1)) * 6,
                color=color, opacity=0.85,
                line=dict(color="white", width=1)
            ),
            text=sub["label"],
            textposition="top center",
            textfont=dict(size=9, color="white"),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Total Spend: $%{x:.1f}M<br>"
                "Growth Momentum: %{y:+.1f}%<br>"
                "<extra>" + cls + "</extra>"
            )
        ))

    # Quadrant lines at median spend, 0% momentum
    med_spend = float(df["total_spend"].median()) / 1e6
    fig.add_vline(x=med_spend, line_dash="dot", line_color=C_MUTED, line_width=1)
    fig.add_hline(y=0,          line_dash="dot", line_color=C_MUTED, line_width=1)

    # Quadrant labels
    x_max = float(df["spend_B"].max()) * 1.05
    y_max = float(df["mom_pct"].max()) * 1.05
    y_min = float(df["mom_pct"].min()) * 1.05
    for txt, xp, yp, color in [
        ("PROTECT & GROW",       x_max*0.98, y_max*0.95, C_GREEN),
        ("EMERGING OPPORTUNITY", med_spend*0.02, y_max*0.95, C_BLUE),
        ("REVENUE AT RISK",      x_max*0.98, y_min*0.95, C_RED),
        ("MONITOR",              med_spend*0.02, y_min*0.95, C_AMBER),
    ]:
        fig.add_annotation(text=txt, x=xp, y=yp,
                           xanchor="right" if xp > med_spend else "left",
                           showarrow=False,
                           font=dict(size=9, color=color, family="Inter"),
                           opacity=0.6)

    fig.update_layout(
        **dark_layout(
            title=dict(text="Strategic Segment Matrix  — Size × Growth Momentum  (bubble = avg locations)",
                       font=dict(size=15), x=0.01),
            height=580,
            showlegend=True,
            legend=dict(orientation="h", y=-0.10, font=dict(size=12)),
            xaxis=dict(**AXIS_BASE, title=dict(text="Total 2025 Spend ($M)", font=dict(size=12)),
                       tickprefix="$", ticksuffix="M", tickfont=dict(size=11)),
            yaxis=dict(**AXIS_BASE, title=dict(text="H2 vs H1 Growth Momentum (%)", font=dict(size=12)),
                       ticksuffix="%", tickfont=dict(size=11)),
            margin=dict(l=16, r=16, t=60, b=80),
        )
    )
    return fig

def build_segment_momentum_bar():
    df = seg_f.sort_values("trend_momentum").copy()
    df["mom_pct"] = df["trend_momentum"] * 100
    df["color"]   = df["trend_momentum"].apply(
        lambda v: C_GREEN if v > 0.05 else (C_RED if v < -0.05 else C_AMBER))
    df["arrow"]   = df["trend_momentum"].apply(
        lambda v: "▲" if v > 0.05 else ("▼" if v < -0.05 else "→"))
    df["text"]    = df.apply(lambda r: f"{r['arrow']} {r['mom_pct']:+.1f}%", axis=1)

    fig = go.Figure(go.Bar(
        x=df["mom_pct"], y=df["label"],
        orientation="h",
        marker_color=df["color"],
        text=df["text"], textposition="outside",
        textfont=dict(size=10),
        hovertemplate="<b>%{y}</b><br>H2 vs H1: %{x:+.1f}%<extra></extra>"
    ))
    fig.add_vline(x=0, line_color="white", line_width=1.5)
    fig.update_layout(
        **dark_layout(
            title=dict(text="Segment Growth Momentum  (H2 vs H1 2025)",
                       font=dict(size=14), x=0.01),
            height=580,
            xaxis=dict(**AXIS_BASE, ticksuffix="%", title=None, tickfont=dict(size=11)),
            yaxis=dict(**AXIS_BASE, title=None, autorange="reversed",
                       tickfont=dict(size=11)),
            margin=dict(l=16, r=60, t=60, b=20),
        )
    )
    return fig

def build_category_bars():
    """
    Horizontal ranked bar: each category = % of total revenue.
    Color = growth direction (H2 vs H1). Immediately readable.
    """
    total_by_cat = cat_m.groupby("cat_l1")["spend"].sum().reset_index()
    total_all    = total_by_cat["spend"].sum()
    total_by_cat["pct"] = total_by_cat["spend"] / total_all * 100

    # Growth signal per category
    q1 = cat_m[cat_m["year_month"] <= 202503].groupby("cat_l1")["spend"].mean()
    q4 = cat_m[cat_m["year_month"] >= 202510].groupby("cat_l1")["spend"].mean()
    mom = ((q4 - q1) / (q1.replace(0, np.nan)) * 100).rename("mom_pct")
    df  = total_by_cat.merge(mom.reset_index(), on="cat_l1", how="left")
    df  = df.sort_values("pct", ascending=True)

    bar_colors = df["mom_pct"].apply(
        lambda v: C_GREEN if v > 3 else (C_RED if v < -3 else C_AMBER))
    trend_label = df["mom_pct"].apply(
        lambda v: f"  ▲ {v:+.0f}% growth" if v > 3
             else (f"  ▼ {v:+.0f}% declining" if v < -3 else f"  → {v:+.0f}% stable"))

    fig = go.Figure(go.Bar(
        x=df["pct"],
        y=df["cat_l1"],
        orientation="h",
        marker_color=bar_colors,
        text=[f"{p:.1f}%{t}" for p, t in zip(df["pct"], trend_label)],
        textposition="outside",
        textfont=dict(size=11, color=C_TEXT),
        customdata=df["spend"],
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Share of Portfolio: %{x:.1f}%<br>"
            "Total Spend: $%{customdata:,.0f}<extra></extra>"
        )
    ))
    fig.update_layout(
        **dark_layout(
            title=dict(
                text="Revenue by Category — Share of Portfolio  "
                     "<span style='color:#1b7f4f'>▲ Growing</span>  "
                     "<span style='color:#d97706'>→ Stable</span>  "
                     "<span style='color:#c0392b'>▼ Declining</span>",
                font=dict(size=14), x=0.01),
            height=440,
            xaxis=dict(**AXIS_BASE, ticksuffix="%", range=[0, df["pct"].max()*1.35],
                       tickfont=dict(size=11), title=None),
            yaxis=dict(**AXIS_BASE, tickfont=dict(size=12), title=None),
            showlegend=False,
            margin=dict(l=16, r=16, t=60, b=20),
        )
    )
    return fig

def build_geo_chart():
    state_totals = (st_m.groupby("state")["spend"].sum().reset_index()
                       .query("state in @US_STATES"))
    # Growth rate per state
    state_q1  = (st_m[st_m["year_month"] <= 202503].groupby("state")["spend"].mean())
    state_q4  = (st_m[st_m["year_month"] >= 202510].groupby("state")["spend"].mean())
    state_gr  = ((state_q4 - state_q1) / (state_q1.replace(0, np.nan)) * 100).rename("growth_pct")
    state_df  = state_totals.merge(state_gr.reset_index(), on="state", how="left")

    fig = go.Figure(go.Choropleth(
        locations=state_df["state"],
        z=state_df["spend"] / 1e6,
        locationmode="USA-states",
        colorscale=[[0,"#1e3a5f"],[0.5,"#2563eb"],[1,"#93c5fd"]],
        colorbar=dict(title=dict(text="Spend ($M)", font=dict(color=C_TEXT, size=11)),
                      tickformat="$,.0f", len=0.6, thickness=12,
                      tickfont=dict(color=C_TEXT, size=10),
                      bgcolor=C_CARD, bordercolor=C_BORDER, borderwidth=1),
        hovertemplate="<b>%{location}</b><br>Spend: $%{z:.1f}M<extra></extra>",
    ))
    fig.update_layout(
        **dark_layout(
            title=dict(text="Geographic Spend Distribution — 2025 Total ($M)",
                       font=dict(size=14), x=0.01),
            height=580,
            geo=dict(scope="usa", bgcolor=C_CARD,
                     lakecolor="#d0e8f5", landcolor="#e8edf2",
                     showlakes=True, showcoastlines=False,
                     subunitcolor=C_BORDER),
            margin=dict(l=16, r=16, t=60, b=20),
        )
    )
    return fig

def build_concentration_chart():
    df = conc.copy()
    gini = float(df["gini"].iloc[0])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["client_pct"], y=df["cum_pct"],
        mode="lines", name="Actual (Lorenz curve)",
        line=dict(color=C_RED, width=2.5),
        fill="tonexty", fillcolor="rgba(218,41,28,0.15)",
        hovertemplate="Top %{x:.1f}% clients → %{y:.1f}% of spend<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=[0, 100], y=[0, 100], mode="lines",
        name="Perfect equality", line=dict(color=C_MUTED, dash="dot", width=1)
    ))
    # Annotate 80% line
    idx_80 = (df["cum_pct"] - 80).abs().idxmin()
    fig.add_hline(y=80, line_dash="dot", line_color=C_AMBER, line_width=1)
    fig.add_vline(x=float(df.loc[idx_80,"client_pct"]),
                  line_dash="dot", line_color=C_AMBER, line_width=1,
                  annotation_text=f"  Top {df.loc[idx_80,'client_pct']:.1f}% clients = 80% revenue",
                  annotation_font=dict(color=C_AMBER, size=11))
    fig.update_layout(
        **dark_layout(
            title=dict(text=f"Revenue Concentration — Lorenz Curve  (Gini = {gini:.3f})",
                       font=dict(size=14), x=0.01),
            height=480,
            legend=dict(orientation="h", y=1.08, font=dict(size=12)),
            xaxis=dict(**AXIS_BASE, title=dict(text="Cumulative % of Clients", font=dict(size=12)),
                       ticksuffix="%", tickfont=dict(size=11)),
            yaxis=dict(**AXIS_BASE, title=dict(text="Cumulative % of Revenue", font=dict(size=12)),
                       ticksuffix="%", tickfont=dict(size=11)),
            margin=dict(l=16, r=16, t=60, b=20),
        )
    )
    return fig

def build_feature_importance():
    df = fi.sort_values("importance")
    labels = {
        "avg_locations":    "Avg Locations per Segment",
        "trend_volatility": "Spend Volatility",
        "ecomm_penetration":"Ecomm Adoption Rate",
        "trend_slope_norm": "Spend Trend Slope",
        "trend_r2":         "Trend Consistency (R²)",
        "trend_momentum":   "H2 vs H1 Momentum",
        "category_breadth": "Category Breadth",
    }
    df["label"] = df["feature"].map(labels).fillna(df["feature"])
    df["pct"]   = df["importance"] * 100

    fig = go.Figure(go.Bar(
        x=df["pct"], y=df["label"],
        orientation="h",
        marker=dict(
            color=df["pct"],
            colorscale=[[0,"#1e3a5f"],[0.5,"#2563eb"],[1,"#60a5fa"]],
            showscale=False
        ),
        text=[f"{v:.1f}%" for v in df["pct"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.1f}%<extra></extra>"
    ))
    fig.update_layout(
        **dark_layout(
            title=dict(text="What Drives High-Revenue Segments?  (Random Forest)",
                       font=dict(size=14), x=0.01),
            height=420,
            xaxis=dict(**AXIS_BASE, ticksuffix="%", title=None, tickfont=dict(size=11)),
            yaxis=dict(**AXIS_BASE, title=None, tickfont=dict(size=12)),
            margin=dict(l=16, r=60, t=60, b=20),
        )
    )
    return fig


def build_segment_detail_chart(seg):
    """Single-segment trend + 3-month forecast. Called dynamically."""
    grp  = (seg_m[seg_m["segment"]==seg].sort_values("year_month")
                .set_index("year_month").reindex(MONTHS, fill_value=0))
    vals = grp["spend"].values
    fc   = seg_forecasts.get(seg, {})
    preds= fc.get("preds", [vals[-1]]*3)
    lo   = fc.get("lo",    preds)
    hi   = fc.get("hi",    preds)
    model= fc.get("model", "—")
    cls  = seg_f.loc[seg_f["segment"]==seg, "strategic_class"].values
    color= CLASS_COLORS.get(cls[0] if len(cls) else "Monitor", C_BLUE)
    ALL_X= MONTH_LBLS + FORECAST_LBLS

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=FORECAST_LBLS+FORECAST_LBLS[::-1], y=hi+lo[::-1],
        fill="toself",
        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.18)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Confidence range", hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=MONTH_LBLS, y=vals, name="Actual",
        mode="lines+markers",
        line=dict(color=color, width=2.5),
        marker=dict(size=6),
        hovertemplate="%{x}: $%{y:,.0f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=[MONTH_LBLS[-1], FORECAST_LBLS[0]],
        y=[float(vals[-1]), preds[0]],
        mode="lines", line=dict(color=color, dash="dot", width=2),
        showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=FORECAST_LBLS, y=preds, name=f"Forecast ({model})",
        mode="lines+markers",
        line=dict(color=color, width=2.5, dash="dash"),
        marker=dict(size=9, symbol="diamond"),
        hovertemplate="%{x}: $%{y:,.0f}<extra></extra>"
    ))
    fig.add_vline(x=len(MONTH_LBLS)-0.5, line_dash="dot",
                  line_color=C_MUTED, line_width=1)
    fig.update_layout(
        **dark_layout(
            title=dict(text=SEG_LABELS.get(seg, seg) + " — Monthly Spend + Q1 2026 Forecast",
                       font=dict(size=14), x=0.01),
            height=340,
            hovermode="x unified",
            legend=dict(orientation="h", y=1.10, font=dict(size=11)),
            yaxis=dict(**AXIS_BASE, tickformat="$,.0f", tickfont=dict(size=11)),
            xaxis=dict(**AXIS_BASE, tickfont=dict(size=11),
                       categoryorder="array", categoryarray=ALL_X),
            margin=dict(l=16, r=16, t=55, b=20),
        )
    )
    return fig

# ── Pre-build all charts ───────────────────────────────────────────────────────
print("Building charts …")
fig_portfolio    = build_portfolio_chart()
fig_momentum     = build_segment_momentum_bar()
fig_category     = build_category_bars()
fig_geo          = build_geo_chart()

fig_importance   = build_feature_importance()
print("  Charts ready.")

# ── KPI data ──────────────────────────────────────────────────────────────────
total_spend     = port["spend"].sum()
forecast_q1     = sum(p_preds)
cl_total        = conc["total_spend"].sum()
gini_val        = float(conc["gini"].iloc[0])
top10_pct       = float(conc[conc["client_rank"]==10]["cum_pct"].values[0]) if len(conc)>=10 else 0
protect_count   = int((seg_f["strategic_class"]=="Protect & Grow").sum())
at_risk_count   = int((seg_f["strategic_class"]=="Revenue at Risk").sum())
emerging_count  = int((seg_f["strategic_class"]=="Emerging Opportunity").sum())
total_avendra_savings = port["savings"].sum()
avendra_adv_pct = total_avendra_savings / (port["avendra_price"].sum() + 1e-9) * 100

# ── Dash App ───────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                title="Aramark Board Intelligence")

def kpi_block(label, value, sub, accent):
    return html.Div([
        html.P(label, style={"fontSize":"10px","color": C_MUTED,"textTransform":"uppercase",
                              "letterSpacing":"0.8px","margin":"0 0 4px 0","fontWeight":"600"}),
        html.H2(value, style={"color": accent,"fontWeight":"800","margin":"0","fontSize":"26px",
                               "lineHeight":"1.1","fontVariantNumeric":"tabular-nums"}),
        html.P(sub, style={"fontSize":"11px","color": C_MUTED,"margin":"4px 0 0 0"}),
    ], style={"borderLeft": f"3px solid {accent}","paddingLeft":"14px"})

KPI_ROW = dbc.Row([
    dbc.Col(kpi_block("Total 2025 Portfolio Spend", fmt(total_spend),
                      f"vs Avendra: saving {fmt(total_avendra_savings)} ({avendra_adv_pct:.1f}%)", C_BLUE), md=3),
    dbc.Col(kpi_block("Q1 2026 Forecast", fmt(forecast_q1),
                      f"Best model: {p_model} | Jan+Feb+Mar combined", C_RED), md=3),
    dbc.Col(kpi_block("Revenue Concentration", f"Gini {gini_val:.2f}",
                      f"Top 10 clients = {top10_pct:.1f}% of revenue", C_AMBER), md=3),
    dbc.Col(kpi_block("Segment Health",
                      f"{protect_count} Strong · {at_risk_count} At Risk",
                      f"{emerging_count} Emerging opportunities identified", C_GREEN), md=3),
], className="g-4", style={"padding":"28px 36px 24px 36px",
                             "borderBottom": f"1px solid {C_BORDER}"})

def card(children):
    return html.Div(children, style={
        "background": C_CARD, "borderRadius": "12px",
        "padding": "24px 24px 20px 24px",
        "border": f"1px solid {C_BORDER}",
        "boxShadow": "0 1px 4px rgba(0,0,0,0.05)",
    })

def ctrl_label(text):
    return html.P(text, style={"fontSize":"10px","color": C_MUTED,
                                "textTransform":"uppercase","letterSpacing":"0.8px",
                                "marginBottom":"6px","fontWeight":"600","margin":"0 0 6px 0"})

G = lambda fig: dcc.Graph(figure=fig, config={"displayModeBar": False})
PAD = {"padding": "28px 36px"}

# ── Segment options for dropdowns ──────────────────────────────────────────────
SEG_OPTIONS = [
    {"label": SEG_LABELS.get(s, s), "value": s}
    for s in seg_f.sort_values("total_spend", ascending=False)["segment"]
]
CLASS_OPTIONS = [{"label": c, "value": c} for c in CLASS_COLORS]
DEFAULT_FIRST_SEG = SEG_OPTIONS[0]["value"]

def tab_content(tab_id):
    if tab_id == "overview":
        return html.Div([
            dbc.Row([
                dbc.Col(card([G(fig_portfolio)]),  md=8),
                dbc.Col(card([G(fig_importance)]), md=4),
            ], className="g-4 mb-4"),
            dbc.Row([
                dbc.Col(card([G(fig_category)]), md=12),
            ], className="g-4"),
        ], style=PAD)

    if tab_id == "strategy":
        # Controls bar above the charts
        controls = html.Div([
            dbc.Row([
                dbc.Col([
                    ctrl_label("Filter by Class"),
                    dcc.Checklist(
                        id="matrix-class-filter",
                        options=CLASS_OPTIONS,
                        value=[c["value"] for c in CLASS_OPTIONS],
                        inline=True,
                        inputStyle={"marginRight": "5px", "accentColor": C_RED},
                        labelStyle={"color": C_TEXT, "fontSize": "12px", "fontWeight": "500",
                                    "marginRight": "16px", "cursor": "pointer"},
                    )
                ], md=7),
                dbc.Col([
                    ctrl_label("Minimum Spend ($M)"),
                    dcc.Slider(
                        id="matrix-spend-slider",
                        min=0, max=int(seg_f["total_spend"].max()/1e6),
                        step=10, value=0,
                        marks={0:"$0", 250:"$250M", 500:"$500M",
                               750:"$750M", 1000:"$1B"},
                        tooltip={"placement":"bottom","always_visible":False},
                    )
                ], md=5),
            ], className="g-3 align-items-end"),
        ], style={"background": C_CARD, "borderRadius":"12px",
                   "padding":"20px 24px", "border": f"1px solid {C_BORDER}",
                   "boxShadow":"0 1px 4px rgba(0,0,0,0.05)",
                   "marginBottom":"20px"})

        return html.Div([
            controls,
            dbc.Row([
                dbc.Col(card([
                    dcc.Graph(
                        id="matrix-chart",
                        config={"displayModeBar": False},
                        style={"height": "600px"},
                    )
                ]), md=12),
            ], className="g-4 mb-4"),
            dbc.Row([
                dbc.Col(card([G(fig_momentum)]), md=12),
            ], className="g-4"),
        ], style=PAD)

    if tab_id == "forecast":
        controls = html.Div([
            dbc.Row([
                dbc.Col([
                    ctrl_label("Segment Deep-Dive"),
                    dcc.Dropdown(
                        id="forecast-seg-dd",
                        options=SEG_OPTIONS,
                        value=DEFAULT_FIRST_SEG,
                        clearable=False,
                        style={"fontSize":"12px", "color":"#0f172a"},
                    )
                ], md=5),
                dbc.Col([
                    html.Br(),
                    html.P("Select a segment to see its monthly trend and Q1 2026 ML forecast.",
                           style={"color": C_MUTED, "fontSize":"12px", "marginTop":"6px",
                                  "fontStyle":"italic"})
                ], md=7),
            ], className="g-3 align-items-end"),
        ], style={"background": C_CARD, "borderRadius":"12px",
                   "padding":"20px 24px", "border": f"1px solid {C_BORDER}",
                   "boxShadow":"0 1px 4px rgba(0,0,0,0.05)",
                   "marginBottom":"20px"})

        return html.Div([
            controls,
            dbc.Row([
                dbc.Col(card([dcc.Graph(id="forecast-detail-chart",
                                        config={"displayModeBar": False},
                                        style={"height": "500px"})]), md=12),
            ], className="g-4"),
        ], style=PAD)

    if tab_id == "risk":
        return html.Div([
            dbc.Row([
                dbc.Col(card([G(fig_geo)]), md=12),
            ], className="g-4"),
        ], style=PAD)

    return html.Div()

TAB_STYLE     = {"color": C_MUTED, "backgroundColor": C_CARD,
                  "border": "none", "padding": "12px 22px",
                  "fontWeight": "600", "fontSize": "13px"}
TAB_SELECTED  = {**TAB_STYLE, "color": C_BLUE,
                  "borderBottom": f"2px solid {C_RED}",
                  "backgroundColor": C_CARD}

app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.Span("ARAMARK", style={"color": C_BLUE, "fontWeight": "900",
                                         "fontSize": "18px", "letterSpacing": "2px"}),
            html.Span(" · Board Intelligence", style={"color": C_MUTED,
                                                       "fontSize": "14px", "marginLeft": "8px"}),
        ]),
        html.Div("FY 2025  ·  28 Segments  ·  6,375 Clients  ·  43,955 Locations",
                 style={"color": C_MUTED, "fontSize": "11px", "marginTop": "3px"})
    ], style={"background": C_CARD, "padding": "16px 28px",
               "borderBottom": f"1px solid {C_BORDER}",
               "boxShadow": "0 1px 3px rgba(0,0,0,0.06)"}),

    # KPIs
    html.Div(KPI_ROW, style={"background": C_CARD,
                               "borderBottom": f"1px solid {C_BORDER}"}),

    # Tabs
    html.Div([
        dcc.Tabs(id="tabs", value="overview", children=[
            dcc.Tab(label="📊  Portfolio Overview", value="overview",
                    style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="🎯  Segment Strategy",   value="strategy",
                    style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="🔮  Segment Forecasts",  value="forecast",
                    style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="⚠️  Risk & Geography",  value="risk",
                    style=TAB_STYLE, selected_style=TAB_SELECTED),
        ], style={"backgroundColor": C_CARD, "borderBottom": f"1px solid {C_BORDER}"},
           colors={"border": "transparent", "primary": C_RED, "background": C_CARD}),
        html.Div(id="tab-content", style={"background": C_BG,
                                           "minHeight": "calc(100vh - 180px)"})
    ]),

], style={"background": C_BG, "minHeight": "100vh",
           "fontFamily": "'Inter','Segoe UI',sans-serif", "color": C_TEXT})

@app.callback(Output("tab-content","children"), Input("tabs","value"))
def render_tab(tab): return tab_content(tab)

@app.callback(
    Output("matrix-chart", "figure"),
    Input("matrix-class-filter", "value"),
    Input("matrix-spend-slider", "value"),
)
def update_matrix(selected_classes, min_spend_m):
    df = seg_f.copy()
    df = df[df["strategic_class"].isin(selected_classes or list(CLASS_COLORS))]
    df = df[df["total_spend"] >= (min_spend_m or 0) * 1e6]
    df["spend_M"]  = df["total_spend"] / 1e6
    df["mom_pct"]  = df["trend_momentum"] * 100

    fig = go.Figure()
    for cls, color in CLASS_COLORS.items():
        sub = df[df["strategic_class"] == cls]
        if sub.empty: continue
        fig.add_trace(go.Scatter(
            x=sub["spend_M"], y=sub["mom_pct"],
            mode="markers",
            name=cls,
            marker=dict(
                size=np.sqrt(sub["avg_locations"].clip(1)) * 7,
                color=color, opacity=0.88,
                line=dict(color="white", width=1.5)
            ),
            customdata=np.column_stack([
                sub["label"], sub["avg_locations"].round(0),
                (sub["avendra_advantage_pct"]).round(1)
            ]),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Spend: $%{x:.1f}M<br>"
                "Momentum: %{y:+.1f}%<br>"
                "Avg Locations: %{customdata[1]}<br>"
                "Avendra Adv.: %{customdata[2]}%"
                "<extra>" + cls + "</extra>"
            )
        ))

    if not df.empty:
        med = df["spend_M"].median()
        x_max = df["spend_M"].max() * 1.08
        y_max = df["mom_pct"].max() * 1.1 if df["mom_pct"].max() > 0 else 10
        y_min = df["mom_pct"].min() * 1.1 if df["mom_pct"].min() < 0 else -10
        fig.add_vline(x=med, line_dash="dot", line_color=C_MUTED, line_width=1)
        fig.add_hline(y=0,   line_dash="dot", line_color=C_MUTED, line_width=1)
        for txt, xp, ya, col, anch in [
            ("PROTECT & GROW",       x_max*0.97, y_max*0.90, C_GREEN,  "right"),
            ("EMERGING OPPORTUNITY", med*0.05,   y_max*0.90, C_BLUE,   "left"),
            ("REVENUE AT RISK",      x_max*0.97, y_min*0.90, C_RED,    "right"),
            ("MONITOR",              med*0.05,   y_min*0.90, C_AMBER,  "left"),
        ]:
            fig.add_annotation(text=txt, x=xp, y=ya, xanchor=anch,
                               showarrow=False,
                               font=dict(size=10, color=col), opacity=0.55)

    fig.update_layout(
        **dark_layout(
            title=dict(text="Strategic Matrix — Hover for details  |  Bubble size = avg locations",
                       font=dict(size=14), x=0.01),
            height=600,
            autosize=True,
            legend=dict(orientation="h", y=-0.08, font=dict(size=12)),
            xaxis=dict(**AXIS_BASE, title=dict(text="Total Spend ($M)", font=dict(size=12)),
                       tickprefix="$", ticksuffix="M", tickfont=dict(size=11)),
            yaxis=dict(**AXIS_BASE, title=dict(text="Growth Momentum (%)", font=dict(size=12)),
                       ticksuffix="%", tickfont=dict(size=11)),
            margin=dict(l=60, r=60, t=60, b=90),
        )
    )
    return fig


@app.callback(
    Output("forecast-detail-chart", "figure"),
    Input("forecast-seg-dd", "value"),
)
def update_forecast_detail(seg):
    if not seg:
        seg = DEFAULT_FIRST_SEG
    return build_segment_detail_chart(seg)


if __name__ == "__main__":
    print("\nBoard Dashboard → http://127.0.0.1:8050\n")
    app.run(debug=False, port=8050)
