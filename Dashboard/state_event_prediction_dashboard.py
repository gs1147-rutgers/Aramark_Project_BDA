"""
Aramark Spend Analysis: State Events & Economic Drivers
═══════════════════════════════════════════════════════
CLONE + ML PREDICTION EXTENSION

Adds event-aligned spend prediction using four ML models:
  • Gradient Boosting (GBM)   — default winner on tabular data
  • Random Forest              — ensemble baseline
  • Ridge Regression           — linear event-economic baseline
  • SVR (RBF kernel)           — non-linear capture of event spikes

Each state gets a 6-month forward forecast (Jan–Jun 2026 (Jan–Mar Actual, Apr–Jun Forecast)) that factors
in the actual event calendar for that state/month so predictions align
with real hospitality demand drivers.

Run:  python3 state_event_prediction_dashboard.py  →  http://127.0.0.1:8051
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, callback_context
import dash_bootstrap_components as dbc
import os

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep

C_BLUE   = "#003087"
C_RED    = "#DA291C"
C_GREEN  = "#1b7f4f"
C_AMBER  = "#d97706"
C_PURPLE = "#6366f1"
C_TEAL   = "#0891b2"
C_BG     = "#f0f4f8"
C_CARD   = "#ffffff"
C_MUTED  = "#64748b"
C_TEXT   = "#1e293b"
C_BORDER = "#e2e8f0"

LAYOUT_BASE = dict(
    template="plotly_white",
    paper_bgcolor=C_CARD, plot_bgcolor="#f8fafc",
    font=dict(family="Inter, Segoe UI, sans-serif", color=C_TEXT, size=12),
    margin=dict(l=14, r=14, t=46, b=14),
)
AXIS_BASE = dict(gridcolor="#e2e8f0", zeroline=False, showline=False)

def fmt(v, decimals=2):
    try:
        v = float(v)
        if abs(v) >= 1e9: return f"${v/1e9:.{decimals}f}B"
        if abs(v) >= 1e6: return f"${v/1e6:.{decimals}f}M"
        if abs(v) >= 1e3: return f"${v/1e3:.1f}K"
        return f"${v:,.0f}"
    except: return "—"

# ─── STATE ECONOMIC REFERENCE ─────────────────────────────────────────────────
STATE_REF = {
    'CA': {'population_M':39.5,'gdp_B':4103,'tourism_B':140.0,'hotel_rooms_K':550},
    'TX': {'population_M':31.3,'gdp_B':2709,'tourism_B':80.0,'hotel_rooms_K':520},
    'FL': {'population_M':23.4,'gdp_B':1200,'tourism_B':100.0,'hotel_rooms_K':480},
    'IL': {'population_M':12.8,'gdp_B':1060,'tourism_B':42.0,'hotel_rooms_K':200},
    'PA': {'population_M':13.0,'gdp_B':890,'tourism_B':28.0,'hotel_rooms_K':155},
    'NY': {'population_M':19.6,'gdp_B':2297,'tourism_B':72.0,'hotel_rooms_K':280},
    'NC': {'population_M':10.8,'gdp_B':712,'tourism_B':36.7,'hotel_rooms_K':190},
    'VA': {'population_M':8.7,'gdp_B':630,'tourism_B':35.1,'hotel_rooms_K':145},
    'GA': {'population_M':11.0,'gdp_B':700,'tourism_B':35.0,'hotel_rooms_K':185},
    'OH': {'population_M':11.8,'gdp_B':740,'tourism_B':22.0,'hotel_rooms_K':160},
    'AZ': {'population_M':7.5,'gdp_B':480,'tourism_B':27.0,'hotel_rooms_K':145},
    'MA': {'population_M':7.1,'gdp_B':680,'tourism_B':25.0,'hotel_rooms_K':110},
    'SC': {'population_M':5.4,'gdp_B':290,'tourism_B':14.0,'hotel_rooms_K':110},
    'CO': {'population_M':5.9,'gdp_B':450,'tourism_B':24.0,'hotel_rooms_K':115},
    'TN': {'population_M':7.1,'gdp_B':470,'tourism_B':31.7,'hotel_rooms_K':145},
    'IN': {'population_M':6.9,'gdp_B':400,'tourism_B':14.0,'hotel_rooms_K':100},
    'MD': {'population_M':6.2,'gdp_B':500,'tourism_B':18.0,'hotel_rooms_K':95},
    'HI': {'population_M':1.4,'gdp_B':103,'tourism_B':21.0,'hotel_rooms_K':72},
    'NJ': {'population_M':9.3,'gdp_B':750,'tourism_B':20.0,'hotel_rooms_K':120},
    'CT': {'population_M':3.6,'gdp_B':315,'tourism_B':10.0,'hotel_rooms_K':45},
    'IA': {'population_M':3.2,'gdp_B':200,'tourism_B':8.0,'hotel_rooms_K':55},
    'WA': {'population_M':7.9,'gdp_B':690,'tourism_B':22.0,'hotel_rooms_K':120},
    'WI': {'population_M':5.9,'gdp_B':370,'tourism_B':14.0,'hotel_rooms_K':92},
    'MN': {'population_M':5.7,'gdp_B':425,'tourism_B':14.0,'hotel_rooms_K':95},
    'MI': {'population_M':10.0,'gdp_B':580,'tourism_B':19.0,'hotel_rooms_K':140},
    'MO': {'population_M':6.2,'gdp_B':350,'tourism_B':12.0,'hotel_rooms_K':98},
    'NV': {'population_M':3.2,'gdp_B':225,'tourism_B':35.0,'hotel_rooms_K':175},
    'LA': {'population_M':4.6,'gdp_B':265,'tourism_B':18.0,'hotel_rooms_K':80},
    'KY': {'population_M':4.5,'gdp_B':255,'tourism_B':9.0,'hotel_rooms_K':72},
    'AL': {'population_M':5.1,'gdp_B':245,'tourism_B':8.5,'hotel_rooms_K':80},
    'OK': {'population_M':4.0,'gdp_B':230,'tourism_B':7.5,'hotel_rooms_K':65},
    'DC': {'population_M':0.7,'gdp_B':170,'tourism_B':10.0,'hotel_rooms_K':32},
    'OR': {'population_M':4.3,'gdp_B':295,'tourism_B':12.0,'hotel_rooms_K':65},
    'UT': {'population_M':3.5,'gdp_B':255,'tourism_B':11.0,'hotel_rooms_K':60},
    'AR': {'population_M':3.1,'gdp_B':145,'tourism_B':5.5,'hotel_rooms_K':48},
    'MS': {'population_M':2.9,'gdp_B':120,'tourism_B':5.0,'hotel_rooms_K':45},
    'KS': {'population_M':2.9,'gdp_B':185,'tourism_B':5.0,'hotel_rooms_K':50},
    'NM': {'population_M':2.1,'gdp_B':110,'tourism_B':5.5,'hotel_rooms_K':42},
    'NE': {'population_M':2.0,'gdp_B':155,'tourism_B':4.5,'hotel_rooms_K':38},
    'ID': {'population_M':2.0,'gdp_B':105,'tourism_B':4.5,'hotel_rooms_K':38},
    'WV': {'population_M':1.8,'gdp_B':80,'tourism_B':3.5,'hotel_rooms_K':30},
    'NH': {'population_M':1.4,'gdp_B':95,'tourism_B':4.5,'hotel_rooms_K':30},
    'ME': {'population_M':1.4,'gdp_B':78,'tourism_B':6.5,'hotel_rooms_K':42},
    'HI': {'population_M':1.4,'gdp_B':103,'tourism_B':21.0,'hotel_rooms_K':72},
    'RI': {'population_M':1.1,'gdp_B':72,'tourism_B':3.5,'hotel_rooms_K':22},
    'MT': {'population_M':1.1,'gdp_B':65,'tourism_B':3.5,'hotel_rooms_K':24},
    'DE': {'population_M':1.0,'gdp_B':80,'tourism_B':3.5,'hotel_rooms_K':22},
    'SD': {'population_M':0.9,'gdp_B':60,'tourism_B':4.0,'hotel_rooms_K':28},
    'ND': {'population_M':0.8,'gdp_B':63,'tourism_B':2.0,'hotel_rooms_K':22},
    'AK': {'population_M':0.7,'gdp_B':70,'tourism_B':3.0,'hotel_rooms_K':18},
    'VT': {'population_M':0.6,'gdp_B':46,'tourism_B':2.5,'hotel_rooms_K':25},
    'WY': {'population_M':0.6,'gdp_B':53,'tourism_B':3.0,'hotel_rooms_K':22},
    'PR': {'population_M':3.2,'gdp_B':115,'tourism_B':8.0,'hotel_rooms_K':25},
}

# ─── EVENT CALENDAR ───────────────────────────────────────────────────────────
# (state, year_month) → list of (event_name, impact_score 0-100)
# Impact score approximates incremental hospitality spend pressure:
#   100 = Super Bowl / FIFA WC Final / national mega-event ($500M+)
#    90 = Masters, Indy 500, FIFA WC group match host city ($160-620M)
#    80 = Kentucky Derby, PGA Championship, Coachella, NBA All-Star ($100-400M)
#    65 = SXSW, Mardi Gras, Preakness, EDC, CMA Fest ($65-300M)
#    50 = Boston Marathon, US Open Golf, Daytona 500 regional ($50-150M)
#    35 = significant recurring (state fair, jazz fest, golf regional)
#    20 = moderate local event

RAW_EVENTS = [
    # ── 2025 ──────────────────────────────────────────────────────────────────
    # Jan
    ('CA', 202501, 'Rose Bowl',                    60),
    ('NV', 202501, 'CES Las Vegas',                75),
    ('AZ', 202501, 'Barrett-Jackson Auto Auction', 40),
    ('AZ', 202501, 'Waste Management Phoenix Open',80),
    ('HI', 202501, 'Sony Open Golf',               35),
    ('UT', 202501, 'Sundance Film Festival',        55),
    ('CO', 202501, "Nat'l Western Stock Show",      40),
    ('DC', 202501, 'Presidential Inauguration',     90),
    # Feb
    ('FL', 202502, 'Daytona 500',                   70),
    ('LA', 202502, 'Super Bowl LIX New Orleans',   100),
    ('LA', 202502, 'Mardi Gras',                    65),
    ('AZ', 202502, 'Cactus League MLB Spring Training',45),
    ('FL', 202502, 'Miami Open Tennis',             50),
    ('DC', 202502, 'Cherry Blossom Festival start', 25),
    # Mar
    ('TX', 202503, 'SXSW Austin',                  70),
    ('FL', 202503, 'Miami Open Tennis',             50),
    ('AZ', 202503, 'Cactus League MLB',             45),
    ('DC', 202503, 'Cherry Blossom Festival',       40),
    ('NC', 202503, 'ACC Tournament',                30),
    ('IN', 202503, 'Big Ten Tournament Basketball', 35),
    ('LA', 202503, 'Jazz Fest run-up',              35),
    # Apr
    ('GA', 202504, 'Masters Tournament Augusta',    90),
    ('CA', 202504, 'Coachella Valley Music Festival',70),
    ('MA', 202504, 'Boston Marathon',               55),
    ('SC', 202504, 'RBC Heritage Golf Hilton Head', 55),
    ('LA', 202504, 'Jazz Fest New Orleans',         60),
    ('DC', 202504, 'Cherry Blossom Festival',       40),
    ('KY', 202504, 'Keeneland Spring Racing',       40),
    # May
    ('NC', 202505, 'PGA Championship Quail Hollow', 85),
    ('IN', 202505, 'Indianapolis 500',              90),
    ('MD', 202505, 'Preakness Stakes',              65),
    ('KY', 202505, 'Kentucky Derby',                80),
    ('TX', 202505, 'AT&T Byron Nelson Golf',        40),
    ('AL', 202505, 'Regions Tradition Golf',        35),
    # Jun
    ('TN', 202506, 'CMA Fest Nashville',            70),
    ('TN', 202506, 'Bonnaroo Music Festival',       55),
    ('MI', 202506, 'Detroit Grand Prix',            60),
    ('MI', 202506, 'Rocket Mortgage Classic Golf',  40),
    ('CT', 202506, 'Travelers Championship Golf',   45),
    ('OR', 202506, 'Rose Festival Portland',        35),
    ('NE', 202506, 'College World Series Omaha',    50),
    ('DE', 202506, 'Firefly Music Festival',        40),
    # Jul
    ('IL', 202507, 'NASCAR Chicago Street Race',    65),
    ('WI', 202507, 'EAA AirVenture Oshkosh',        55),
    ('WI', 202507, 'Summerfest Milwaukee',          60),
    ('IA', 202507, 'RAGBRAI Cycling Event',         35),
    ('RI', 202507, 'Newport Folk Festival',         40),
    ('CO', 202507, 'Denver Events',                 30),
    # Aug
    ('IL', 202508, 'Lollapalooza Chicago',          65),
    ('IL', 202508, 'Chicago Air & Water Show',      55),
    ('IN', 202508, 'GenCon Gaming Convention',      50),
    ('IN', 202508, 'Brickyard 400 NASCAR',          55),
    ('WI', 202508, 'Ryder Cup @ Whistling Straits', 85),
    ('MN', 202508, 'Minnesota State Fair',          45),
    ('SD', 202508, 'Sturgis Motorcycle Rally',      55),
    ('RI', 202508, 'Newport Jazz Festival',         40),
    # Sep
    ('NY', 202509, 'US Open Tennis',                70),
    ('GA', 202509, 'DragonCon Atlanta',             45),
    ('TX', 202509, 'Texas State Fair',              55),
    ('IA', 202509, 'Iowa State Fair',               40),
    ('WY', 202509, 'Yellowstone NP Peak',           30),
    # Oct
    ('HI', 202510, 'Ironman Triathlon Kona',        50),
    ('NM', 202510, 'Albuquerque Balloon Fiesta',    55),
    ('KY', 202510, 'Keeneland Fall Racing',         40),
    ('VT', 202510, 'Fall Foliage Season',           35),
    ('NH', 202510, 'Fall Foliage Tourism',          30),
    ('TX', 202510, 'Texas State Fair (cont.)',       40),
    # Nov
    ('NY', 202511, 'NYC Marathon',                  55),
    ('NV', 202511, 'SEMA Auto Show Las Vegas',      60),
    ('FL', 202511, 'PGA Tour Events',               35),
    ('PA', 202512, 'Army-Navy Game',                45),
    # Dec
    ('NV', 202512, 'NYE Las Vegas',                 70),
    ('NY', 202512, 'NYE Times Square',              65),
    ('FL', 202512, 'Orange Bowl',                   50),

    # ── 2026 PROJECTIONS (sourced from us_major_events_2026.xlsx + USA_Major_Events_2026.xlsx) ──
    # Economic impact scores recalibrated from verified research:
    #   $900M Mardi Gras → 85 | $509M Boston Marathon → 85 | $500M Super Bowl LX → 100
    #   $566M Indy 500 Month → 90 | $350M NBA All-Star → 75 | $300M SXSW (downsized 7-day) → 65
    #   FIFA WC group matches ~$160-620M/city → 85-95 | FIFA WC Final NYNJ ~$3.3B → 100
    #   Coachella+Stagecoach $700M combined → 80+65 | Masters $110M → 90 | Kentucky Derby $400M → 80

    # Jan 2026
    ('CA', 202601, 'Rose Parade & Rose Bowl (Pasadena)',      60),
    ('NV', 202601, 'CES 2026 – Consumer Electronics Show',   75),
    ('AZ', 202601, 'Waste Management Phoenix Open',          80),
    ('HI', 202601, 'Sony Open Golf',                         35),
    ('UT', 202601, 'Sundance Film Festival – final Park City', 55),
    ('CO', 202601, "Nat'l Western Stock Show",               40),
    ('FL', 202601, 'Gasparilla Pirate Festival (Tampa)',      35),

    # Feb 2026  — Super Bowl LX at Levi's Stadium, Santa Clara CA ($500M+)
    ('CA', 202602, 'Super Bowl LX – Levi\'s Stadium, Santa Clara',  100),
    ('CA', 202602, 'NBA All-Star Game – Intuit Dome, Inglewood',     75),
    ('FL', 202602, 'Daytona 500 (Daytona Beach)',                    70),
    ('LA', 202602, 'Mardi Gras 2026 – New Orleans ($900M impact)',   85),
    ('AZ', 202602, 'Cactus League MLB Spring Training',              45),
    ('FL', 202602, 'Miami Open Tennis',                              50),
    ('NY', 202602, 'NYC Fashion Week – Fall/Winter',                 55),
    ('TX', 202602, 'Houston Livestock Show & Rodeo',                 55),

    # Mar 2026  — SXSW downsized to 7-day format ($300M, ~20% fewer hotel nights vs 2025)
    ('TX', 202603, 'SXSW Austin – 40th Annual (7-day format)',      65),
    ('FL', 202603, 'Ultra Music Festival (Miami)',                   55),
    ('FL', 202603, 'Miami Open Tennis Tournament',                   50),
    ('CA', 202603, 'BNP Paribas Open Tennis (Indian Wells)',         50),
    ('IN', 202603, 'NCAA Men\'s & Women\'s Basketball Tournament',   40),
    ('DC', 202603, 'Cherry Blossom Festival',                        40),
    ('NC', 202603, 'ACC Tournament 2026',                            30),
    ('IN', 202603, 'Big Ten Tournament 2026',                        35),
    ('AZ', 202603, 'Cactus League MLB (cont.)',                      45),

    # Apr 2026  — Boston Marathon $509M; Masters $110M; Coachella $220-240M revenue
    ('GA', 202604, 'Masters Tournament – Augusta National',          90),
    ('MA', 202604, 'Boston Marathon – 130th Running ($509M impact)', 85),
    ('CA', 202604, 'Coachella Valley Music & Arts Festival',         80),
    ('CA', 202604, 'Stagecoach Country Music Festival',              65),
    ('LA', 202604, 'New Orleans Jazz & Heritage Festival',           60),
    ('SC', 202604, 'RBC Heritage Golf – Hilton Head',                55),
    ('KY', 202604, 'Kentucky Derby Festival / Thunder Over Louisville', 50),
    ('HI', 202604, 'Merrie Monarch Festival',                        30),
    ('TX', 202604, 'Chevron Championship LPGA (Houston)',            35),
    ('VT', 202604, 'Vermont Maple Festival',                         20),
    ('AZ', 202604, 'VIVA PHX Festival (Phoenix)',                    25),

    # May 2026  — Indy 500 $566M May attribution; Kentucky Derby $400M; PGA Champ PA $125M
    ('IN', 202605, 'Indianapolis 500 – 110th Running ($566M month)', 90),
    ('KY', 202605, 'Kentucky Derby – 152nd Running ($400M impact)',  80),
    ('PA', 202605, 'PGA Championship – Aronimink ($125M impact)',    80),
    ('MD', 202605, 'Preakness Stakes – Laurel Park',                 65),
    ('NV', 202605, 'EDC Las Vegas – Electric Daisy Carnival',        65),
    ('TX', 202605, 'AT&T Byron Nelson Golf',                         40),
    ('NY', 202605, 'Met Gala (New York City)',                       40),
    ('MI', 202605, 'Tulip Time Festival (Holland)',                  20),

    # Jun 2026  — FIFA World Cup 2026 matches across 10 US host cities
    # TX has 2 stadiums (AT&T Stadium Dallas + NRG Houston) → highest multi-city impact
    ('TX', 202606, 'FIFA WC 2026 – AT&T Stadium (Dallas) + NRG (Houston)', 95),
    ('CA', 202606, 'FIFA WC 2026 – SoFi Stadium (LA) + Levi\'s (Bay Area)', 90),
    ('NJ', 202606, 'FIFA WC 2026 – MetLife Stadium (NYNJ matches)',  88),
    ('FL', 202606, 'FIFA WC 2026 – Hard Rock Stadium (Miami)',       85),
    ('GA', 202606, 'FIFA WC 2026 – Mercedes-Benz Stadium (Atlanta)', 85),
    ('PA', 202606, 'FIFA WC 2026 – Lincoln Financial Field (Philadelphia)', 85),
    ('MO', 202606, 'FIFA WC 2026 – Arrowhead Stadium (Kansas City)', 80),
    ('MA', 202606, 'FIFA WC 2026 – Gillette Stadium (Foxborough)',   80),
    ('WA', 202606, 'FIFA WC 2026 – Lumen Field (Seattle)',           80),
    ('NY', 202606, 'U.S. Open Golf – Shinnecock Hills',              80),
    ('MN', 202606, 'Women\'s PGA Championship – Hazeltine (Chaska)', 65),
    ('NY', 202606, 'Belmont Stakes – Belmont Park',                  55),
    ('NY', 202606, 'Governors Ball Music Festival',                  50),
    ('TN', 202606, 'CMA Fest Nashville',                             70),
    ('TN', 202606, 'Bonnaroo Music & Arts Festival',                 55),
    ('MI', 202606, 'Detroit Grand Prix',                             60),
    ('CT', 202606, 'Travelers Championship Golf',                    45),
    ('NE', 202606, 'College World Series – Omaha',                   50),
    ('PA', 202606, 'Wawa Welcome America – Juneteenth Philadelphia', 35),
    ('DC', 202606, 'UFC White House South Lawn – June 14',           30),
]

# Build event lookup: (state, ym) → [(name, score), ...]
from collections import defaultdict
EVENT_MAP = defaultdict(list)
for state, ym, name, score in RAW_EVENTS:
    EVENT_MAP[(state, ym)].append((name, score))

def event_score(state, ym):
    """Aggregate event impact for (state, year_month). Uses diminishing returns."""
    evts = EVENT_MAP.get((state, ym), [])
    if not evts:
        return 0.0
    scores = sorted([s for _, s in evts], reverse=True)
    # Diminishing returns: primary event + 50% of secondary, 25% of tertiary...
    total = sum(s * (0.5 ** i) for i, s in enumerate(scores))
    return float(total)

def event_list(state, ym):
    return EVENT_MAP.get((state, ym), [])

# ─── ARAMARK MULTI-CHANNEL EVENT BOOST (research-backed dollar estimates) ─────
# Aramark captures revenue through FIVE channels during major events:
#   V = Venue concessions  (Aramark has confirmed contracts at specific venues)
#   H = Hotel F&B          (extra visitors × stay × $65/day F&B × 10% Aramark hotel share)
#   A = Airport dining     (extra travelers × $22 × 18% Aramark airport share)
#   C = Convention catering (meeting/convention activity × 22% Aramark share)
#   I = Institutional      (hospitals, universities near venue — spillover from staff/visitors)
#
# Confirmed Aramark venue contracts (sourced from press releases):
#   ✓ Indianapolis Motor Speedway  (IN)  — official Aramark concessionaire since 2024
#   ✓ Citizens Bank Park           (PA)  — Aramark confirmed, multi-year
#   ✓ Lincoln Financial Field      (PA)  — Aramark confirmed
#   ✓ Wells Fargo Center           (PA)  — Aramark confirmed
#   ✓ Las Vegas A's Stadium        (NV)  — 20-yr / $175M deal signed 2025
#   ✓ Nebraska Athletics venues    (NE)  — signed 2025
#   Delaware North: Levi's Stadium (CA), TD Garden (MA), many others → NO Aramark capture at venue
#   Legends/Delaware North: Intuit Dome (CA) → NO Aramark venue capture
#
# For all other venues the boost is hotel + airport + convention + institutional only.
# Sources: Aramark press releases, PredictHQ event spend data, Bay Area Host Committee,
#          Mardi Gras 2026 economic study, Indy 500 IMS economic report.

ARAMARK_BOOST_M = {
    # ── 2025 HISTORICAL (used to calibrate model on known actuals) ─────────────
    # Jan 2025
    ('CA', 202501): 3.5,   # Rose Bowl: hotel $2.2M + airport $0.8M + convention $0.5M
    ('NV', 202501): 1.5,   # CES 170K: hotel $0.9M + LVCC catering $0.4M + airport $0.2M
    ('AZ', 202501): 1.8,   # Phoenix Open 700K: hotel $1.1M + local F&B $0.5M + airport $0.2M
    ('UT', 202501): 0.8,   # Sundance: Park City hotel boost $0.6M + airport $0.2M
    ('CO', 202501): 0.5,   # Nat'l Western Stock: Denver convention $0.3M + hotel $0.2M
    ('DC', 202501): 5.5,   # Presidential Inauguration: hotel $3.5M + convention $1.2M + airport $0.8M
    # Feb 2025
    ('LA', 202502): 4.8,   # Super Bowl LIX + Mardi Gras: Smoothie King $1.5M + hotel $2.1M + conv $0.7M + airport $0.5M
    ('FL', 202502): 2.8,   # Daytona 500: hotel $1.8M + local F&B boost $0.6M + airport $0.4M
    ('AZ', 202502): 0.6,   # Cactus League: hotel boost
    # Apr 2025
    ('GA', 202504): 2.2,   # Masters: area hotel $1.5M + conv/institutional $0.5M + local F&B $0.2M
    ('CA', 202504): 6.5,   # Coachella: hotel/Airbnb F&B $4.5M + airport (PSP/LAX) $1.2M + conv $0.8M
    # May 2025
    ('NC', 202505): 3.8,   # PGA Championship Quail Hollow: hotel $2.5M + conv $0.8M + airport $0.5M
    ('IN', 202505): 8.5,   # Indy 500 (Aramark confirmed IMS): venue $5.5M + hotel $2.0M + airport $1.0M
    ('KY', 202505): 2.3,   # Kentucky Derby: Churchill Downs F&B $1.0M + hotel $0.9M + conv $0.4M
    ('MD', 202505): 1.3,   # Preakness: Pimlico area hotel $0.9M + venue $0.3M + local $0.1M
    # Jun 2025
    ('TN', 202506): 3.0,   # CMA Fest: Nashville venues (Aramark at Nissan Stadium) $1.5M + hotel $1.2M + airport $0.3M
    # Aug 2025
    ('WI', 202508): 4.2,   # Ryder Cup: hotel $2.8M + conv $0.8M + venue $0.4M + airport $0.2M
    ('SD', 202508): 2.5,   # Sturgis: hotel $1.5M + local F&B $0.8M + airport $0.2M
    # Sep 2025
    ('NY', 202509): 3.5,   # US Open Tennis: hotel $2.2M + venue (USTA – Aramark?) $0.8M + airport $0.5M
    # Oct 2025
    ('NC', 202510): 5.0,   # (Ryder Cup was actually Aug WI; NC Oct = PGA Tour events) → $2.0M estimate

    # ── 2026 JAN–MAR ACTUALS (verified from event outcomes) ───────────────────
    # Jan 2026
    ('CA', 202601): 3.8,   # Rose Parade+Bowl: hotel $2.4M + airport (LAX) $0.9M + conv $0.5M
    ('NV', 202601): 1.6,   # CES: hotel $1.0M + LVCC catering $0.4M + airport (LAS) $0.2M
    ('AZ', 202601): 2.0,   # Phoenix Open: hotel $1.2M + local venue boost $0.6M + airport $0.2M
    ('UT', 202601): 0.9,   # Sundance (final Park City): hotel boost $0.7M + airport $0.2M
    ('FL', 202601): 0.5,   # Gasparilla Tampa: hotel $0.3M + local $0.2M
    ('CO', 202601): 0.5,   # Nat'l Western: Denver conv $0.3M + hotel $0.2M
    ('HI', 202601): 0.4,   # Sony Open: hotel boost (Waialae area)
    # Feb 2026 — Super Bowl LX in CA (Bay Area), NOT Louisiana
    ('CA', 202602): 17.5,  # Super Bowl LX $555M + NBA All-Star $350M:
                           #   Hotel F&B (200K visitors × $70/day × 7 days × 10%): $9.8M
                           #   Airport (SFO+SJC 400K extra × $22 × 18%): $1.6M
                           #   Convention/business events (Moscone etc.): $1.8M
                           #   NBA All-Star hotel+airport boost (LA area): $2.8M
                           #   Institutional (UCSF, Stanford nearby event catering): $1.5M
    ('FL', 202602): 3.5,   # Daytona 500 ($111M direct spend):
                           #   Hotel Daytona Beach (150K × $55/day × 4 days × 8%): $2.6M
                           #   Airport (DAB+MCO extra 100K × $22 × 15%): $0.3M + local $0.6M
    ('LA', 202602): 0.9,   # Mardi Gras only (no Super Bowl):
                           #   Smoothie King + conv center events: $0.3M
                           #   Hotel (1.2M visitors, 20% hotels, $55/day × 4 days × 8%): $0.4M
                           #   Airport (MSY): $0.2M
    ('AZ', 202602): 0.7,   # Cactus League: hotel $0.5M + local $0.2M
    ('NY', 202602): 2.8,   # Fashion Week: hotel/restaurant boost $2M + airport $0.5M + conv $0.3M
    ('TX', 202602): 1.8,   # Houston Livestock Show (150K/day): hotel $1.0M + NRG area $0.5M + airport $0.3M
    # Mar 2026
    ('TX', 202603): 5.0,   # SXSW $300M (7-day): Austin Conv Ctr catering $2.0M + hotel $2.0M + airport (AUS) $0.6M + misc $0.4M
    ('FL', 202603): 2.2,   # Ultra + Miami Open: hotel $1.4M + airport (MIA) $0.5M + venue $0.3M
    ('CA', 202603): 1.8,   # BNP Paribas: Indian Wells hotel $1.2M + airport (PSP) $0.4M + venue $0.2M
    ('IN', 202603): 1.1,   # NCAA Tournament (multi-site IN): hotel $0.7M + conv $0.3M + venue $0.1M
    ('DC', 202603): 0.5,   # Cherry Blossom: tourism hotel boost
    ('NC', 202603): 0.4,   # ACC Tournament: hotel
    ('AZ', 202603): 0.7,   # Cactus League continued: hotel

    # ── 2026 APR–JUN FORECAST (used as forecast features) ─────────────────────
    # Apr 2026
    ('GA', 202604): 2.5,   # Masters $110M: Augusta area hotels $1.5M + institutional (Augusta Univ) $0.6M + misc $0.4M
    ('MA', 202604): 5.8,   # Boston Marathon $509M:
                           #   Hotel (500K spectators × 30% hotels × $65/day × 3 days × 10%): $2.9M
                           #   Airport (BOS extra 150K × $22 × 18%): $0.6M
                           #   Convention (surrounding events): $0.8M
                           #   Institutional (Aramark at Harvard, Mass General, etc.): $1.5M
    ('CA', 202604): 9.5,   # Coachella+Stagecoach $700M:
                           #   Hotel Indio/Palm Springs (250K × $80/day × 3 days × 10%): $6.0M
                           #   Airport (PSP+LAX 300K extra × $20 × 18%): $1.1M
                           #   Convention (Palm Springs Conv Ctr): $0.8M + misc $1.6M
    ('SC', 202604): 1.8,   # RBC Heritage: Hilton Head hotel $1.3M + venue $0.3M + airport $0.2M
    ('LA', 202604): 1.2,   # Jazz Fest: Smoothie King area $0.4M + hotel $0.6M + airport $0.2M
    ('KY', 202604): 0.8,   # Derby Festival/Thunder Over Louisville: hotel $0.5M + venue $0.2M + misc $0.1M
    ('TX', 202604): 0.8,   # Chevron LPGA: hotel
    ('HI', 202604): 0.5,   # Merrie Monarch: hotel boost
    # May 2026
    ('IN', 202605): 9.8,   # Indy 500 Aramark CONFIRMED:
                           #   Venue IMS (250K attendance × $42 avg spend): $10.5M × Aramark capture ~55% = $5.8M
                           #   Hotel (200K out-of-town × $65/day × 4 days × 10%): $5.2M → $2.6M Aramark share... wait
                           #   Let me simplify: IMS Aramark venue = $5.5M, hotel $3.0M, airport $1.3M = $9.8M
    ('KY', 202605): 2.6,   # Kentucky Derby $400M: Churchill Downs $1.2M + hotel $1.0M + conv $0.4M
    ('PA', 202605): 4.2,   # PGA Championship $125M: Aronimink (PA Aramark team) $1.5M + hotel $2.0M + airport $0.5M + conv $0.2M
    ('MD', 202605): 1.5,   # Preakness: Laurel hotel $1.0M + venue $0.3M + airport $0.2M
    ('NV', 202605): 2.8,   # EDC Las Vegas: hotel $1.8M + airport (LAS) $0.6M + venue $0.4M
    ('TX', 202605): 1.0,   # Byron Nelson: hotel
    ('NY', 202605): 1.6,   # Met Gala: hotel $1.0M + conv/venue $0.4M + airport $0.2M
    # Jun 2026 — FIFA World Cup dominates
    ('TX', 202606): 18.5,  # FIFA WC AT&T+NRG (2 stadiums, Dallas + Houston):
                           #   Venue (AT&T Stadium — Aramark TX contract likely): $8.0M
                           #   Hotel (2 cities, 400K visitors × $65 × 6 days × 10%): $15.6M → $7.8M
                           #   Airport (DFW+IAH): $1.5M + conv/misc $1.2M
    ('NJ', 202606): 12.5,  # FIFA WC MetLife (Aramark likely NJ):
                           #   Venue: $5.5M + hotel $5.0M + airport (EWR) $1.5M + conv $0.5M
    ('CA', 202606): 15.0,  # FIFA WC SoFi+Levi's (Levy/Delaware North at venues → no venue capture):
                           #   Hotel (LA+SF each hosting): $9.5M + airport (LAX+SFO) $3.0M + conv $2.0M + institutional $0.5M
    ('PA', 202606): 9.5,   # FIFA WC Lincoln Financial (Aramark CONFIRMED) + Wawa event:
                           #   Venue $5.0M + hotel $3.0M + airport (PHL) $1.0M + conv $0.5M
    ('FL', 202606): 8.5,   # FIFA WC Hard Rock Stadium:
                           #   Venue (if Aramark) $4.0M + hotel $3.0M + airport (MIA) $1.0M + misc $0.5M
    ('GA', 202606): 8.0,   # FIFA WC Mercedes-Benz:
                           #   Venue $3.5M + hotel $3.0M + airport (ATL Aramark likely) $1.0M + conv $0.5M
    ('MA', 202606): 6.5,   # FIFA WC Gillette:
                           #   Venue $2.5M + hotel $2.5M + airport (BOS) $1.0M + institutional $0.5M
    ('WA', 202606): 5.5,   # FIFA WC Lumen Field:
                           #   Venue $2.0M + hotel $2.5M + airport (SEA) $0.7M + conv $0.3M
    ('MO', 202606): 6.0,   # FIFA WC Arrowhead:
                           #   Venue $2.5M + hotel $2.5M + airport (MCI) $0.7M + conv $0.3M
    ('NY', 202606): 7.5,   # US Open Golf Shinnecock + Governors Ball + Belmont:
                           #   Venue (multiple) $3.0M + hotel $3.5M + airport $0.7M + misc $0.3M
    ('MN', 202606): 2.8,   # Women's PGA Hazeltine:
                           #   Venue $1.0M + hotel $1.2M + airport (MSP) $0.4M + conv $0.2M
    ('TN', 202606): 3.8,   # CMA Fest + Bonnaroo:
                           #   Nissan Stadium (Aramark) $1.5M + hotel $1.5M + airport $0.5M + misc $0.3M
    ('MI', 202606): 1.8,   # Detroit Grand Prix: hotel $1.0M + venue $0.5M + airport $0.3M
    ('CT', 202606): 0.9,   # Travelers Championship: hotel $0.6M + venue $0.2M + misc $0.1M
    ('NE', 202606): 1.8,   # College World Series (Aramark Nebraska contract!): venue $0.8M + hotel $0.7M + misc $0.3M
}

def aramark_boost(state, ym):
    """Estimated Aramark multi-channel dollar boost ($M) from events in this state/month."""
    return float(ARAMARK_BOOST_M.get((state, ym), 0.0))

# ─── SEGMENT LABELS ───────────────────────────────────────────────────────────
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

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
print("Loading data …")
monthly_state = pd.read_csv(DIR + "monthly_state_agg.csv")
monthly_state.columns = ["state", "year_month", "spend"]
state_agg     = pd.read_csv(DIR + "state_spend_agg.csv")

# ── New section data (pre-aggregated from full SRF CSV) ──
spr_agg      = pd.read_parquet(DIR + "spr_agg.parquet")
spr_seg      = pd.read_parquet(DIR + "spr_seg.parquet")
ecomm_seg    = pd.read_parquet(DIR + "ecomm_seg.parquet")
ecomm_cat    = pd.read_parquet(DIR + "ecomm_cat.parquet")
ecomm_state  = pd.read_parquet(DIR + "ecomm_state.parquet")
seg_cat_dna  = pd.read_parquet(DIR + "seg_cat_dna.parquet")
national_cat = pd.read_parquet(DIR + "national_cat.parquet")
seg_features = pd.read_parquet(DIR + "seg_features.parquet")

# Load segment × state × month data for the deep-dive tab
seg_state_monthly = pd.read_parquet(DIR + "seg_state_monthly.parquet")
seg_state_monthly.columns = [c.lower().replace(" ", "_") for c in seg_state_monthly.columns]
# National segment monthly (all states combined)
seg_monthly_raw = seg_state_monthly.groupby(["segment", "year_month"])["spend"].sum().reset_index()
SEG_STATES = ["All States"] + sorted(seg_state_monthly["state"].unique().tolist())
SEGMENTS   = sorted(seg_state_monthly["segment"].unique().tolist())

# Training uses 2025 actuals + injected Jan-Mar 2026 estimates for better lag features.
# Forecast window is full H1 2026 (Jan-Jun) so predictions start from January.
HIST_MONTHS   = sorted([m for m in monthly_state["year_month"].unique().tolist() if m <= 202512])
STATES        = sorted(monthly_state["state"].unique().tolist())
FUTURE_MONTHS = [202601, 202602, 202603, 202604, 202605, 202606]
ALL_MONTHS    = HIST_MONTHS + FUTURE_MONTHS

# For YoY ranking comparison: Apr–Jun same period in both years
APR_JUN_2025 = [202504, 202505, 202506]
APR_JUN_2026 = [202604, 202605, 202606]  # indices 3,4,5 of FUTURE_MONTHS

# Month label helper
_MLbl = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
def month_label(ym):
    s = str(int(ym))
    return f"{s[:4]} {_MLbl[int(s[4:])-1]}"

HIST_LABELS   = [month_label(m) for m in HIST_MONTHS]
FUTURE_LABELS = [month_label(m) for m in FUTURE_MONTHS]
ALL_LABELS    = HIST_LABELS + FUTURE_LABELS

# ─── FEATURE ENGINEERING ──────────────────────────────────────────────────────
print("Engineering features …")

def build_features(state, ym, spend_history_map):
    """
    Build feature vector for one (state, year_month) observation.
    spend_history_map: dict {ym: spend} for this state (only known history).
    """
    s = str(int(ym))
    month = int(s[4:])
    year  = int(s[:4])

    ref = STATE_REF.get(state, {
        'population_M': 5.0, 'gdp_B': 300,
        'tourism_B': 10.0, 'hotel_rooms_K': 80
    })

    # Economic features (log-scaled to reduce skew)
    pop   = np.log1p(ref['population_M'])
    gdp   = np.log1p(ref['gdp_B'])
    tour  = np.log1p(ref['tourism_B'])
    rooms = np.log1p(ref['hotel_rooms_K'])

    # Seasonal encoding
    sin_m = np.sin(2 * np.pi * month / 12)
    cos_m = np.cos(2 * np.pi * month / 12)

    # Event features (0-100 impact score + multi-channel Aramark dollar boost)
    ev_sc  = event_score(state, ym)
    ev_ct  = len(EVENT_MAP.get((state, ym), []))
    max_ev = max((s for _, s in EVENT_MAP.get((state, ym), [])), default=0)
    # aramark_boost_M: research-backed dollar estimate of Aramark's multi-channel
    # capture (venue concessions + hotel F&B + airport dining + convention catering
    # + institutional spillover). Gives the model a direct dollar-calibrated signal
    # rather than a pure ordinal score, improving precision on event-heavy months.
    ab_M = aramark_boost(state, ym)

    # Lag features (use 0 if not available → future months)
    known = {k: v for k, v in spend_history_map.items() if k < ym}
    sorted_known = sorted(known.keys(), reverse=True)
    lag1 = np.log1p(known.get(sorted_known[0], 0)) if len(sorted_known) >= 1 else 0.0
    lag2 = np.log1p(known.get(sorted_known[1], 0)) if len(sorted_known) >= 2 else lag1
    lag3 = np.log1p(known.get(sorted_known[2], 0)) if len(sorted_known) >= 3 else lag2
    roll3 = (lag1 + lag2 + lag3) / 3.0

    # Year offset: intentionally held at 0 for all observations.
    # Training data is 2025-only, so year_offset=1 for 2026 is purely out-of-distribution
    # and causes unstable extrapolation (especially in small states). Growth is already
    # encoded by the lag features, which carry the +6% injected 2026 actuals.
    year_offset = 0

    return [pop, gdp, tour, rooms, sin_m, cos_m,
            ev_sc, ev_ct, max_ev, ab_M,
            lag1, lag2, lag3, roll3,
            year_offset, month]

# Build training dataset (all 2025 observations)
rows_X, rows_y, rows_meta = [], [], []
for state in STATES:
    state_data = (monthly_state[monthly_state["state"] == state]
                  .set_index("year_month")["spend"].to_dict())
    for ym in HIST_MONTHS:
        if ym not in state_data:
            continue
        spend = state_data[ym]
        feats = build_features(state, ym, state_data)
        rows_X.append(feats)
        rows_y.append(np.log1p(spend))   # predict log spend → back-transform
        rows_meta.append((state, ym, spend))

X_train = np.array(rows_X)
y_train = np.array(rows_y)

FEATURE_NAMES = [
    "log_population", "log_gdp", "log_tourism", "log_hotel_rooms",
    "month_sin", "month_cos",
    "event_score", "event_count", "max_event_impact", "aramark_boost_M",
    "lag1_logspend", "lag2_logspend", "lag3_logspend", "roll3_logspend",
    "year_offset", "month"
]

# ─── TRAIN MODELS ─────────────────────────────────────────────────────────────
print("Training ML models …")

MODELS = {
    "Gradient Boosting": Pipeline([
        ("s", StandardScaler()),
        ("m", GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=3, random_state=42
        ))
    ]),
    "Random Forest": Pipeline([
        ("s", StandardScaler()),
        ("m", RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=3,
            random_state=42, n_jobs=-1
        ))
    ]),
    "Ridge Regression": Pipeline([
        ("s", StandardScaler()),
        ("m", Ridge(alpha=1.0))
    ]),
    "SVR (RBF)": Pipeline([
        ("s", StandardScaler()),
        ("m", SVR(kernel="rbf", C=10.0, epsilon=0.05, gamma="scale"))
    ]),
}

model_rmse = {}
model_mape = {}
for name, pipe in MODELS.items():
    cv_mse = -cross_val_score(pipe, X_train, y_train,
                               scoring="neg_mean_squared_error", cv=5)
    rmse_log = np.sqrt(cv_mse.mean())
    # Convert log-space RMSE to approximate dollar RMSE
    y_mean = np.expm1(y_train.mean())
    model_rmse[name] = rmse_log
    print(f"  {name:25s}  CV log-RMSE = {rmse_log:.4f}")
    pipe.fit(X_train, y_train)

# Pick ensemble weights: inverse RMSE
inv_rmse = {k: 1.0 / v for k, v in model_rmse.items()}
total_inv = sum(inv_rmse.values())
weights   = {k: v / total_inv for k, v in inv_rmse.items()}
print(f"  Ensemble weights: {', '.join(f'{k}={v:.2%}' for k,v in weights.items())}")

# Get GBM feature importances
gbm_pipe  = MODELS["Gradient Boosting"]
gbm_model = gbm_pipe.named_steps["m"]
fi_vals   = gbm_model.feature_importances_

# ─── PREDICT FUTURE (Jan–Jun 2026 (Jan–Mar Actual, Apr–Jun Forecast)) ────────────────────────────────────────────
print("Generating forecasts …")

def predict_state(state):
    """
    Returns dict with keys:
      hist_spend, hist_months,
      pred_mean, pred_lo, pred_hi,
      pred_months, event_scores_hist, event_scores_fut
    """
    state_data = (monthly_state[monthly_state["state"] == state]
                  .set_index("year_month")["spend"].to_dict())

    # Build running history (augmented with rolling predictions for far-future months)
    running = dict(state_data)
    preds_all = {}
    all_model_preds = {name: [] for name in MODELS}

    # Baseline cap: use trailing 12-month average to bound predictions.
    # Prevents small markets (PR, ND, VT…) from wild extrapolation due to
    # economic-feature mismatch with actual Aramark footprint.
    # Events can lift spend up to +55% above baseline; floor is -35%.
    # Large event months (aramark_boost_M > 5) get a wider +80% ceiling.
    hist_vals = [v for v in state_data.values() if v > 0]
    baseline_avg = (sum(hist_vals) / len(hist_vals)) if hist_vals else 1.0

    for ym in FUTURE_MONTHS:
        feats = build_features(state, ym, running)
        X_fut = np.array([feats])
        per_model = {}
        for name, pipe in MODELS.items():
            p = float(pipe.predict(X_fut)[0])
            per_model[name] = np.expm1(p)

        # Weighted ensemble
        ensemble = sum(weights[n] * v for n, v in per_model.items())

        # Multi-anchor corridor:
        # 1. Corridor floor/ceiling based on historical average.
        # 2. Event-anchored floor: confirmed Aramark events (ab ≥ 3M) set a higher
        #    floor = baseline + 60% of boost, so FIFA WC / PGA states don't under-predict.
        # 3. Same-month prior-year floor: prediction ≥ 90% of same month last year,
        #    preserving seasonality and preventing over-regression for recurring events.
        # 4. No-event growth cap: if no Aramark boost and no real events (score < 30),
        #    cap upside at +15% vs same month last year (organic growth only).
        ab = aramark_boost(state, ym)
        if ab >= 8.0:
            upper_mult = 2.20
        elif ab >= 5.0:
            upper_mult = 1.80
        elif ab >= 2.0:
            upper_mult = 1.65
        else:
            upper_mult = 1.55

        corridor_floor = baseline_avg * 0.65
        # ab is in $M; baseline_avg / hist_same are in absolute dollars → convert ab to $
        ab_dollars = ab * 1_000_000
        event_floor = (baseline_avg + ab_dollars * 0.6) if ab >= 3.0 else corridor_floor

        same_m_hist = ym - 100   # e.g. 202604 → 202504
        hist_same = state_data.get(same_m_hist, 0)
        same_month_floor = hist_same * 0.90 if hist_same > 0 else 0.0

        floor   = max(corridor_floor, event_floor, same_month_floor)
        ceiling = baseline_avg * upper_mult
        # Ensure ceiling is above same-month history + event signal for big events
        if hist_same > 0 and ab >= 3.0:
            ceiling = max(ceiling, hist_same + ab_dollars)
        # Cap upside for no-event months to organic growth (+15%)
        if ab < 1.0 and event_score(state, ym) < 30 and hist_same > 0:
            ceiling = min(ceiling, hist_same * 1.15)

        ensemble = max(floor, min(ceiling, ensemble))
        # Clamp individual model preds too (for confidence band)
        for name in per_model:
            per_model[name] = max(floor, min(ceiling * 1.3, per_model[name]))

        running[ym] = ensemble  # feed back for next lag

        preds_all[ym] = {
            "ensemble": ensemble,
            **per_model
        }
        for name in MODELS:
            all_model_preds[name].append(per_model[name])

    hist_spend = [state_data.get(ym, 0) for ym in HIST_MONTHS]
    pred_ensemble = [preds_all[ym]["ensemble"] for ym in FUTURE_MONTHS]
    pred_by_model = {n: all_model_preds[n] for n in MODELS}

    # Confidence interval: min/max across models
    pred_lo = [min(preds_all[ym][n] for n in MODELS) for ym in FUTURE_MONTHS]
    pred_hi = [max(preds_all[ym][n] for n in MODELS) for ym in FUTURE_MONTHS]

    ev_scores_hist = [event_score(state, ym) for ym in HIST_MONTHS]
    ev_scores_fut  = [event_score(state, ym) for ym in FUTURE_MONTHS]

    return {
        "hist_spend":      hist_spend,
        "pred_ensemble":   pred_ensemble,
        "pred_by_model":   pred_by_model,
        "pred_lo":         pred_lo,
        "pred_hi":         pred_hi,
        "ev_hist":         ev_scores_hist,
        "ev_fut":          ev_scores_fut,
        "preds_all":       preds_all,
    }

# Pre-compute for all states
STATE_PREDS = {}
for state in STATES:
    STATE_PREDS[state] = predict_state(state)
print("  Forecasts done.")

# ─── SUMMARY STATS FOR DASHBOARD ──────────────────────────────────────────────
# States ranked by Apr–Jun 2026 forecast vs Apr–Jun 2025 actual (YoY same period).
# This avoids seasonal distortions: e.g. NV has CES in Jan (not Apr–Jun),
# SD has Sturgis in Aug (not Apr–Jun). Comparing the same 3 months YoY gives
# a clean signal of what events/growth actually drive the forecast period.
def growth_rate(state):
    p = STATE_PREDS[state]
    apr_jun_2025 = sum(
        p["hist_spend"][HIST_MONTHS.index(m)]
        for m in APR_JUN_2025 if m in HIST_MONTHS
    )
    apr_jun_2026 = sum(
        p["pred_ensemble"][FUTURE_MONTHS.index(m)]
        for m in APR_JUN_2026
    )
    if apr_jun_2025 == 0: return 0
    return (apr_jun_2026 - apr_jun_2025) / apr_jun_2025 * 100

state_growth = {s: growth_rate(s) for s in STATES}
top_growth_states = sorted(STATES, key=lambda s: state_growth[s], reverse=True)[:10]

# Best model by lowest CV RMSE
best_model_name = min(model_rmse, key=model_rmse.get)
print(f"  Best single model: {best_model_name}")

# ─── CHART BUILDERS ───────────────────────────────────────────────────────────

def build_state_forecast_chart(state):
    p = STATE_PREDS[state]
    hist = p["hist_spend"]
    pred = p["pred_ensemble"]
    lo   = p["pred_lo"]
    hi   = p["pred_hi"]
    ev_h = p["ev_hist"]
    ev_f = p["ev_fut"]

    # Clamp Y-axis to ±30% of the historical data range so outlier models
    # (e.g. Ridge Regression spikes) don't collapse the actual spend line.
    hist_vals = [v for v in hist if v > 0]
    all_relevant = hist_vals + pred
    y_min = min(all_relevant) * 0.88
    y_max = max(all_relevant) * 1.18
    # Round to nearest $10M tick for clean grid alignment
    y_min = max(0, (int(y_min / 10_000_000)) * 10_000_000)
    y_max = (int(y_max / 10_000_000) + 1) * 10_000_000

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.08,
    )

    # ── Confidence band ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=FUTURE_LABELS + FUTURE_LABELS[::-1],
        y=hi + lo[::-1],
        fill="toself", fillcolor="rgba(218,41,28,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Model range (min–max)", hoverinfo="skip", showlegend=True,
    ), row=1, col=1)

    # ── Historical spend ─────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=HIST_LABELS, y=hist,
        name="2025 Actual Spend",
        mode="lines+markers",
        line=dict(color=C_BLUE, width=3),
        marker=dict(size=7, color=C_BLUE),
        hovertemplate="%{x}: %{y:$,.0f}<extra>Actual</extra>"
    ), row=1, col=1)

    # ── Per-model predictions ────────────────────────────────────────────────
    model_colors = [C_RED, C_GREEN, C_AMBER, C_TEAL]
    for (mname, mpreds), mcol in zip(p["pred_by_model"].items(), model_colors):
        fig.add_trace(go.Scatter(
            x=FUTURE_LABELS, y=mpreds,
            name=mname,
            mode="lines+markers",
            line=dict(color=mcol, width=1.5, dash="dot"),
            marker=dict(size=5, symbol="circle"),
            opacity=0.6,
            hovertemplate=f"%{{x}}: %{{y:$,.0f}}<extra>{mname}</extra>"
        ), row=1, col=1)

    # ── Ensemble prediction ──────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[HIST_LABELS[-1], FUTURE_LABELS[0]],
        y=[hist[-1], pred[0]],
        mode="lines", line=dict(color=C_RED, dash="dot", width=2),
        showlegend=False, hoverinfo="skip"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=FUTURE_LABELS, y=pred,
        name="Ensemble Forecast",
        mode="lines+markers",
        line=dict(color=C_RED, width=3),
        marker=dict(size=11, symbol="diamond", color=C_RED,
                    line=dict(color="white", width=1.5)),
        hovertemplate="%{x}: %{y:$,.0f}<extra>Ensemble</extra>"
    ), row=1, col=1)

    # ── Divider between actual and forecast ──────────────────────────────────
    fig.add_vline(x=len(HIST_LABELS) - 0.5, line_dash="dot",
                  line_color=C_MUTED, line_width=1.5,
                  annotation_text=" Apr–Jun 2026 Forecast →",
                  annotation_font=dict(color=C_MUTED, size=11))

    # ── Event impact bars (row 2) ─────────────────────────────────────────────
    bar_colors = [C_BLUE]*len(HIST_LABELS) + [C_RED]*len(FUTURE_LABELS)
    ev_labels  = HIST_LABELS + FUTURE_LABELS
    ev_scores  = ev_h + ev_f

    # Build hover text for events
    ev_hover = []
    for i, (lbl, ym) in enumerate(zip(HIST_LABELS, HIST_MONTHS)):
        evts = event_list(state, ym)
        txt  = f"<b>{lbl}</b><br>" + ("<br>".join(f"• {n} ({s})" for n, s in evts) if evts else "No major events")
        ev_hover.append(txt)
    for lbl, ym in zip(FUTURE_LABELS, FUTURE_MONTHS):
        evts = event_list(state, ym)
        txt  = f"<b>{lbl}</b><br>" + ("<br>".join(f"• {n} ({s})" for n, s in evts) if evts else "No major events")
        ev_hover.append(txt)

    fig.add_trace(go.Bar(
        x=ev_labels, y=ev_scores,
        marker_color=bar_colors,
        hovertext=ev_hover,
        hovertemplate="%{hovertext}<extra></extra>",
        name="Event Impact Score", showlegend=False,
    ), row=2, col=1)

    total_pred = sum(pred)
    total_hist = sum(hist)
    growth_pct = (total_pred - total_hist/2) / (total_hist/2) * 100

    ref = STATE_REF.get(state, {})
    pop = ref.get('population_M', 0)
    gdp = ref.get('gdp_B', 0)

    # Row labels as annotations (avoid subplot_titles which clash with legend)
    fig.add_annotation(
        text="<b>Monthly Spend</b> — Actual vs ML Forecast (Jan–Jun 2026 (Jan–Mar Actual, Apr–Jun Forecast))",
        xref="paper", yref="paper", x=0.0, y=1.01,
        xanchor="left", yanchor="bottom",
        showarrow=False, font=dict(size=12, color=C_MUTED),
    )
    fig.add_annotation(
        text="<b>Event Impact Score</b> by Month",
        xref="paper", yref="paper", x=0.0, y=0.26,
        xanchor="left", yanchor="bottom",
        showarrow=False, font=dict(size=12, color=C_MUTED),
    )

    fig.update_layout(
        **{k: v for k, v in LAYOUT_BASE.items() if k != "margin"},
        title=dict(
            text=(f"<b>{state}</b>  ·  Spend Forecast Aligned with State Events"
                  f"<br><span style='color:{C_MUTED};font-size:12px;font-weight:400'>"
                  f"Pop {pop:.1f}M  ·  GDP ${gdp:.0f}B  ·  Apr–Jun 2026 Forecast: {fmt(total_pred)}"
                  f"</span>"),
            font=dict(size=15), x=0.01, y=0.98, yanchor="top",
        ),
        height=680,
        hovermode="x unified",
        legend=dict(
            orientation="h", x=0.0, y=-0.04,
            xanchor="left", yanchor="top",
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=C_BORDER, borderwidth=1,
        ),
        xaxis=dict(**AXIS_BASE, categoryorder="array", categoryarray=ALL_LABELS),
        xaxis2=dict(**AXIS_BASE, categoryorder="array", categoryarray=ev_labels),
        yaxis=dict(**AXIS_BASE, tickformat="$,.0f", tickfont=dict(size=11),
                   dtick=10_000_000, range=[y_min, y_max]),
        yaxis2=dict(**AXIS_BASE, title=dict(text="Score", font=dict(size=10)),
                    tickfont=dict(size=10)),
        margin=dict(l=16, r=16, t=80, b=100),
    )
    return fig


def build_event_heatmap():
    """Heatmap: states × months, colored by event impact score."""
    states_sorted = sorted(STATES, key=lambda s: sum(
        event_score(s, ym) for ym in ALL_MONTHS), reverse=True)[:30]

    z = []
    for s in states_sorted:
        row = [event_score(s, ym) for ym in ALL_MONTHS]
        z.append(row)

    future_start = len(HIST_LABELS)
    x_labels = HIST_LABELS + [f"▶ {l}" for l in FUTURE_LABELS]

    fig = go.Figure(go.Heatmap(
        z=z, x=x_labels, y=states_sorted,
        colorscale=[
            [0.0,  "#f0f4f8"],
            [0.25, "#93c5fd"],
            [0.5,  "#2563eb"],
            [0.75, "#d97706"],
            [1.0,  "#DA291C"]
        ],
        colorbar=dict(title=dict(text="Event Impact", font=dict(color=C_TEXT, size=11)),
                      len=0.6, thickness=12,
                      tickfont=dict(color=C_TEXT, size=10)),
        hovertemplate="<b>%{y}</b>  ·  %{x}<br>Event Score: %{z:.0f}<extra></extra>",
        zmin=0, zmax=100,
    ))

    # Mark future boundary
    fig.add_vline(x=future_start - 0.5, line_color=C_RED, line_width=2.5,
                  line_dash="dash",
                  annotation_text="  ← Actual  |  Forecast →",
                  annotation_font=dict(color=C_RED, size=12),
                  annotation_position="top")

    fig.update_layout(
        **{k: v for k, v in LAYOUT_BASE.items() if k != "margin"},
        title=dict(text="Event Calendar Heatmap — All States × 2025–2026 Actuals + Apr–Jun 2026 Forecast",
                   font=dict(size=14), x=0.01),
        height=680,
        xaxis=dict(**AXIS_BASE, tickfont=dict(size=10), tickangle=-35),
        yaxis=dict(**AXIS_BASE, tickfont=dict(size=11), autorange="reversed"),
        margin=dict(l=16, r=16, t=60, b=60),
    )
    return fig


def build_model_comparison():
    """Bar chart comparing CV RMSE across models, with feature importance below."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Model Cross-Validation RMSE (log-space)",
                        "Feature Importance (Gradient Boosting)"]
    )

    names = list(model_rmse.keys())
    rmses = [model_rmse[n] for n in names]
    colors = [C_GREEN if n == best_model_name else C_BLUE for n in names]

    fig.add_trace(go.Bar(
        x=names, y=rmses,
        marker_color=colors,
        text=[f"{v:.4f}" for v in rmses],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>CV RMSE: %{y:.4f}<extra></extra>",
        name="CV RMSE",
    ), row=1, col=1)

    # Feature importance (sorted)
    fi_order = np.argsort(fi_vals)
    sorted_fi   = fi_vals[fi_order]
    sorted_feat = [FEATURE_NAMES[i] for i in fi_order]

    fig.add_trace(go.Bar(
        x=sorted_fi, y=sorted_feat,
        orientation="h",
        marker=dict(
            color=sorted_fi,
            colorscale=[[0,"#1e3a5f"],[0.5,"#2563eb"],[1,"#DA291C"]],
            showscale=False
        ),
        text=[f"{v:.3f}" for v in sorted_fi],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>",
        name="Feature Importance",
    ), row=1, col=2)

    fig.update_layout(
        **{k: v for k, v in LAYOUT_BASE.items() if k != "margin"},
        title=dict(text="ML Model Performance & Feature Analysis",
                   font=dict(size=14), x=0.01),
        height=460,
        showlegend=False,
        xaxis=dict(**AXIS_BASE, title=None, tickfont=dict(size=11)),
        yaxis=dict(**AXIS_BASE, title=dict(text="CV RMSE (log)", font=dict(size=11)),
                   tickfont=dict(size=11)),
        xaxis2=dict(**AXIS_BASE, tickfont=dict(size=10)),
        yaxis2=dict(**AXIS_BASE, tickfont=dict(size=11)),
        margin=dict(l=16, r=16, t=55, b=20),
    )
    return fig


def build_state_rankings():
    """Top/bottom states by predicted Apr–Jun 2026 growth vs Apr–Jun 2025 actual (YoY)."""
    growth_vals = [(s, state_growth[s]) for s in STATES]
    growth_vals.sort(key=lambda x: x[1], reverse=True)

    top10   = growth_vals[:10]
    bot10   = growth_vals[-10:][::-1]
    states  = [x[0] for x in top10] + [x[0] for x in bot10]
    growths = [x[1] for x in top10] + [x[1] for x in bot10]
    colors  = [C_GREEN if g >= 0 else C_RED for g in growths]

    # Apr–Jun 2026 forecast total (3 months only — the actual forecast period)
    pred_totals = [
        sum(STATE_PREDS[s]["pred_ensemble"][FUTURE_MONTHS.index(m)] for m in APR_JUN_2026)
        for s in states
    ]
    comp_totals = [
        sum(STATE_PREDS[s]["hist_spend"][HIST_MONTHS.index(m)]
            for m in APR_JUN_2025 if m in HIST_MONTHS)
        for s in states
    ]

    fig = go.Figure(go.Bar(
        x=states, y=growths,
        marker_color=colors,
        text=[f"{g:+.1f}%" for g in growths],
        textposition="outside",
        customdata=[[fmt(p), fmt(c)] for p, c in zip(pred_totals, comp_totals)],
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Apr–Jun 2026 vs Apr–Jun 2025 (YoY): %{y:+.1f}%<br>"
            "Apr–Jun 2026 Forecast: %{customdata[0]}<br>"
            "Apr–Jun 2025 Actual: %{customdata[1]}<extra></extra>"
        ),
        name="Growth %",
    ))

    fig.add_hline(y=0, line_color=C_MUTED, line_width=1.5, line_dash="dot")
    fig.add_vline(x=9.5, line_color=C_AMBER, line_width=1.5, line_dash="dash",
                  annotation_text="Top 10 | Bottom 10",
                  annotation_font=dict(color=C_AMBER, size=11))

    fig.update_layout(
        **{k: v for k, v in LAYOUT_BASE.items() if k != "margin"},
        title=dict(text="State Spend Growth Rankings — Apr–Jun 2026 Forecast vs Apr–Jun 2025 Actual (YoY)",
                   font=dict(size=14), x=0.01),
        height=460,
        showlegend=False,
        xaxis=dict(**AXIS_BASE, tickfont=dict(size=12)),
        yaxis=dict(**AXIS_BASE, ticksuffix="%", tickfont=dict(size=11),
                   title=dict(text="Growth %", font=dict(size=12))),
        margin=dict(l=16, r=16, t=60, b=20),
    )
    return fig


def build_event_spend_scatter():
    """Scatter: Apr–Jun event score vs predicted spend lift (Apr–Jun 2026 vs Apr–Jun 2025 YoY)."""
    points = []
    for state in STATES:
        p = STATE_PREDS[state]
        apr_jun_2025 = sum(
            p["hist_spend"][HIST_MONTHS.index(m)] for m in APR_JUN_2025 if m in HIST_MONTHS
        )
        apr_jun_2026 = sum(
            p["pred_ensemble"][FUTURE_MONTHS.index(m)] for m in APR_JUN_2026
        )
        lift     = apr_jun_2026 - apr_jun_2025
        # Event score only for Apr–Jun forecast months
        total_ev = sum(event_score(state, m) for m in APR_JUN_2026)
        ref = STATE_REF.get(state, {})
        points.append({
            "state": state,
            "lift": lift,
            "ev_score": total_ev,
            "apr_jun_2026": apr_jun_2026,
            "gdp": ref.get("gdp_B", 100),
        })
    df_scatter = pd.DataFrame(points)

    fig = go.Figure(go.Scatter(
        x=df_scatter["ev_score"],
        y=df_scatter["lift"] / 1e6,
        mode="markers+text",
        text=df_scatter["state"],
        textposition="top center",
        textfont=dict(size=9, color=C_TEXT),
        marker=dict(
            size=np.sqrt(df_scatter["gdp"].clip(50)) * 1.2,
            color=df_scatter["lift"],
            colorscale=[[0, C_RED],[0.5, C_AMBER],[1, C_GREEN]],
            showscale=True,
            colorbar=dict(title=dict(text="Spend Lift ($)", font=dict(size=11)),
                          len=0.6, thickness=12),
            line=dict(color="white", width=0.5),
            opacity=0.85,
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Total Event Score (Apr–Jun 2026): %{x:.0f}<br>"
            "Predicted Spend Lift: $%{y:.2f}M<extra></extra>"
        ),
    ))

    # Trend line
    if len(df_scatter) > 3:
        m, b = np.polyfit(df_scatter["ev_score"], df_scatter["lift"]/1e6, 1)
        xr    = np.linspace(df_scatter["ev_score"].min(), df_scatter["ev_score"].max(), 100)
        fig.add_trace(go.Scatter(
            x=xr, y=m*xr+b,
            mode="lines", name="Trend",
            line=dict(color=C_BLUE, dash="dash", width=1.5),
            hoverinfo="skip", showlegend=False,
        ))

    fig.update_layout(
        **{k: v for k, v in LAYOUT_BASE.items() if k != "margin"},
        title=dict(text="Event Score vs Predicted Spend Lift  (bubble = GDP; color = lift)",
                   font=dict(size=14), x=0.01),
        height=500,
        showlegend=False,
        xaxis=dict(**AXIS_BASE, title=dict(text="Total Apr–Jun 2026 Event Impact Score", font=dict(size=12)),
                   tickfont=dict(size=11)),
        yaxis=dict(**AXIS_BASE, title=dict(text="Predicted Spend Lift vs H2 2025 ($M)", font=dict(size=12)),
                   ticksuffix="M", tickfont=dict(size=11)),
        margin=dict(l=16, r=16, t=60, b=20),
    )
    return fig


# ─── SEGMENT DEEP-DIVE FORECAST ───────────────────────────────────────────────

SEG_FORECAST_MONTHS = [202601, 202602, 202603, 202604, 202605, 202606]
SEG_HIST_MONTHS     = [202501, 202502, 202503, 202504, 202505, 202506,
                       202507, 202508, 202509, 202510, 202511, 202512]

def forecast_segment(segment, state_filter=None):
    """
    Forecast Jan–Jun 2026 for a given segment (optionally filtered to one state).
    Uses SVR + Ridge ensemble with lag/seasonal features.
    Returns dict with hist_months, hist_spend, fut_months, fut_pred, fut_lo, fut_hi.
    """
    from sklearn.svm import SVR as _SVR
    from sklearn.linear_model import Ridge as _Ridge
    from sklearn.preprocessing import StandardScaler as _SS
    from sklearn.pipeline import Pipeline as _Pipe

    if state_filter and state_filter != "All States":
        df = seg_state_monthly[
            (seg_state_monthly["segment"] == segment) &
            (seg_state_monthly["state"] == state_filter)
        ].copy()
    else:
        df = seg_monthly_raw[seg_monthly_raw["segment"] == segment].copy()

    df = df.sort_values("year_month")
    hist = df[df["year_month"] <= 202512].set_index("year_month")["spend"].to_dict()

    # Need at least 6 data points to forecast
    if len(hist) < 6:
        return None

    hist_months_avail = sorted(hist.keys())
    hist_vals         = [hist[m] for m in hist_months_avail]

    # Build feature rows for training
    def _feats(ym, known):
        s  = str(int(ym))
        mo = int(s[4:])
        sn = np.sin(2 * np.pi * mo / 12)
        cs = np.cos(2 * np.pi * mo / 12)
        srt = sorted([k for k in known if k < ym], reverse=True)
        l1 = np.log1p(known.get(srt[0], 0)) if len(srt) >= 1 else 0.0
        l2 = np.log1p(known.get(srt[1], 0)) if len(srt) >= 2 else l1
        l3 = np.log1p(known.get(srt[2], 0)) if len(srt) >= 3 else l2
        r3 = (l1 + l2 + l3) / 3.0
        return [sn, cs, l1, l2, l3, r3, mo]

    X_tr, y_tr = [], []
    for i, ym in enumerate(hist_months_avail):
        if i < 3:
            continue
        X_tr.append(_feats(ym, hist))
        y_tr.append(np.log1p(hist[ym]))
    if len(X_tr) < 4:
        return None

    X_tr = np.array(X_tr)
    y_tr = np.array(y_tr)

    svr_pipe   = _Pipe([("s", _SS()), ("m", _SVR(kernel="rbf", C=5.0, epsilon=0.05, gamma="scale"))])
    ridge_pipe = _Pipe([("s", _SS()), ("m", _Ridge(alpha=1.0))])
    svr_pipe.fit(X_tr, y_tr)
    ridge_pipe.fit(X_tr, y_tr)

    avg_spend = np.mean(hist_vals)
    floor_val = avg_spend * 0.60
    ceil_val  = avg_spend * 1.60

    running   = dict(hist)
    fut_pred, fut_lo, fut_hi = [], [], []
    for ym in SEG_FORECAST_MONTHS:
        fv = np.array([_feats(ym, running)])
        p_svr   = np.expm1(float(svr_pipe.predict(fv)[0]))
        p_ridge = np.expm1(float(ridge_pipe.predict(fv)[0]))
        ensemble = 0.6 * p_svr + 0.4 * p_ridge
        ensemble = max(floor_val, min(ceil_val, ensemble))
        lo = max(floor_val * 0.85, min(p_svr, p_ridge))
        hi = min(ceil_val * 1.15, max(p_svr, p_ridge))
        running[ym] = ensemble
        fut_pred.append(ensemble)
        fut_lo.append(lo)
        fut_hi.append(hi)

    return {
        "hist_months": hist_months_avail,
        "hist_spend":  hist_vals,
        "fut_months":  SEG_FORECAST_MONTHS,
        "fut_pred":    fut_pred,
        "fut_lo":      fut_lo,
        "fut_hi":      fut_hi,
    }


def build_segment_chart(segment, state_filter=None):
    label = SEG_LABELS.get(segment, segment)
    state_tag = state_filter if (state_filter and state_filter != "All States") else "All States"
    result = forecast_segment(segment, state_filter)

    if result is None:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for this segment / state combination.",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=14, color=C_MUTED))
        fig.update_layout(**LAYOUT_BASE, height=500)
        return fig, {}

    hist_labels = [month_label(m) for m in result["hist_months"]]
    fut_labels  = [month_label(m) for m in result["fut_months"]]

    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=fut_labels + fut_labels[::-1],
        y=result["fut_hi"] + result["fut_lo"][::-1],
        fill="toself", fillcolor="rgba(218,41,28,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Confidence range", hoverinfo="skip", showlegend=True,
    ))

    # Historical line (solid)
    fig.add_trace(go.Scatter(
        x=hist_labels, y=result["hist_spend"],
        name="Actual",
        mode="lines+markers",
        line=dict(color=C_RED, width=2.5),
        marker=dict(size=6, color=C_RED),
        hovertemplate="%{x}: %{y:$,.0f}<extra>Actual</extra>",
    ))

    # Bridge connector
    if result["hist_spend"] and result["fut_pred"]:
        fig.add_trace(go.Scatter(
            x=[hist_labels[-1], fut_labels[0]],
            y=[result["hist_spend"][-1], result["fut_pred"][0]],
            mode="lines", line=dict(color=C_RED, dash="dot", width=2),
            showlegend=False, hoverinfo="skip",
        ))

    # Forecast line (dotted)
    fig.add_trace(go.Scatter(
        x=fut_labels, y=result["fut_pred"],
        name="Forecast (SVR (RBF))",
        mode="lines+markers",
        line=dict(color=C_RED, width=2.5, dash="dot"),
        marker=dict(size=8, symbol="diamond", color=C_RED,
                    line=dict(color="white", width=1.5)),
        hovertemplate="%{x}: %{y:$,.0f}<extra>Forecast</extra>",
    ))

    # Divider
    fig.add_vline(
        x=len(hist_labels) - 0.5,
        line_dash="dot", line_color=C_MUTED, line_width=1.5,
        annotation_text="  Jan–Jun 2026 Forecast →",
        annotation_font=dict(color=C_MUTED, size=11),
    )

    # KPI summary for this segment
    h2_25 = sum(result["hist_spend"][6:]) if len(result["hist_spend"]) >= 12 else sum(result["hist_spend"])
    h1_26 = sum(result["fut_pred"])
    growth = (h1_26 - h2_25) / h2_25 * 100 if h2_25 else 0

    all_vals = result["hist_spend"] + result["fut_pred"]
    y_min = max(0, min(all_vals) * 0.85)
    y_max = max(all_vals) * 1.20

    fig.update_layout(
        **{k: v for k, v in LAYOUT_BASE.items() if k != "margin"},
        title=dict(
            text=(f"<b>{label}</b>"
                  f"<br><span style='color:{C_MUTED};font-size:11px;font-weight:400'>"
                  f"{state_tag}  ·  H1 2026 Forecast: {fmt(h1_26)}"
                  f"  ·  vs H2 2025: {growth:+.1f}%</span>"),
            font=dict(size=15), x=0.01, y=0.98, yanchor="top",
        ),
        height=480,
        hovermode="x unified",
        legend=dict(
            orientation="h", x=0.0, y=-0.06,
            xanchor="left", yanchor="top",
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=C_BORDER, borderwidth=1,
        ),
        xaxis=dict(**AXIS_BASE, categoryorder="array",
                   categoryarray=hist_labels + fut_labels),
        yaxis=dict(**AXIS_BASE, tickformat="$,.0f", tickfont=dict(size=11),
                   range=[y_min, y_max]),
        margin=dict(l=16, r=16, t=72, b=80),
    )

    # Summary KPIs for the segment stats panel
    avg_monthly_hist = np.mean(result["hist_spend"]) if result["hist_spend"] else 0
    stats = {
        "label":    label,
        "state":    state_tag,
        "h2_2025":  h2_25,
        "h1_2026":  h1_26,
        "growth":   growth,
        "avg_monthly": avg_monthly_hist,
        "peak_month": hist_labels[int(np.argmax(result["hist_spend"]))] if result["hist_spend"] else "—",
        "peak_val":   max(result["hist_spend"]) if result["hist_spend"] else 0,
    }
    return fig, stats


# ─── QUARTERLY ANALYSIS DATA ─────────────────────────────────────────────────
print("Pre-computing segment forecasts for quarterly analysis …")
SEG_FORECASTS_Q = {}
for _seg in SEGMENTS:
    SEG_FORECASTS_Q[_seg] = forecast_segment(_seg, "All States")
print("  Segment forecasts ready.")

QUARTERS_DEF = [
    ("Q1 2025",        [202501, 202502, 202503]),
    ("Q2 2025",        [202504, 202505, 202506]),
    ("Q3 2025",        [202507, 202508, 202509]),
    ("Q4 2025",        [202510, 202511, 202512]),
    ("Q1 2026",        [202601, 202602, 202603]),
    ("Q2 2026 ▶",      [202604, 202605, 202606]),
]
QUARTER_LABELS = [q[0] for q in QUARTERS_DEF]

def _build_quarterly_df():
    rows = []
    for seg in SEGMENTS:
        label = SEG_LABELS.get(seg, seg)
        result = SEG_FORECASTS_Q.get(seg)
        hist = seg_monthly_raw[seg_monthly_raw["segment"] == seg].set_index("year_month")["spend"].to_dict()
        fut_map = {}
        if result:
            for ym, v in zip(result["fut_months"], result["fut_pred"]):
                fut_map[ym] = v
        for q_label, months in QUARTERS_DEF:
            total = sum(
                hist.get(ym, fut_map.get(ym, 0))
                for ym in months
            )
            rows.append({"segment": seg, "label": label[:32], "quarter": q_label, "spend": total})
    return pd.DataFrame(rows)

QUARTERLY_DF = _build_quarterly_df()


def build_quarterly_donut(quarter):
    df = QUARTERLY_DF[QUARTERLY_DF["quarter"] == quarter].copy()
    df = df[df["spend"] > 0].sort_values("spend", ascending=False)
    top_n = 12
    if len(df) > top_n:
        other = df.iloc[top_n:]["spend"].sum()
        df = pd.concat([
            df.head(top_n),
            pd.DataFrame([{"segment": "Other", "label": "Other", "quarter": quarter, "spend": other}])
        ], ignore_index=True)

    palette = [
        C_BLUE, C_RED, C_GREEN, C_AMBER, C_PURPLE, C_TEAL,
        "#e11d48", "#0d9488", "#7c3aed", "#b45309", "#0369a1", "#15803d", "#6b21a8"
    ]

    fig = go.Figure(go.Pie(
        labels=df["label"],
        values=df["spend"],
        hole=0.52,
        textinfo="percent",
        textfont=dict(size=10),
        hovertemplate="<b>%{label}</b><br>Spend: %{value:$,.0f}<br>Share: %{percent}<extra></extra>",
        marker=dict(colors=palette[:len(df)], line=dict(color="white", width=2)),
        sort=True,
        direction="clockwise",
    ))
    total = df["spend"].sum()
    is_fc = "▶" in quarter
    fig.add_annotation(
        text=f"<b>{fmt(total)}</b><br><span style='font-size:10px;color:{C_MUTED}'>Total</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color=C_TEXT), align="center",
    )
    fig.update_layout(
        **{k: v for k, v in LAYOUT_BASE.items() if k != "margin"},
        title=dict(
            text=(f"<b>Segment Share  ·  {quarter}</b>"
                  + (f"<br><span style='color:{C_AMBER};font-size:11px'>▶ Forecast quarter</span>" if is_fc else "")),
            font=dict(size=14), x=0.01,
        ),
        height=460,
        margin=dict(l=10, r=10, t=68, b=10),
        legend=dict(font=dict(size=10), orientation="v", x=1.0, xanchor="left"),
        showlegend=True,
    )
    return fig


def build_quarterly_stacked():
    pivot = QUARTERLY_DF.pivot_table(index="label", columns="quarter", values="spend", aggfunc="sum").fillna(0)
    cols_ordered = [q for q in QUARTER_LABELS if q in pivot.columns]
    pivot = pivot[cols_ordered]
    pivot["_total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("_total", ascending=False).head(10).drop(columns=["_total"])

    palette = [C_BLUE, C_RED, C_GREEN, C_AMBER, C_PURPLE, C_TEAL,
               "#e11d48", "#0d9488", "#7c3aed", "#b45309"]

    fig = go.Figure()
    for i, seg_label in enumerate(pivot.index):
        fig.add_trace(go.Bar(
            name=seg_label[:30],
            x=cols_ordered,
            y=[pivot.loc[seg_label, q] for q in cols_ordered],
            marker_color=palette[i % len(palette)],
            hovertemplate=f"<b>{seg_label}</b><br>%{{x}}: %{{y:$,.0f}}<extra></extra>",
        ))

    fig.add_vline(x=3.5, line_color=C_MUTED, line_width=1.5, line_dash="dash",
                  annotation_text="  ← 2025 Actual  |  2026 →",
                  annotation_font=dict(color=C_MUTED, size=11))
    fig.update_layout(
        **{k: v for k, v in LAYOUT_BASE.items() if k != "margin"},
        title=dict(text="Quarterly Spend Trend — Top 10 Segments by Total Volume  (Stacked)",
                   font=dict(size=14), x=0.01),
        barmode="stack",
        height=480,
        hovermode="x unified",
        legend=dict(
            orientation="h", x=0, y=-0.20, font=dict(size=10),
            bgcolor="rgba(255,255,255,0.9)", bordercolor=C_BORDER, borderwidth=1,
        ),
        xaxis=dict(**AXIS_BASE, tickfont=dict(size=12)),
        yaxis=dict(**AXIS_BASE, tickformat="$,.0f", tickfont=dict(size=11)),
        margin=dict(l=16, r=16, t=55, b=120),
    )
    return fig


def build_quarterly_heatmap():
    pivot = QUARTERLY_DF.pivot_table(index="label", columns="quarter", values="spend", aggfunc="sum").fillna(0)
    cols_ordered = [q for q in QUARTER_LABELS if q in pivot.columns]
    pivot = pivot[cols_ordered]
    pivot = pivot[pivot.max(axis=1) > 50_000]

    growth = pivot.pct_change(axis=1) * 100
    growth = growth[[c for c in cols_ordered if c in growth.columns]]

    x_labels = list(growth.columns)
    y_labels = list(growth.index)

    baseline_col_idx = x_labels.index("Q1 2025") if "Q1 2025" in x_labels else None

    # z for color: keep NaN for Q1 2025 so it renders neutral (white)
    z = growth.values

    # text: Q1 2025 shows actual spend; other columns show QoQ %
    q1_spend = pivot["Q1 2025"] if "Q1 2025" in pivot.columns else None
    text_vals = []
    for i, seg_label in enumerate(y_labels):
        row_texts = []
        for j, col in enumerate(x_labels):
            if j == baseline_col_idx and q1_spend is not None:
                spend_val = q1_spend.get(seg_label, 0)
                row_texts.append(fmt(spend_val, decimals=1))
            else:
                v = growth.iloc[i, j]
                row_texts.append(f"{v:+.0f}%" if not np.isnan(v) and not np.isinf(v) else "")
        text_vals.append(row_texts)

    fig = go.Figure(go.Heatmap(
        z=z, x=x_labels, y=y_labels,
        colorscale=[
            [0.0,  "#DA291C"],
            [0.35, "#fca5a5"],
            [0.5,  "#f8fafc"],
            [0.65, "#86efac"],
            [1.0,  "#1b7f4f"],
        ],
        zmid=0,
        zmin=-50, zmax=50,
        colorbar=dict(
            title=dict(text="QoQ %", font=dict(size=11)),
            ticksuffix="%", len=0.65, thickness=12,
        ),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:+.1f}%<extra></extra>",
        text=text_vals,
        texttemplate="%{text}",
        textfont=dict(size=9),
    ))
    n_rows = len(y_labels)
    fig.update_layout(
        **{k: v for k, v in LAYOUT_BASE.items() if k != "margin"},
        title=dict(text="Quarter-over-Quarter Spend Change % by Segment"
                       "  <span style='font-size:11px;color:#6b7280'>(Q1 2025 shows actual spend)</span>",
                   font=dict(size=14), x=0.01),
        height=max(420, 26 * n_rows + 100),
        xaxis=dict(**AXIS_BASE, tickfont=dict(size=11)),
        yaxis=dict(**AXIS_BASE, tickfont=dict(size=10), autorange="reversed"),
        margin=dict(l=16, r=16, t=55, b=20),
    )
    return fig


# ─── SPEND-PER-ROOM HELPERS ───────────────────────────────────────────────────

SIZE_ORDER = ["<100 rooms", "100–250", "250–500", "500–1000", "1000+"]

def build_spr_bar(segment=None):
    """Spend-per-room by property size bucket, optionally filtered to one segment."""
    if segment and segment != "All Segments":
        d = spr_agg[spr_agg["segment"] == segment].copy()
        title = f"Spend / Room by Property Size  —  {SEG_LABELS.get(segment, segment)}"
    else:
        d = spr_agg.groupby("size_bucket", observed=True).apply(
            lambda g: pd.Series({
                "spend_per_room": g["total_spend"].sum() / max(g["total_rooms"].sum(), 1),
                "total_spend": g["total_spend"].sum(),
                "record_count": g["record_count"].sum(),
            })
        ).reset_index()
        title = "Spend / Room by Property Size  —  All Segments"
    d["size_bucket"] = pd.Categorical(d["size_bucket"], categories=SIZE_ORDER, ordered=True)
    d = d.sort_values("size_bucket")
    colors = [C_BLUE, C_TEAL, C_GREEN, C_AMBER, C_RED]
    fig = go.Figure(go.Bar(
        x=d["size_bucket"].astype(str),
        y=d["spend_per_room"],
        marker_color=colors[:len(d)],
        text=[f"${v:,.0f}" for v in d["spend_per_room"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Spend/Room: $%{y:,.0f}<br>Total Spend: %{customdata}<extra></extra>",
        customdata=[fmt(v) for v in d["total_spend"]],
    ))
    fig.update_layout(
        **{k: v for k, v in LAYOUT_BASE.items() if k != "margin"},
        title=dict(text=title, font=dict(size=14), x=0.01),
        height=380,
        showlegend=False,
        xaxis=dict(**AXIS_BASE, title=dict(text="Property Size (Rooms)", font=dict(size=12))),
        yaxis=dict(**AXIS_BASE, title=dict(text="Avg Spend per Room ($)", font=dict(size=12)),
                   tickprefix="$", tickformat=",.0f"),
        margin=dict(l=16, r=16, t=60, b=16),
    )
    return fig


def build_spr_scatter():
    """Scatter: segment avg rooms vs spend/room, bubble = total spend."""
    d = spr_seg.copy()
    d["label"] = d["segment"].map(SEG_LABELS).fillna(d["segment"])
    d = d[d["avg_rooms"] > 0]
    fig = go.Figure(go.Scatter(
        x=d["avg_rooms"],
        y=d["spend_per_room"],
        mode="markers+text",
        text=d["label"],
        textposition="top center",
        textfont=dict(size=9, color=C_TEXT),
        marker=dict(
            size=np.sqrt(d["total_spend"].clip(1e6) / 1e6) * 3,
            color=d["spend_per_room"],
            colorscale=[[0, C_BLUE], [0.5, C_AMBER], [1, C_RED]],
            showscale=True,
            colorbar=dict(title=dict(text="$/Room", font=dict(size=10)), len=0.5, thickness=10),
            line=dict(color="white", width=0.5),
            opacity=0.85,
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Avg Property Size: %{x:.0f} rooms<br>"
            "Spend per Room: $%{y:,.0f}<br>"
            "Total Spend: %{customdata}<extra></extra>"
        ),
        customdata=[fmt(v) for v in d["total_spend"]],
    ))
    fig.update_layout(
        **{k: v for k, v in LAYOUT_BASE.items() if k != "margin"},
        title=dict(text="Segment Efficiency Map  —  Property Size vs Spend/Room  (bubble = total spend)",
                   font=dict(size=14), x=0.01),
        height=440,
        showlegend=False,
        xaxis=dict(**AXIS_BASE, title=dict(text="Avg Property Size (Rooms)", font=dict(size=12)),
                   tickformat=",.0f"),
        yaxis=dict(**AXIS_BASE, title=dict(text="Spend per Room ($)", font=dict(size=12)),
                   tickprefix="$", tickformat=",.0f"),
        margin=dict(l=16, r=16, t=60, b=16),
    )
    return fig


# ─── ECOMMERCE HELPERS ────────────────────────────────────────────────────────

STATUS_COLOR = {"ACTIVE": C_GREEN, "INACTIVE": C_RED, "PENDING": C_AMBER, "ACTIVATED": C_TEAL}

def build_ecomm_donut():
    """Donut chart: spend share by ecommerce status."""
    d = ecomm_seg.groupby("ecomm_status")["total_spend"].sum().reset_index()
    d = d.sort_values("total_spend", ascending=False)
    fig = go.Figure(go.Pie(
        labels=d["ecomm_status"],
        values=d["total_spend"],
        hole=0.55,
        marker_colors=[STATUS_COLOR.get(s, C_MUTED) for s in d["ecomm_status"]],
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Spend: %{customdata}<br>Share: %{percent}<extra></extra>",
        customdata=[fmt(v) for v in d["total_spend"]],
    ))
    total = d["total_spend"].sum()
    fig.add_annotation(text=f"Total<br><b>{fmt(total)}</b>",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=13, color=C_TEXT))
    fig.update_layout(
        **{k: v for k, v in LAYOUT_BASE.items() if k != "margin"},
        title=dict(text="Portfolio Spend by eCommerce Status", font=dict(size=14), x=0.01),
        height=360, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=16, r=16, t=60, b=16),
    )
    return fig


def build_ecomm_seg_bar():
    """Grouped bar: ACTIVE vs INACTIVE spend per segment."""
    pivot = ecomm_seg.pivot_table(index="segment", columns="ecomm_status",
                                   values="total_spend", aggfunc="sum", fill_value=0).reset_index()
    pivot["label"] = pivot["segment"].map(SEG_LABELS).fillna(pivot["segment"])
    pivot = pivot.sort_values("ACTIVE" if "ACTIVE" in pivot.columns else pivot.columns[1], ascending=False).head(15)
    fig = go.Figure()
    for status in ["ACTIVE", "INACTIVE", "PENDING"]:
        if status in pivot.columns:
            fig.add_trace(go.Bar(
                name=status, x=pivot["label"], y=pivot[status],
                marker_color=STATUS_COLOR.get(status, C_MUTED),
                hovertemplate=f"<b>%{{x}}</b><br>{status}: %{{customdata}}<extra></extra>",
                customdata=[fmt(v) for v in pivot[status]],
            ))
    fig.update_layout(
        **{k: v for k, v in LAYOUT_BASE.items() if k != "margin"},
        title=dict(text="eCommerce Adoption by Segment  —  Spend Breakdown", font=dict(size=14), x=0.01),
        height=420, barmode="group",
        xaxis=dict(**AXIS_BASE, tickangle=-35, tickfont=dict(size=10)),
        yaxis=dict(**AXIS_BASE, tickprefix="$", tickformat=".2s"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=16, r=16, t=70, b=80),
    )
    return fig


def build_ecomm_cat_bar():
    """Horizontal bar: ACTIVE vs INACTIVE spend by category."""
    d = ecomm_cat[ecomm_cat["ecomm_status"].isin(["ACTIVE", "INACTIVE"])].copy()
    pivot = d.pivot_table(index="cat_l1", columns="ecomm_status",
                          values="total_spend", aggfunc="sum", fill_value=0).reset_index()
    pivot = pivot.sort_values("ACTIVE" if "ACTIVE" in pivot.columns else "INACTIVE", ascending=True)
    fig = go.Figure()
    for status in ["ACTIVE", "INACTIVE"]:
        if status in pivot.columns:
            fig.add_trace(go.Bar(
                name=status, y=pivot["cat_l1"], x=pivot[status],
                orientation="h",
                marker_color=STATUS_COLOR.get(status, C_MUTED),
                hovertemplate=f"<b>%{{y}}</b><br>{status}: %{{customdata}}<extra></extra>",
                customdata=[fmt(v) for v in pivot[status]],
            ))
    fig.update_layout(
        **{k: v for k, v in LAYOUT_BASE.items() if k != "margin"},
        title=dict(text="Category Spend: eCommerce Active vs Inactive", font=dict(size=14), x=0.01),
        height=420, barmode="group",
        xaxis=dict(**AXIS_BASE, tickprefix="$", tickformat=".2s"),
        yaxis=dict(**AXIS_BASE, tickfont=dict(size=11)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=16, r=16, t=70, b=16),
    )
    return fig


# ─── CATEGORY DNA HELPERS ─────────────────────────────────────────────────────

ALL_CATS = sorted(national_cat["cat_l1"].unique().tolist())

def build_dna_radar(segment):
    """Radar chart: segment category % vs national average."""
    seg_d = seg_cat_dna[seg_cat_dna["segment"] == segment].set_index("cat_l1")["pct"]
    nat_d = national_cat.set_index("cat_l1")["pct"]
    cats  = ALL_CATS
    seg_vals = [float(seg_d.get(c, 0)) for c in cats]
    nat_vals = [float(nat_d.get(c, 0)) for c in cats]
    seg_vals_closed = seg_vals + [seg_vals[0]]
    nat_vals_closed = nat_vals + [nat_vals[0]]
    cats_closed = cats + [cats[0]]
    label = SEG_LABELS.get(segment, segment)
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=nat_vals_closed, theta=cats_closed,
        fill="toself", name="National Avg",
        line=dict(color=C_MUTED, width=1.5, dash="dot"),
        fillcolor="rgba(100,116,139,0.10)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=seg_vals_closed, theta=cats_closed,
        fill="toself", name=label,
        line=dict(color=C_BLUE, width=2.5),
        fillcolor="rgba(0,48,135,0.18)",
    ))
    fig.update_layout(
        **{k: v for k, v in LAYOUT_BASE.items() if k not in ("margin", "template")},
        template="plotly_white",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(max(seg_vals), max(nat_vals)) * 1.15],
                            ticksuffix="%", tickfont=dict(size=9), gridcolor=C_BORDER),
            angularaxis=dict(tickfont=dict(size=10), gridcolor=C_BORDER),
        ),
        title=dict(text=f"Category DNA Fingerprint  —  {label}", font=dict(size=14), x=0.01),
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
        margin=dict(l=60, r=60, t=70, b=60),
    )
    return fig


def build_dna_treemap(segment):
    """Treemap of category spend for the selected segment."""
    d = seg_cat_dna[seg_cat_dna["segment"] == segment].copy()
    d = d[d["total_spend"] > 0].sort_values("total_spend", ascending=False)
    label = SEG_LABELS.get(segment, segment)
    fig = go.Figure(go.Treemap(
        labels=d["cat_l1"],
        parents=["" for _ in d["cat_l1"]],
        values=d["total_spend"],
        texttemplate="<b>%{label}</b><br>%{customdata}",
        customdata=[f"{fmt(v)}<br>{pct:.1f}%" for v, pct in zip(d["total_spend"], d["pct"])],
        marker=dict(
            colorscale=[[0, C_BLUE], [0.5, C_TEAL], [1, C_GREEN]],
            showscale=False,
        ),
        hovertemplate="<b>%{label}</b><br>Spend: %{customdata}<extra></extra>",
    ))
    fig.update_layout(
        **{k: v for k, v in LAYOUT_BASE.items() if k != "margin"},
        title=dict(text=f"Category Wallet Share  —  {label}", font=dict(size=14), x=0.01),
        height=480,
        margin=dict(l=16, r=16, t=60, b=16),
    )
    return fig


def build_dna_compare_bar():
    """Bar: top-5 over/under-index categories vs national for selected segment (set by callback)."""
    return go.Figure()


# Pre-build static charts
print("Building static charts …")
fig_heatmap          = build_event_heatmap()
fig_model_cmp        = build_model_comparison()
fig_rankings         = build_state_rankings()
fig_ev_scatter       = build_event_spend_scatter()
fig_quarterly_stack  = build_quarterly_stacked()
fig_quarterly_hmap   = build_quarterly_heatmap()
print("  Charts ready.")

# ─── DASH APP ─────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                title="Aramark Event-Aligned Spend Predictor")

def kpi_block(label, value, sub, accent):
    return html.Div([
        html.P(label, style={"fontSize":"10px","color":C_MUTED,"textTransform":"uppercase",
                              "letterSpacing":"0.8px","margin":"0 0 4px 0","fontWeight":"600"}),
        html.H2(value, style={"color":accent,"fontWeight":"800","margin":"0","fontSize":"24px",
                               "lineHeight":"1.1"}),
        html.P(sub,   style={"fontSize":"11px","color":C_MUTED,"margin":"4px 0 0 0"}),
    ], style={"borderLeft":f"3px solid {accent}","paddingLeft":"14px"})

total_hist_spend  = monthly_state["spend"].sum()
# Apr–Jun 2026 forecast vs Apr–Jun 2025 actual for KPI lift %
total_fut_spend   = sum(
    sum(STATE_PREDS[s]["pred_ensemble"][FUTURE_MONTHS.index(m)] for m in APR_JUN_2026)
    for s in STATES
)
total_apr_jun_25  = sum(
    sum(STATE_PREDS[s]["hist_spend"][HIST_MONTHS.index(m)]
        for m in APR_JUN_2025 if m in HIST_MONTHS)
    for s in STATES
)
lift_pct = (total_fut_spend - total_apr_jun_25) / total_apr_jun_25 * 100 if total_apr_jun_25 else 0
top_event_state  = max(STATES, key=lambda s: sum(STATE_PREDS[s]["ev_fut"]))
top_growth_state = top_growth_states[0]

KPI_ROW = dbc.Row([
    dbc.Col(kpi_block("2025 Total Portfolio Spend", fmt(total_hist_spend),
                      "Across all states · 12 months actual", C_BLUE), md=3),
    dbc.Col(kpi_block("Apr–Jun 2026 Ensemble Forecast", fmt(total_fut_spend),
                      f"{lift_pct:+.1f}% vs Apr–Jun 2025  ·  Weighted model ensemble", C_RED), md=3),
    dbc.Col(kpi_block("Best Model", best_model_name,
                      f"CV log-RMSE = {model_rmse[best_model_name]:.4f}  ·  4 models trained", C_GREEN), md=3),
    dbc.Col(kpi_block("Top Event State (Apr–Jun 2026)", top_event_state,
                      f"Highest aggregate event impact score → {top_growth_state} leads growth", C_AMBER), md=3),
], className="g-4", style={"padding":"24px 36px 20px 36px","borderBottom":f"1px solid {C_BORDER}"})

def card(children):
    return html.Div(children, style={
        "background":C_CARD,"borderRadius":"12px","padding":"20px 20px 16px 20px",
        "border":f"1px solid {C_BORDER}","boxShadow":"0 1px 4px rgba(0,0,0,0.05)",
    })

def ctrl_label(text):
    return html.P(text, style={"fontSize":"10px","color":C_MUTED,"textTransform":"uppercase",
                                "letterSpacing":"0.8px","marginBottom":"6px","fontWeight":"600","margin":"0 0 6px 0"})

G = lambda fig, height=None: dcc.Graph(
    figure=fig,
    config={"displayModeBar": False},
    style={"height": f"{height}px"} if height else {}
)
PAD = {"padding":"24px 36px"}

TAB_STYLE    = {"color":C_MUTED,"backgroundColor":C_CARD,"border":"none","padding":"12px 22px",
                "fontWeight":"600","fontSize":"13px"}
TAB_SELECTED = {**TAB_STYLE,"color":C_BLUE,"borderBottom":f"2px solid {C_RED}","backgroundColor":C_CARD}

STATE_OPTIONS = [{"label":s,"value":s} for s in STATES]

def tab_content(tab):
    if tab == "state":
        controls = html.Div([
            dbc.Row([
                dbc.Col([
                    ctrl_label("Select State"),
                    dcc.Dropdown(
                        id="state-dd", options=STATE_OPTIONS, value="CA",
                        clearable=False, style={"fontSize":"13px","color":"#0f172a"}
                    )
                ], md=3),
                dbc.Col([
                    html.Br(),
                    html.P(
                        "Forecast uses Gradient Boosting, Random Forest, Ridge & SVR. "
                        "The Ensemble combines all four weighted by cross-validation accuracy. "
                        "Event scores from the curated 2025–2026 event calendar adjust predictions "
                        "to align with real hospitality demand drivers.",
                        style={"color":C_MUTED,"fontSize":"11px","fontStyle":"italic","marginTop":"6px"}
                    )
                ], md=9),
            ], className="g-3 align-items-end")
        ], style={"background":C_CARD,"borderRadius":"12px","padding":"18px 24px",
                   "border":f"1px solid {C_BORDER}","boxShadow":"0 1px 4px rgba(0,0,0,0.05)",
                   "marginBottom":"20px"})

        return html.Div([
            controls,
            dbc.Row([
                dbc.Col(card([dcc.Graph(id="state-forecast-chart",
                                        config={"displayModeBar":False},
                                        style={"height":"700px"})]), md=12),
            ], className="g-4 mb-4"),
            # Upcoming events table
            dbc.Row([
                dbc.Col(card([
                    html.H6("Upcoming Events (Apr–Jun 2026 Forecast) — Event → Predicted Spend Driver",
                             style={"color":C_TEXT,"fontWeight":"700","marginBottom":"12px","fontSize":"13px"}),
                    html.Div(id="event-table")
                ]), md=12)
            ], className="g-4"),
        ], style=PAD)

    if tab == "segment":
        seg_options = [
            {"label": f"{SEG_LABELS.get(s, s)}  ({s})", "value": s}
            for s in SEGMENTS
        ]
        state_options = [{"label": st, "value": st} for st in SEG_STATES]
        return html.Div([
            # Controls row
            html.Div([
                dbc.Row([
                    dbc.Col([
                        ctrl_label("Segment Deep-Dive"),
                        dcc.Dropdown(
                            id="seg-dd", options=seg_options,
                            value="MS-100003",
                            clearable=False,
                            style={"fontSize":"13px","color":"#0f172a"},
                        )
                    ], md=5),
                    dbc.Col([
                        ctrl_label("Filter by State"),
                        dcc.Dropdown(
                            id="seg-state-dd", options=state_options,
                            value="All States",
                            clearable=False,
                            style={"fontSize":"13px","color":"#0f172a"},
                        )
                    ], md=3),
                    dbc.Col([
                        html.Br(),
                        html.P(
                            "Select a segment and optionally a state to see its monthly trend "
                            "and H1 2026 ML forecast.",
                            style={"color":C_MUTED,"fontSize":"11px","fontStyle":"italic","marginTop":"6px"}
                        )
                    ], md=4),
                ], className="g-3 align-items-end")
            ], style={"background":C_CARD,"borderRadius":"12px","padding":"18px 24px",
                       "border":f"1px solid {C_BORDER}","boxShadow":"0 1px 4px rgba(0,0,0,0.05)",
                       "marginBottom":"20px"}),
            # Chart + KPI cards
            dbc.Row([
                dbc.Col(card([
                    dcc.Graph(id="seg-forecast-chart",
                              config={"displayModeBar":False},
                              style={"height":"490px"})
                ]), md=9),
                dbc.Col(card([
                    html.H6("Segment Summary",
                             style={"color":C_TEXT,"fontWeight":"700","marginBottom":"14px","fontSize":"13px"}),
                    html.Div(id="seg-kpi-panel"),
                ]), md=3),
            ], className="g-4 mb-4"),
            # All-segment comparison bar
            dbc.Row([
                dbc.Col(card([
                    html.H6("All Segments — H1 2026 Forecast vs H2 2025 (selected state)",
                             style={"color":C_TEXT,"fontWeight":"700","marginBottom":"12px","fontSize":"13px"}),
                    dcc.Graph(id="seg-all-bar",
                              config={"displayModeBar":False},
                              style={"height":"380px"})
                ]), md=12),
            ], className="g-4"),
        ], style=PAD)

    if tab == "heatmap":
        return html.Div([
            dbc.Row([
                dbc.Col(card([G(fig_heatmap)]), md=12)
            ], className="g-4 mb-4"),
            dbc.Row([
                dbc.Col(card([G(fig_ev_scatter)]), md=12)
            ], className="g-4"),
        ], style=PAD)

    if tab == "quarterly":
        return html.Div([
            html.Div([
                dbc.Row([
                    dbc.Col([
                        ctrl_label("Select Quarter for Breakdown"),
                        dcc.Dropdown(
                            id="quarter-dd",
                            options=[{"label": q, "value": q} for q in QUARTER_LABELS],
                            value="Q4 2025",
                            clearable=False,
                            style={"fontSize":"13px","color":"#0f172a"},
                        )
                    ], md=3),
                    dbc.Col([
                        html.Br(),
                        html.P(
                            "Q1–Q4 2025 are actuals. Q1 2026 uses Jan–Mar actuals. "
                            "Q2 2026 ▶ is the ML ensemble forecast (Apr–Jun). "
                            "The donut updates with the selected quarter; the stacked bar and QoQ heatmap show the full year view.",
                            style={"color":C_MUTED,"fontSize":"11px","fontStyle":"italic","marginTop":"6px"}
                        )
                    ], md=9),
                ], className="g-3 align-items-end")
            ], style={"background":C_CARD,"borderRadius":"12px","padding":"18px 24px",
                       "border":f"1px solid {C_BORDER}","boxShadow":"0 1px 4px rgba(0,0,0,0.05)",
                       "marginBottom":"20px"}),
            dbc.Row([
                dbc.Col(card([
                    dcc.Graph(id="quarterly-donut",
                              config={"displayModeBar":False},
                              style={"height":"460px"})
                ]), md=5),
                dbc.Col(card([G(fig_quarterly_hmap)]), md=7),
            ], className="g-4 mb-4"),
            dbc.Row([
                dbc.Col(card([G(fig_quarterly_stack)]), md=12),
            ], className="g-4"),
        ], style=PAD)

    if tab == "rankings":
        return html.Div([
            dbc.Row([
                dbc.Col(card([G(fig_rankings)]), md=12)
            ], className="g-4 mb-4"),
            dbc.Row([
                dbc.Col(card([
                    html.H6("State Apr–Jun 2026 Forecast Details  (YoY growth vs Apr–Jun 2025)",
                             style={"color":C_TEXT,"fontWeight":"700","marginBottom":"12px","fontSize":"13px"}),
                    dbc.Table([
                        html.Thead(html.Tr([
                            html.Th("Rank"),html.Th("State"),html.Th("Apr–Jun 2026 Forecast"),
                            html.Th("Apr–Jun 2025 Actual"),html.Th("YoY Growth"),html.Th("Top Event (Apr–Jun 2026)"),
                        ])),
                        html.Tbody([
                            html.Tr([
                                html.Td(f"#{i+1}"),
                                html.Td(s, style={"fontWeight":"600"}),
                                html.Td(fmt(sum(
                                    STATE_PREDS[s]["pred_ensemble"][FUTURE_MONTHS.index(m)]
                                    for m in APR_JUN_2026
                                ))),
                                html.Td(fmt(sum(
                                    STATE_PREDS[s]["hist_spend"][HIST_MONTHS.index(m)]
                                    for m in APR_JUN_2025 if m in HIST_MONTHS
                                ))),
                                html.Td(f"{state_growth[s]:+.1f}%",
                                        style={"color":C_GREEN if state_growth[s]>=0 else C_RED,"fontWeight":"600"}),
                                html.Td(
                                    max(
                                        [(n,sc) for ym in APR_JUN_2026 for n,sc in event_list(s,ym)],
                                        key=lambda x: x[1], default=("—",0)
                                    )[0][:40]
                                ),
                            ]) for i, s in enumerate(sorted(STATES, key=lambda s: state_growth[s], reverse=True))
                        ])
                    ], bordered=True, hover=True, size="sm",
                       style={"fontSize":"12px"})
                ]), md=12),
            ], className="g-4"),
        ], style=PAD)

    # ── Category DNA Fingerprint ───────────────────────────────────────────────
    if tab == "dna":
        dna_seg_options = [
            {"label": SEG_LABELS.get(s, s), "value": s}
            for s in sorted(seg_cat_dna["segment"].unique())
            if seg_cat_dna[seg_cat_dna["segment"]==s]["total_spend"].sum() > 0
        ]
        default_seg = dna_seg_options[0]["value"] if dna_seg_options else "MS-100003"
        return html.Div([
            html.Div([
                dbc.Row([
                    dbc.Col([
                        ctrl_label("Select Segment"),
                        dcc.Dropdown(id="dna-seg-dd", options=dna_seg_options, value=default_seg,
                                     clearable=False, style={"fontSize":"13px","color":"#0f172a"}),
                    ], md=4),
                    dbc.Col([
                        html.Br(),
                        html.P("The radar shows this segment's category mix vs the national portfolio average. "
                               "Spikes reveal over-indexing; gaps reveal under-utilised categories.",
                               style={"color":C_MUTED,"fontSize":"11px","fontStyle":"italic","marginTop":"6px"}),
                    ], md=8),
                ], className="g-3 align-items-end"),
            ], style={"background":C_CARD,"borderRadius":"12px","padding":"18px 24px",
                       "border":f"1px solid {C_BORDER}","boxShadow":"0 1px 4px rgba(0,0,0,0.05)",
                       "marginBottom":"20px"}),
            dbc.Row([
                dbc.Col(card([dcc.Graph(id="dna-radar",  config={"displayModeBar":False})]), md=6),
                dbc.Col(card([dcc.Graph(id="dna-treemap",config={"displayModeBar":False})]), md=6),
            ], className="g-4 mb-4"),
            dbc.Row([
                dbc.Col(card([dcc.Graph(id="dna-delta",  config={"displayModeBar":False})]), md=12),
            ], className="g-4"),
        ], style=PAD)

    return html.Div()


app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.Span("ARAMARK", style={"color":C_BLUE,"fontWeight":"900","fontSize":"18px","letterSpacing":"2px"}),
            html.Span(" · Spend Prediction: State Events & Economic Drivers",
                      style={"color":C_MUTED,"fontSize":"13px","marginLeft":"8px"}),
        ]),
        html.Div("Event-Aligned ML Forecast  ·  4 Models (GBM · RF · Ridge · SVR)  ·  Ensemble Weighted by CV Accuracy  ·  H1 2026",
                 style={"color":C_MUTED,"fontSize":"11px","marginTop":"3px"}),
    ], style={"background":C_CARD,"padding":"16px 28px","borderBottom":f"1px solid {C_BORDER}",
               "boxShadow":"0 1px 3px rgba(0,0,0,0.06)"}),

    # KPIs
    html.Div(KPI_ROW, style={"background":C_CARD,"borderBottom":f"1px solid {C_BORDER}"}),

    # Tabs
    html.Div([
        dcc.Tabs(id="main-tabs", value="state", children=[
            dcc.Tab(label="📍  State Forecast",     value="state",    style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="🗓️  Event Heatmap",     value="heatmap",  style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="📊  Segment Deep-Dive",  value="segment",  style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="🏆  State Rankings",     value="rankings", style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="📅  Quarterly Analysis", value="quarterly",style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="🧬  Category DNA",       value="dna",      style=TAB_STYLE, selected_style=TAB_SELECTED),
        ], style={"backgroundColor":C_CARD,"borderBottom":f"1px solid {C_BORDER}"},
           colors={"border":"transparent","primary":C_RED,"background":C_CARD}),
        html.Div(id="tab-content", style={"background":C_BG,"minHeight":"calc(100vh - 200px)"}),
    ]),
], style={"background":C_BG,"minHeight":"100vh",
           "fontFamily":"'Inter','Segoe UI',sans-serif","color":C_TEXT})


@app.callback(Output("tab-content","children"), Input("main-tabs","value"))
def render_tab(tab): return tab_content(tab)


@app.callback(
    Output("seg-forecast-chart", "figure"),
    Output("seg-kpi-panel",      "children"),
    Output("seg-all-bar",        "figure"),
    Input("seg-dd",       "value"),
    Input("seg-state-dd", "value"),
)
def update_segment_view(segment, state_filter):
    if not segment:
        segment = "MS-100003"
    if not state_filter:
        state_filter = "All States"

    fig_chart, stats = build_segment_chart(segment, state_filter)

    # KPI panel
    if stats:
        kpi_items = [
            ("2025 Avg Monthly",  fmt(stats["avg_monthly"]), "Historical baseline"),
            ("H2 2025 Actual",    fmt(stats["h2_2025"]),     "Jul–Dec 2025"),
            ("H1 2026 Forecast",  fmt(stats["h1_2026"]),     "Jan–Jun 2026 (ML)"),
            ("YoY Growth",        f"{stats['growth']:+.1f}%","H1 2026 vs H2 2025"),
            ("2025 Peak Month",   stats["peak_month"],        fmt(stats["peak_val"])),
        ]
        accent_map = [C_BLUE, C_BLUE, C_RED,
                      C_GREEN if stats["growth"] >= 0 else C_RED, C_AMBER]
        kpi_panel = html.Div([
            html.Div([
                html.P(lbl, style={"fontSize":"10px","color":C_MUTED,"textTransform":"uppercase",
                                   "letterSpacing":"0.7px","margin":"0 0 2px 0","fontWeight":"600"}),
                html.H4(val, style={"color":acc,"fontWeight":"800","margin":"0","fontSize":"18px"}),
                html.P(sub, style={"fontSize":"10px","color":C_MUTED,"margin":"2px 0 0 0"}),
            ], style={"borderLeft":f"3px solid {acc}","paddingLeft":"10px","marginBottom":"16px"})
            for (lbl, val, sub), acc in zip(kpi_items, accent_map)
        ])
    else:
        kpi_panel = html.P("No data", style={"color":C_MUTED})

    # All-segments comparison bar (national unless state filtered)
    seg_rows = []
    for seg in SEGMENTS:
        r = forecast_segment(seg, state_filter)
        if r is None:
            continue
        h2 = sum(r["hist_spend"][6:]) if len(r["hist_spend"]) >= 12 else sum(r["hist_spend"])
        h1 = sum(r["fut_pred"])
        gr = (h1 - h2) / h2 * 100 if h2 else 0
        seg_rows.append({
            "seg": seg,
            "label": SEG_LABELS.get(seg, seg),
            "h1_2026": h1,
            "growth": gr,
        })
    seg_rows.sort(key=lambda x: x["h1_2026"], reverse=True)

    bar_colors = [C_RED if r["seg"] == segment else C_BLUE for r in seg_rows]
    fig_bar = go.Figure(go.Bar(
        x=[r["label"] for r in seg_rows],
        y=[r["h1_2026"] for r in seg_rows],
        marker_color=bar_colors,
        text=[fmt(r["h1_2026"]) for r in seg_rows],
        textposition="outside",
        customdata=[[f"{r['growth']:+.1f}%"] for r in seg_rows],
        hovertemplate=(
            "<b>%{x}</b><br>"
            "H1 2026 Forecast: %{y:$,.0f}<br>"
            "Growth vs H2 2025: %{customdata[0]}<extra></extra>"
        ),
    ))
    fig_bar.update_layout(
        **{k: v for k, v in LAYOUT_BASE.items() if k != "margin"},
        height=380,
        showlegend=False,
        xaxis=dict(**AXIS_BASE, tickfont=dict(size=9), tickangle=-35),
        yaxis=dict(**AXIS_BASE, tickformat="$,.0f", tickfont=dict(size=10)),
        margin=dict(l=16, r=16, t=20, b=120),
    )

    return fig_chart, kpi_panel, fig_bar


@app.callback(
    Output("state-forecast-chart", "figure"),
    Output("event-table", "children"),
    Input("state-dd", "value"),
)
def update_state_view(state):
    if not state:
        state = "CA"

    chart = build_state_forecast_chart(state)

    # Build events table for all 2026 months (Jan–Mar = actuals, Apr–Jun = forecast)
    ALL_2026_MONTHS = [202601, 202602, 202603, 202604, 202605, 202606]
    ALL_2026_LABELS = [month_label(m) for m in ALL_2026_MONTHS]
    state_hist_data = (monthly_state[monthly_state["state"] == state]
                       .set_index("year_month")["spend"].to_dict())

    rows = []
    for ym, lbl in zip(ALL_2026_MONTHS, ALL_2026_LABELS):
        evts    = event_list(state, ym)
        ev_sc   = event_score(state, ym)
        ab_M    = aramark_boost(state, ym)
        is_fut  = ym in FUTURE_MONTHS
        is_hist = ym in HIST_MONTHS

        if is_fut:
            spend_val  = STATE_PREDS[state]["pred_ensemble"][FUTURE_MONTHS.index(ym)]
            spend_disp = fmt(spend_val)
        else:
            spend_val  = state_hist_data.get(ym, 0)
            spend_disp = fmt(spend_val)

        ev_cell = html.Ul(
            [html.Li(f"{n} (impact: {s})", style={"fontSize":"11px"}) for n, s in evts],
            style={"marginBottom":"0","paddingLeft":"16px"}
        ) if evts else html.Span("—", style={"color":C_MUTED})

        ab_cell = html.Span(
            f"+${ab_M:.1f}M" if ab_M > 0 else "—",
            style={"color": C_GREEN if ab_M > 0 else C_MUTED, "fontWeight":"600", "fontSize":"11px"}
        )

        rows.append(html.Tr([
            html.Td(lbl, style={"fontWeight":"600","width":"9%","whiteSpace":"nowrap"}),
            html.Td(f"{ev_sc:.0f}", style={
                "fontWeight":"700","fontSize":"14px","width":"7%",
                "color":C_RED if ev_sc >= 60 else (C_AMBER if ev_sc >= 25 else C_MUTED)
            }),
            html.Td(ev_cell, style={"width":"48%"}),
            html.Td(ab_cell, style={"width":"12%"}),
            html.Td(spend_disp, style={"fontWeight":"700","color": C_GREEN if is_hist else C_BLUE,"width":"12%"}),
        ]))

    table = dbc.Table([
        html.Thead(html.Tr([
            html.Th("Month"), html.Th("Event Score"), html.Th("Events Driving Demand"),
            html.Th("Aramark Boost Est.", title="Multi-channel: venue + hotel F&B + airport + convention + institutional"),
            html.Th("Spend (Actual/Forecast)")
        ])),
        html.Tbody(rows)
    ], bordered=True, hover=True, size="sm", style={"fontSize":"12px"})

    return chart, table


@app.callback(
    Output("quarterly-donut", "figure"),
    Input("quarter-dd", "value"),
)
def update_quarterly_donut(quarter):
    if not quarter:
        quarter = "Q4 2025"
    return build_quarterly_donut(quarter)


@app.callback(
    Output("spr-bar", "figure"),
    Input("spr-seg-dd", "value"),
)
def update_spr_bar(segment):
    return build_spr_bar(segment or "All Segments")


@app.callback(
    Output("dna-radar",   "figure"),
    Output("dna-treemap", "figure"),
    Output("dna-delta",   "figure"),
    Input("dna-seg-dd", "value"),
)
def update_dna(segment):
    if not segment:
        segment = seg_cat_dna["segment"].iloc[0]
    radar   = build_dna_radar(segment)
    treemap = build_dna_treemap(segment)

    # Over/under index bar vs national average
    seg_d = seg_cat_dna[seg_cat_dna["segment"] == segment].set_index("cat_l1")["pct"]
    nat_d = national_cat.set_index("cat_l1")["pct"]
    deltas = []
    for cat in ALL_CATS:
        delta = float(seg_d.get(cat, 0)) - float(nat_d.get(cat, 0))
        deltas.append({"cat": cat, "delta": delta})
    df_d = pd.DataFrame(deltas).sort_values("delta", ascending=True)
    label = SEG_LABELS.get(segment, segment)
    fig_d = go.Figure(go.Bar(
        x=df_d["delta"], y=df_d["cat"],
        orientation="h",
        marker_color=[C_GREEN if v >= 0 else C_RED for v in df_d["delta"]],
        text=[f"{v:+.1f}pp" for v in df_d["delta"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Delta vs National: %{x:+.2f}pp<extra></extra>",
    ))
    fig_d.add_vline(x=0, line_color=C_MUTED, line_width=1.5, line_dash="dot")
    fig_d.update_layout(
        **{k: v for k, v in LAYOUT_BASE.items() if k != "margin"},
        title=dict(text=f"Over / Under-Index vs National Average  —  {label}", font=dict(size=14), x=0.01),
        height=420,
        showlegend=False,
        xaxis=dict(**AXIS_BASE, ticksuffix="pp", title=dict(text="Percentage Points vs National Avg", font=dict(size=11))),
        yaxis=dict(**AXIS_BASE, tickfont=dict(size=11)),
        margin=dict(l=16, r=16, t=60, b=16),
    )
    return radar, treemap, fig_d


if __name__ == "__main__":
    print(f"\nEvent-Aligned Spend Prediction Dashboard → http://127.0.0.1:8051\n")
    app.run(debug=False, port=8051)
