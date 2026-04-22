# Aramark Spend Prediction Dashboard

Event-aligned ML spend forecasting for Aramark's H1 2026 portfolio, built on state-level SRF data with four trained models (GBM, Random Forest, Ridge, SVR).

## Prerequisites

- Python 3.9+
- The raw data file `Andrew_Meszaros_SRF_2026-04-01-0936.csv` in the project root

Install dependencies:

```bash
pip install pandas numpy scikit-learn plotly dash dash-bootstrap-components openpyxl Pillow kaleido
```

## Running the Interactive Dashboard

```bash
cd Dashboard
python state_event_prediction_dashboard.py
```

Then open [http://127.0.0.1:8051](http://127.0.0.1:8051) in your browser.

The dashboard loads the CSV, engineers features, trains all four ML models, and starts a Dash web server. First launch takes ~1–2 minutes while models train.

## Generating the PDF Report

```bash
cd Dashboard
python generate_dashboard_pdf.py
```

This exports every chart section to `Dashboard/aramark_spend_prediction_dashboard.pdf`. It imports the dashboard module (which re-trains models), then renders each chart to PNG and assembles a multi-page PDF. Expect ~3–5 minutes to complete.

## Project Structure

```
Aramark/
├── Andrew_Meszaros_SRF_2026-04-01-0936.csv   # Raw spend data (43M rows)
├── USA_Major_Events_2026.xlsx                  # Event calendar
├── Dashboard/
│   ├── state_event_prediction_dashboard.py    # Main Dash app (run this)
│   ├── generate_dashboard_pdf.py              # PDF export script
│   ├── state_event_analysis.py                # Event feature engineering
│   ├── advanced_ml.py                         # ML model helpers
│   ├── feature_engineering.py                 # Feature pipeline
│   ├── visualizations.py / visualizations_v2.py
│   └── *.parquet / *.csv                      # Pre-aggregated data caches
```

## Dashboard Sections

| Section | Description |
|---|---|
| 1. State Forecast | Per-state event-aligned ML predictions for top 12 states |
| 2. Event Heatmap | State × month impact scores with scatter overlay |
| 3. Segment Deep-Dive | Monthly trend + H1 2026 forecast by business segment |
| 4. State Rankings | YoY growth forecast Apr–Jun 2026 vs Apr–Jun 2025 |
| 5. Quarterly Analysis | Q1–Q4 2025 actual + Q1 2026 actual + Q2 2026 forecast |
| 6. Category DNA | Segment category mix vs national portfolio average |

## Notes

- The parquet files in `Dashboard/` are pre-aggregated caches that speed up load time; deleting them forces a full rebuild from the CSV.
- PDF generation requires `kaleido` for Plotly static image export and `Pillow` for PDF assembly.
- Font rendering in the PDF uses `/System/Library/Fonts/Helvetica.ttc` (macOS only); on other platforms it falls back to the default PIL font.
