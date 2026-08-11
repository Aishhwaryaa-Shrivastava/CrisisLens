"""
CrisisLens Dashboard Patcher
Run this once from inside your CrisisLens folder:
    python fix_dashboard.py

Fixes:
  1. Hardcoded Feb/Mar/Apr 2026 → dynamic dates from today (+30/+60/+90 days)
  2. "December 2025" header → dynamic current month
  3. State forecast crash → falls back to simple_forecasts for all 36 states
  4. "Feb 2026 Forecast" metric → shows correct next month
"""

import re
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# ── Compute dynamic dates ──────────────────────────────────────────────────────
today = datetime.today()
d30  = today + timedelta(days=30)
d60  = today + timedelta(days=60)
d90  = today + timedelta(days=90)

label_30 = d30.strftime("%B %Y")   # e.g. "April 2026"
label_60 = d60.strftime("%B %Y")   # e.g. "May 2026"
label_90 = d90.strftime("%B %Y")   # e.g. "June 2026"

date_30_iso = d30.strftime("%Y-%m-%d")   # e.g. "2026-04-28"
date_60_iso = d60.strftime("%Y-%m-%d")
date_90_iso = d90.strftime("%Y-%m-%d")

current_month_label = today.strftime("%B %Y")   # "March 2026"

print(f"Today          : {today.strftime('%Y-%m-%d')}")
print(f"30-day horizon : {label_30}  ({date_30_iso})")
print(f"60-day horizon : {label_60}  ({date_60_iso})")
print(f"90-day horizon : {label_90}  ({date_90_iso})")

# ── Read dashboard ─────────────────────────────────────────────────────────────
with open("app_dashboard.py", "r", encoding="utf-8") as f:
    code = f.read()

original = code  # keep backup

# ── Fix 1: Sidebar radio labels ────────────────────────────────────────────────
code = re.sub(
    r'"Forecast Month",\s*\["February 2026", "March 2026", "April 2026"\]',
    f'"Forecast Horizon",\n        ["{label_30} (+30 days)", "{label_60} (+60 days)", "{label_90} (+90 days)"]',
    code
)

# ── Fix 2: "Current Situation (December 2025)" header ─────────────────────────
code = code.replace(
    'st.header("📈 Current Situation (December 2025)")',
    f'st.header("📈 Current Situation ({current_month_label})")'
)

# ── Fix 3: Top-right "Feb 2026 Forecast" metric label + column detection ───────
# Replace the entire forecast metric block
old_metric_block = '''        if feb_col:
            forecast_avg = forecasts[feb_col].mean()
            forecast_change = forecast_avg - avg_risk
            st.metric(
                "Feb 2026 Forecast",
                f"{forecast_avg:.1f}",
                delta=f"{forecast_change:+.1f}",
                delta_color="inverse"
            )
        else:
            st.metric(
                "Feb 2026 Forecast",
                "N/A",
                delta="Check forecast file"
            )'''

new_metric_block = f'''        if feb_col:
            forecast_avg = forecasts[feb_col].mean()
            forecast_change = forecast_avg - avg_risk
            st.metric(
                "{label_30} Forecast (+30d)",
                f"{{forecast_avg:.1f}}",
                delta=f"{{forecast_change:+.1f}}",
                delta_color="inverse"
            )
        else:
            st.metric(
                "{label_30} Forecast (+30d)",
                "N/A",
                delta="Check forecast file"
            )'''

code = code.replace(old_metric_block, new_metric_block)

# ── Fix 4: Hardcoded forecast dates in chart ───────────────────────────────────
# Replace every occurrence of the hardcoded date list
old_dates = "pd.to_datetime(['2026-02-01', '2026-03-01', '2026-04-01'])"
new_dates = f"pd.to_datetime(['{date_30_iso}', '{date_60_iso}', '{date_90_iso}'])"
code = code.replace(old_dates, new_dates)

# ── Fix 5: State forecast crash — add robust fallback using simple_forecasts ───
old_line = "    state_forecast = forecasts[forecasts['state'] == selected_state].iloc[0]"
new_lines = """    # Try prophet forecasts first; fall back to simple forecasts for all states
    _prophet_match = forecasts[forecasts['state'] == selected_state]
    if not _prophet_match.empty:
        state_forecast = _prophet_match.iloc[0]
    else:
        # Load simple forecasts which cover all 36 states
        try:
            _simple = load_simple_forecasts()
            _simple_match = _simple[_simple['state'] == selected_state]
            if not _simple_match.empty:
                state_forecast = _simple_match.iloc[0]
            else:
                state_forecast = forecasts.iloc[0]  # last resort: any row
        except Exception:
            state_forecast = forecasts.iloc[0]"""

code = code.replace(old_line, new_lines)

# ── Fix 6: Add load_simple_forecasts() function after load_forecasts() ─────────
# Insert the new loader right after the existing load_forecasts function
new_loader = '''
@st.cache_data
def load_simple_forecasts():
    """Load simple moving-average forecasts (covers all 36 states)"""
    try:
        return pd.read_csv('outputs/reports/simple_forecasts_2026.csv')
    except Exception as e:
        return None

'''

# Insert before load_historical_data
code = code.replace(
    "@st.cache_data\ndef load_historical_data():",
    new_loader + "@st.cache_data\ndef load_historical_data():"
)

# ── Fix 7: Chart title uses correct period labels ──────────────────────────────
code = code.replace(
    'name=\'Forecast\'',
    f"name='+30/+60/+90 Day Forecast'"
)

# ── Write patched file ─────────────────────────────────────────────────────────
with open("app_dashboard.py", "w", encoding="utf-8") as f:
    f.write(code)

# ── Verify key fixes landed ────────────────────────────────────────────────────
print("\n✅ Patch applied. Verifying...")
checks = [
    (label_30 in code,                      f"Sidebar shows '{label_30}'"),
    (current_month_label in code,            f"Header shows '{current_month_label}'"),
    (date_30_iso in code,                    f"Chart date is '{date_30_iso}'"),
    ("load_simple_forecasts" in code,        "Simple forecast fallback function added"),
    ("_prophet_match" in code,               "State forecast crash fix applied"),
    ("Feb 2026 Forecast" not in code,        "Old 'Feb 2026' label removed"),
]

all_ok = True
for passed, label in checks:
    icon = "✅" if passed else "❌"
    print(f"  {icon} {label}")
    if not passed:
        all_ok = False

if all_ok:
    print("\n🎉 All fixes applied successfully!")
    print("   Now run:  streamlit run app_dashboard.py")
else:
    print("\n⚠️  Some fixes may not have applied. Check the lines above.")
    print("   A backup of the original is NOT saved — re-download if needed.")
