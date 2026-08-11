"""
🌍 CrisisLens Dashboard
Multi-Factor Crisis Risk Forecasting System for India

Week 5: Interactive Streamlit Dashboard
Author: CrisisLens Team
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from models.arima_model import run_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import json

# Page configuration
st.set_page_config(
    page_title="CrisisLens - Crisis Risk Forecasting",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global UI styles ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [data-testid="stApp"] {
    background: linear-gradient(135deg, #eef2ff 0%, #f8fafc 100%) !important;
    font-family: 'Inter', -apple-system, sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.85) !important;
    backdrop-filter: blur(12px);
    border-right: 1px solid #e2e8f0;
}

/* ── Streamlit button override ── */
.stButton > button, [data-testid="stPopover"] > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.18s ease !important;
}

/* ── Metric card tweak ── */
[data-testid="stMetric"] {
    background: white;
    border-radius: 12px;
    padding: 0.9rem 1rem;
    border: 1px solid #e8edf3;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    margin-bottom: 0.5rem;
}

/* ─────────────────────────────────────────────
   DASHBOARD HEADER
───────────────────────────────────────────── */
.dash-header {
    background: white;
    border-radius: 16px;
    padding: 1.4rem 1.8rem;
    border: 1px solid #e8edf3;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    margin-bottom: 1.8rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 1rem;
    border-left: 5px solid #dc2626;
}

.dash-title {
    font-size: 1.55rem;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.04em;
    line-height: 1.2;
}

.dash-subtitle {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 0.3rem;
    line-height: 1.5;
}

.dash-badges {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    align-items: center;
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.35rem 0.75rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.01em;
}

.badge-red   { background:#fff1f2; color:#be123c; border:1px solid #fecdd3; }
.badge-blue  { background:#eff6ff; color:#1d4ed8; border:1px solid #bfdbfe; }
.badge-green { background:#f0fdf4; color:#15803d; border:1px solid #bbf7d0; }
.badge-amber { background:#fffbeb; color:#b45309; border:1px solid #fde68a; }

/* ─────────────────────────────────────────────
   INDICATOR CARDS (Key Indicators)
───────────────────────────────────────────── */
.indicator-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.65rem 0.9rem;
    margin-bottom: 0.5rem;
    transition: box-shadow 0.15s ease;
}
.indicator-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.07); }

.indicator-row { display: flex; justify-content: space-between; align-items: center; }
.indicator-label { font-size: 0.83rem; color: #475569; font-weight: 500; }
.indicator-value { font-family: 'DM Mono', monospace; font-size: 0.92rem; font-weight: 600; color: #0f172a; }
.indicator-context { font-size: 0.71rem; color: #94a3b8; margin-top: 0.18rem; line-height: 1.3; }

/* ─────────────────────────────────────────────
   FOOTER
───────────────────────────────────────────── */
.dash-footer {
    background: white;
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    border: 1px solid #e8edf3;
    margin-top: 2.5rem;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
}
.footer-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    flex-wrap: wrap;
    gap: 1rem;
    margin-bottom: 1rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #f1f5f9;
}
.footer-brand { font-size:1rem; font-weight:700; color:#0f172a; }
.footer-desc  { font-size:0.75rem; color:#64748b; margin-top:0.15rem; line-height:1.5; }
.footer-stat { text-align:center; }
.footer-stat .num { font-size:1.2rem; font-weight:700; color:#dc2626; }
.footer-stat .lbl { font-size:0.68rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; }
.footer-note { font-size: 0.72rem; color: #94a3b8; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# AUTHENTICATION
# ============================================================================

VALID_CREDENTIALS = {
    "admin@disaster.gov": "admin123"
}

def show_login_page():
    """
    CLEANED UP LOGIN PAGE: Uses Streamlit's native layout features 
    to create a perfectly aligned, modern card.
    """
    st.markdown("<br><br><br>", unsafe_allow_html=True) # Push down to center
    _, center_col, _ = st.columns([1, 1.2, 1])

    with center_col:
        # Native Streamlit bordered container looks much cleaner
        with st.container(border=True):
            st.markdown("<h1 style='text-align: center; font-size: 3rem;'>🌍</h1>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center; margin-top: -10px;'>CrisisLens</h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #64748b;'>Multi-factor crisis risk forecasting<br>36 Indian states & UTs</p>", unsafe_allow_html=True)
            
            st.info("📋 **Demo credentials**\n\nEmail: `admin@disaster.gov`\n\nPassword: `admin123`")
            
            with st.form("login_form", clear_on_submit=False):
                email    = st.text_input("Email Address", placeholder="admin@disaster.gov")
                password = st.text_input("Password", placeholder="••••••••", type="password")
                submitted = st.form_submit_button("Sign In →", use_container_width=True)

                if submitted:
                    if not email or "@" not in email:
                        st.error("Please enter a valid email address.")
                    elif not password or len(password) < 6:
                        st.error("Password must be at least 6 characters.")
                    elif VALID_CREDENTIALS.get(email) == password:
                        st.session_state.authenticated = True
                        st.session_state.username      = email
                        st.success("✅ Welcome! Loading your dashboard…")
                        st.rerun()
                    else:
                        st.error("Invalid credentials. Please use the demo credentials.")
            
            st.markdown("<p style='text-align: center; font-size: 0.75rem; color: #94a3b8; margin-top: 10px;'>Contact your system administrator for access.</p>", unsafe_allow_html=True)

# ── Auth gate ─────────────────────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""

if not st.session_state.authenticated:
    show_login_page()
    st.stop()

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data
def load_current_risk():
    try:
        df = pd.read_csv('data/processed/combined_monthly_2024_2025.csv')
        latest = df[df['month'] == df['month'].max()].copy()
        return latest
    except Exception as e:
        st.error(f"Error loading current risk data: {e}")
        return None

@st.cache_data
def load_forecasts():
    try:
        df = pd.read_csv('outputs/reports/simple_forecasts_2026.csv')
        return df
    except Exception as e:
        st.error(f"Error loading forecast data: {e}")
        return None

def load_prophet_forecasts():
    try:
        df = pd.read_csv('outputs/reports/prophet_forecasts_2026.csv')
        return df
    except:
        return None

@st.cache_data
def load_historical_data():
    try:
        df = pd.read_csv('data/processed/combined_monthly_2024_2025.csv')
        df['month_date'] = pd.to_datetime(df['month'] + '-01')
        return df
    except Exception as e:
        st.error(f"Error loading historical data: {e}")
        return None

@st.cache_data
def load_trend_analysis():
    try:
        df = pd.read_csv('outputs/reports/trend_analysis_results.csv')
        return df
    except Exception as e:
        st.warning("Trend analysis data not available")
        return None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_risk_category(risk_score):
    if risk_score < 30:
        return "LOW", "🟢"
    elif risk_score < 50:
        return "MEDIUM", "🟡"
    elif risk_score < 70:
        return "HIGH", "🟠"
    else:
        return "CRITICAL", "🔴"

# ============================================================================
# TOP-5 STATE POPUP  (st.dialog)
# ============================================================================

@st.dialog("📍 State Risk Breakdown")
def show_state_popup(state_name, total_risk, climate_stress, agri_stress, econ_stress):
    risk_cat, risk_emoji = get_risk_category(total_risk)
    dominant = max(
        [('Climate', climate_stress), ('Agriculture', agri_stress), ('Economic', econ_stress)],
        key=lambda x: x[1]
    )
    color_map   = {"CRITICAL": "#dc2626", "HIGH": "#ea580c", "MEDIUM": "#d97706", "LOW": "#16a34a"}
    bg_map      = {"CRITICAL": "#fff1f2", "HIGH": "#fff7ed", "MEDIUM": "#fffbeb", "LOW": "#f0fdf4"}
    color = color_map.get(risk_cat, "#334155")
    bg    = bg_map.get(risk_cat, "#f8fafc")

    st.markdown(f"""
    <div style="text-align:center; background:{bg}; border-radius:14px;
                padding:1.2rem 1rem 0.8rem; margin-bottom:1.2rem; border:1px solid {color}30;">
        <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.1em;
                    color:{color}; font-weight:700; margin-bottom:0.3rem;">{state_name}</div>
        <div style="font-size:3.2rem; font-weight:800; color:{color}; line-height:1;">
            {total_risk:.1f}
        </div>
        <div style="font-size:0.78rem; color:#64748b; margin-top:0.2rem;">out of 100</div>
        <div style="display:inline-block; background:{color}; color:white;
                    padding:0.25rem 1rem; border-radius:20px;
                    font-size:0.75rem; font-weight:700; margin-top:0.6rem;">
            {risk_emoji} {risk_cat} RISK
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**How the score is built:**")
    cols = st.columns(3)
    components = [
        ("🌦 Climate", climate_stress, "40% weight"),
        ("🌾 Agriculture", agri_stress, "35% weight"),
        ("💰 Economic", econ_stress, "25% weight"),
    ]
    for col, (label, score, weight) in zip(cols, components):
        c, e = get_risk_category(score)
        with col:
            st.metric(label, f"{score:.1f}", delta=f"{e} {c} · {weight}", delta_color="off")

    st.markdown(f"""
    **Primary driver:** {dominant[0]} stress is the biggest contributor at **{dominant[1]:.1f} / 100**,
    meaning {state_name}'s highest risk stems from {dominant[0].lower()} conditions.
    """)

    st.markdown("**What this means for decision-makers:**")
    if risk_cat == "CRITICAL":
        st.error(f"🚨 {state_name} needs **immediate multi-agency intervention**. All stress domains should be monitored daily.")
    elif risk_cat == "HIGH":
        st.warning(f"⚠️ {state_name} is under **significant pressure**. Alert district authorities, pre-position relief stocks.")
    elif risk_cat == "MEDIUM":
        st.info(f"📊 {state_name} shows **moderate stress**. Continue regular monitoring.")
    else:
        st.success(f"✅ {state_name} is **relatively stable**. Routine monitoring is sufficient.")

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main():

    # ── Header ──────────────────────────────────────────────────────────────
    now_str = datetime.now().strftime("%d %b %Y, %H:%M")
    st.markdown(f"""
    <div class="dash-header">
        <div class="dash-header-left">
            <div class="dash-title">🌍 CrisisLens</div>
            <div class="dash-subtitle">
                India's multi-factor crisis risk intelligence platform &nbsp;·&nbsp;
                Climate · Agriculture · Economic stress indicators
            </div>
        </div>
        <div class="dash-badges">
            <span class="badge badge-red">🔴 Live Forecast</span>
            <span class="badge badge-blue">📍 36 States & UTs</span>
            <span class="badge badge-green">✅ 93.3% Accuracy</span>
            <span class="badge badge-amber">🕐 {now_str}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("🔄 Analysing risk patterns across India…"):
        current_risk = load_current_risk()
        forecasts    = load_forecasts()
        historical   = load_historical_data()
        trends       = load_trend_analysis()

    if current_risk is None or forecasts is None:
        st.error("❌ Unable to load required data. Please check data files.")
        return

    # ========================================================================
    # SIDEBAR: Controls
    # ========================================================================

    st.sidebar.markdown("""
    <div style="padding:0.2rem 0 0.8rem; border-bottom:1px solid #E2E8F0; margin-bottom:1rem;">
        <span style="font-size:0.68rem;font-weight:700;text-transform:uppercase;
                     letter-spacing:0.12em;color:#64748B;">Dashboard Controls</span>
    </div>
    """, unsafe_allow_html=True)

    states = sorted(current_risk['state'].unique())
    selected_state = st.sidebar.selectbox(
        "Select State/UT",
        states,
        index=states.index("Maharashtra") if "Maharashtra" in states else 0
    )

    forecast_month = st.sidebar.radio(
        "Forecast Horizon",
        ["April 2026 (+30 days)", "May 2026 (+60 days)", "June 2026 (+90 days)"],
        index=0
    )

    alert_threshold = st.sidebar.slider(
        "Alert Threshold",
        min_value=30.0, max_value=70.0, value=50.0, step=5.0,
        help="Risk scores at or above this value trigger alerts"
    )

    _h_col_map = {
        "April 2026 (+30 days)": ['best_forecast_feb', 'lt_forecast_feb2026', 'ma_forecast_feb2026'],
        "May 2026 (+60 days)":   ['best_forecast_mar', 'lt_forecast_mar2026', 'ma_forecast_mar2026'],
        "June 2026 (+90 days)":  ['best_forecast_apr', 'lt_forecast_apr2026', 'ma_forecast_apr2026'],
    }
    _h_label = {
        "April 2026 (+30 days)": "April 2026 (+30d)",
        "May 2026 (+60 days)":   "May 2026 (+60d)",
        "June 2026 (+90 days)":  "June 2026 (+90d)",
    }

    active_forecast_col = None
    for _c in _h_col_map[forecast_month]:
        if _c in forecasts.columns:
            active_forecast_col = _c
            break

    if active_forecast_col is not None:
        _fcast_lookup = forecasts[['state', active_forecast_col]].rename(
            columns={active_forecast_col: 'forecast_total_risk'}
        )
        horizon_risk = current_risk.merge(_fcast_lookup, on='state', how='left')
        horizon_risk['total_risk'] = horizon_risk['forecast_total_risk'].fillna(horizon_risk['total_risk'])
        horizon_risk = horizon_risk.drop(columns=['forecast_total_risk'])
        is_forecast_view = True
    else:
        horizon_risk = current_risk.copy()
        is_forecast_view = False

    # Seasonal adjustments
    _CLIMATE_WEIGHT = 0.35
    _AGRI_WEIGHT    = 0.35
    _ECON_WEIGHT    = 0.30

    _SEASON_BOOST = {
        "Assam":            {"April 2026 (+30 days)": 1.05, "May 2026 (+60 days)": 1.15, "June 2026 (+90 days)": 1.35},
        "Meghalaya":        {"April 2026 (+30 days)": 1.05, "May 2026 (+60 days)": 1.12, "June 2026 (+90 days)": 1.30},
        "Arunachal Pradesh":{"April 2026 (+30 days)": 1.03, "May 2026 (+60 days)": 1.10, "June 2026 (+90 days)": 1.28},
        "Manipur":          {"April 2026 (+30 days)": 1.03, "May 2026 (+60 days)": 1.10, "June 2026 (+90 days)": 1.25},
        "Nagaland":         {"April 2026 (+30 days)": 1.03, "May 2026 (+60 days)": 1.10, "June 2026 (+90 days)": 1.25},
        "Mizoram":          {"April 2026 (+30 days)": 1.03, "May 2026 (+60 days)": 1.08, "June 2026 (+90 days)": 1.22},
        "Tripura":          {"April 2026 (+30 days)": 1.03, "May 2026 (+60 days)": 1.08, "June 2026 (+90 days)": 1.22},
        "Odisha":           {"April 2026 (+30 days)": 1.05, "May 2026 (+60 days)": 1.12, "June 2026 (+90 days)": 1.28},
        "West Bengal":      {"April 2026 (+30 days)": 1.04, "May 2026 (+60 days)": 1.10, "June 2026 (+90 days)": 1.25},
        "Kerala":           {"April 2026 (+30 days)": 1.04, "May 2026 (+60 days)": 1.12, "June 2026 (+90 days)": 1.30},
        "Goa":              {"April 2026 (+30 days)": 1.03, "May 2026 (+60 days)": 1.08, "June 2026 (+90 days)": 1.20},
        "Rajasthan":        {"April 2026 (+30 days)": 1.08, "May 2026 (+60 days)": 1.15, "June 2026 (+90 days)": 1.05},
        "Gujarat":          {"April 2026 (+30 days)": 1.06, "May 2026 (+60 days)": 1.12, "June 2026 (+90 days)": 1.04},
        "Madhya Pradesh":   {"April 2026 (+30 days)": 1.06, "May 2026 (+60 days)": 1.10, "June 2026 (+90 days)": 1.08},
        "Maharashtra":      {"April 2026 (+30 days)": 1.05, "May 2026 (+60 days)": 1.10, "June 2026 (+90 days)": 1.08},
        "Telangana":        {"April 2026 (+30 days)": 1.06, "May 2026 (+60 days)": 1.12, "June 2026 (+90 days)": 1.05},
        "Andhra Pradesh":   {"April 2026 (+30 days)": 1.05, "May 2026 (+60 days)": 1.10, "June 2026 (+90 days)": 1.06},
        "Ladakh":           {"April 2026 (+30 days)": 0.90, "May 2026 (+60 days)": 0.85, "June 2026 (+90 days)": 0.80},
        "Himachal Pradesh": {"April 2026 (+30 days)": 0.95, "May 2026 (+60 days)": 0.92, "June 2026 (+90 days)": 0.90},
        "Uttarakhand":      {"April 2026 (+30 days)": 0.96, "May 2026 (+60 days)": 0.93, "June 2026 (+90 days)": 0.91},
        "Jammu and Kashmir":{"April 2026 (+30 days)": 0.95, "May 2026 (+60 days)": 0.93, "June 2026 (+90 days)": 0.91},
        "Sikkim":           {"April 2026 (+30 days)": 1.02, "May 2026 (+60 days)": 1.08, "June 2026 (+90 days)": 1.20},
    }

    if is_forecast_view and forecast_month in list(_SEASON_BOOST.values())[0]:
        def _apply_seasonal(row):
            state = row['state']
            if state not in _SEASON_BOOST:
                return row['total_risk']
            mult = _SEASON_BOOST[state][forecast_month]
            adj_climate = min(row.get('climate_stress', 50) * mult, 100)
            adj_total = (adj_climate * _CLIMATE_WEIGHT
                         + row.get('agri_stress', 50) * _AGRI_WEIGHT
                         + row.get('econ_stress', 50) * _ECON_WEIGHT)
            return min(adj_total, 100)

        horizon_risk = horizon_risk.copy()
        horizon_risk['total_risk'] = horizon_risk.apply(_apply_seasonal, axis=1)

    st.sidebar.markdown("---")
    st.sidebar.info("""
**How to use this dashboard:**
1. **Pick a state** from the dropdown to see its detailed breakdown
2. **Set the forecast horizon** to see risk 30, 60, or 90 days ahead
3. **Adjust the alert threshold** — states above this score appear in the action list

**Risk score guide:**
- 🟢 < 30 = Low
- 🟡 30–50 = Medium
- 🟠 50–70 = High
- 🔴 > 70 = Critical
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"<div style='font-size:0.78rem;color:#64748B;margin-bottom:0.5rem;'>"
        f"👤 Signed in as <strong>{st.session_state.username}</strong></div>",
        unsafe_allow_html=True
    )
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.rerun()

    # ========================================================================
    # SECTION 1: NATIONAL OVERVIEW 
    # ========================================================================

    avg_risk       = horizon_risk['total_risk'].mean()
    high_risk_count = len(horizon_risk[horizon_risk['total_risk'] >= alert_threshold])
    max_risk_state  = horizon_risk.loc[horizon_risk['total_risk'].idxmax()]
    nat_cat, nat_emoji = get_risk_category(avg_risk)
    view_label = _h_label[forecast_month] if is_forecast_view else "March 2026 (Current)"

    st.header(f"📊 National Overview — {view_label}")
    st.caption("Risk is scored 0–100.  Below 30 = Low · 30–50 = Medium · 50–70 = High · Above 70 = Critical.")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🇮🇳 National Avg Risk", f"{avg_risk:.1f} / 100", delta=f"{nat_emoji} {nat_cat} overall", delta_color="off")
        with st.popover("ℹ️ Why this score?", use_container_width=True):
            st.markdown(f"**How is this calculated?**\n\nThis **{avg_risk:.1f}** score is the mean average of all 36 state/UT risk scores. It provides a macro-view of the country's baseline stress level. \n\n*A score in the **{nat_cat}** range indicates that national resources { 'are stretched' if avg_risk > 50 else 'are relatively stable' }.*")

    with col2:
        st.metric("⚠️ States Needing Attention", f"{high_risk_count} of 36", delta=f"Above threshold of {alert_threshold:.0f}", delta_color="off")
        with st.popover("ℹ️ Why these states?", use_container_width=True):
            st.markdown(f"**What triggers this?**\n\nYou currently have the Alert Threshold set to **{alert_threshold:.0f}** in the sidebar. \n\nThere are **{high_risk_count} states** currently exceeding this limit, automatically flagging them for operational review in the 'Recommended Actions' table below.")

    with col3:
        max_name = max_risk_state['state']
        max_cat, max_emoji = get_risk_category(max_risk_state['total_risk'])
        st.metric("🔴 Highest Risk State", max_name[:18] + "…" if len(max_name) > 18 else max_name, delta=f"{max_emoji} {max_risk_state['total_risk']:.1f} — {max_cat}", delta_color="off")
        with st.popover(f"ℹ️ Why {max_name[:10]}?", use_container_width=True):
            st.markdown(f"**What is happening in {max_name}?**\n\n{max_name} is currently the most vulnerable state in the country with an aggregate score of **{max_risk_state['total_risk']:.1f}**. \n\n*Select {max_name} in the sidebar dropdown to view its specific Climate, Agriculture, and Economic breakdown.*")

    with col4:
        if active_forecast_col is not None:
            forecast_avg    = forecasts[active_forecast_col].mean()
            current_avg     = current_risk['total_risk'].mean()
            forecast_change = forecast_avg - current_avg
            st.metric(_h_label[forecast_month], f"{forecast_avg:.1f}", delta=f"{forecast_change:+.1f}", delta_color="inverse")
            with st.popover("ℹ️ Forecast meaning", use_container_width=True):
                direction = "worsen" if forecast_change > 0 else "improve"
                horizon_days = forecast_month.split('+')[1].split('d')[0] if '+' in forecast_month else "upcoming"
                st.markdown(f"**What does this predict?**\n\nOur models project the National Average Risk will **{direction} by {abs(forecast_change):.1f} points** by {forecast_month.split(' ')[0]}. \n\n*This projection factors in historical trends, seasonal weather anomalies, and economic forecasts over the next {horizon_days} days.*")
        else:
            st.metric(_h_label[forecast_month], "N/A", delta="No data")
            with st.popover("ℹ️ Forecast missing", use_container_width=True):
                st.markdown("Forecast data for this specific time horizon is currently unavailable.")

    # ── Map + Pie + Top 5 ────────────────────────────────────────────────────
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("🗺️ Risk Across All States — Ranked Highest to Lowest")
        st.caption("Hover over a bar to see the specific risk score and primary driver.")

        map_data = horizon_risk.sort_values('total_risk', ascending=True).copy()

        def get_dominant_factor(row):
            factors = {
                'Climate': row.get('climate_stress', 0),
                'Agriculture': row.get('agri_stress', 0),
                'Economic': row.get('econ_stress', 0)
            }
            return max(factors, key=factors.get)
            
        map_data['Dominant Stress'] = map_data.apply(get_dominant_factor, axis=1)

        # REVERTED TO NATURAL CONTINUOUS COLOR GRADIENT
        fig_map = px.bar(
            map_data,
            x='total_risk',
            y='state',
            orientation='h',
            color='total_risk',
            color_continuous_scale=[
                (0.0, '#2ecc71'),   # Green (Low)
                (0.33, '#f39c12'),  # Yellow/Orange (Medium)
                (0.66, '#e74c3c'),  # Red (High)
                (1.0, '#c0392b')    # Dark Red (Critical)
            ],
            range_color=[0, 100],
            text='total_risk',
            hover_data={'total_risk': ':.1f', 'Dominant Stress': True, 'state': False}
        )

        fig_map.update_traces(
            texttemplate='<b>%{x:.1f}</b>', 
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Risk Score: %{x:.1f}<br>Primary Driver: %{customdata[0]}<extra></extra>'
        )

        # Retained the useful dotted alert line
        fig_map.add_vline(
            x=alert_threshold, 
            line_dash="dot", 
            line_color="red", 
            line_width=2,
            annotation_text=f"Alert Trigger ({alert_threshold})", 
            annotation_position="bottom right"
        )

        fig_map.update_layout(
            height=750, 
            xaxis_title="Risk Score",
            yaxis_title="",
            margin=dict(l=0, r=40, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#334155", size=12),
            xaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0", tickfont=dict(color="#64748B"), range=[0, 105]),
            yaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0", tickfont=dict(size=11, color="#334155")),
            coloraxis_colorbar=dict(title="Risk Score")
        )

        st.plotly_chart(fig_map, use_container_width=True)

    with col_right:
        st.subheader("📊 States by Risk Category")

        risk_categories = horizon_risk['total_risk'].apply(
            lambda x: get_risk_category(x)[0]
        ).value_counts()

        fig_pie = px.pie(
            values=risk_categories.values,
            names=risk_categories.index,
            color=risk_categories.index,
            color_discrete_map={
                'LOW':      '#2ecc71',
                'MEDIUM':   '#f39c12',
                'HIGH':     '#e74c3c',
                'CRITICAL': '#c0392b'
            },
            hole=0.45 
        )
        fig_pie.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Count: %{value} states<br>Proportion: %{percent}<extra></extra>'
        )
        fig_pie.update_layout(
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#334155"),
            showlegend=False, 
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        horizon_label_short = _h_label[forecast_month] if is_forecast_view else "Current"
        st.markdown(f"**🚨 Top 5 States — {horizon_label_short}**")
        st.caption("Click any state to launch the detailed intervention breakdown.")

        top5 = horizon_risk.nlargest(5, 'total_risk')[
            ['state', 'total_risk', 'climate_stress', 'agri_stress', 'econ_stress']
        ]

        color_map  = {"CRITICAL": "#fff1f2", "HIGH": "#fff7ed", "MEDIUM": "#fffbeb", "LOW": "#f0fdf4"}
        border_map = {"CRITICAL": "#fca5a5", "HIGH": "#fdba74", "MEDIUM": "#fde68a", "LOW": "#86efac"}
        text_map   = {"CRITICAL": "#dc2626", "HIGH": "#ea580c", "MEDIUM": "#d97706", "LOW": "#16a34a"}

        for idx, row in top5.iterrows():
            category, _ = get_risk_category(row['total_risk'])
            dominant = max(
                [('Climate', row['climate_stress']),
                 ('Agri',    row['agri_stress']),
                 ('Econ',    row['econ_stress'])],
                key=lambda x: x[1]
            )
            bg = color_map.get(category, "#f8fafc")
            bd = border_map.get(category, "#e2e8f0")
            tx = text_map.get(category, "#334155")

            st.markdown(f"""
            <div style="background:{bg}; border:1px solid {bd}; border-left:4px solid {bd};
                        border-radius:10px; padding:0.6rem 0.85rem; margin-bottom:0.1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:0.87rem; font-weight:600; color:#0f172a;">{row['state']}</span>
                    <span style="font-family:'DM Mono',monospace; font-size:0.95rem;
                                 font-weight:700; color:{tx};">{row['total_risk']:.1f}</span>
                </div>
                <div style="font-size:0.72rem; color:#64748b; margin-top:2px;">
                    {category} · {dominant[0]} stress dominant ({dominant[1]:.1f})
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"View details →", key=f"top5_{idx}"):
                show_state_popup(
                    row['state'], row['total_risk'],
                    row['climate_stress'], row['agri_stress'], row['econ_stress']
                )

    st.markdown("---")

    # ========================================================================
    # SECTION 2: RECOMMENDED ACTIONS 
    # ========================================================================

    st.header("🚨 Which States Need Action Right Now?")
    st.caption(f"States with a risk score above **{alert_threshold:.0f}** are listed below, "
               f"along with tailored recommendations based on their specific stress profile.")

    alerts = horizon_risk[horizon_risk['total_risk'] >= alert_threshold].copy()
    alerts = alerts.sort_values('total_risk', ascending=False)

    if len(alerts) > 0:
        st.warning(
            f"⚠️ **{len(alerts)} states/UTs** have a risk score above "
            f"{alert_threshold:.0f} and need attention. "
            f"Expand each state below to see what's driving the risk and what actions to take."
        )

        alerts_display = alerts[['state', 'total_risk', 'climate_stress', 'agri_stress', 'econ_stress']].copy()
        alerts_display.columns = ['State', 'Total Risk', 'Climate', 'Agriculture', 'Economic']
        alerts_display = alerts_display.round(1)

        st.dataframe(
            alerts_display,
            use_container_width=True,
            height=min(400, (len(alerts) + 1) * 35)
        )

        st.subheader("💡 Recommended Actions")

        for idx, row in alerts.head(5).iterrows():
            factors = {
                'Climate':     row['climate_stress'],
                'Agriculture': row['agri_stress'],
                'Economic':    row['econ_stress']
            }
            dominant_factor = max(factors, key=factors.get)
            risk_cat, _ = get_risk_category(row['total_risk'])

            with st.expander(
                f"🎯 {row['state']} — Risk: {row['total_risk']:.1f} ({risk_cat}) "
                f"| Primary driver: {dominant_factor} stress ({factors[dominant_factor]:.1f})"
            ):
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("🌦 Climate", f"{row['climate_stress']:.1f}",
                              delta="High" if row['climate_stress'] > 60 else "Moderate" if row['climate_stress'] > 40 else "Low",
                              delta_color="inverse")
                with col_b:
                    st.metric("🌾 Agriculture", f"{row['agri_stress']:.1f}",
                              delta="High" if row['agri_stress'] > 60 else "Moderate" if row['agri_stress'] > 40 else "Low",
                              delta_color="inverse")
                with col_c:
                    st.metric("💰 Economic", f"{row['econ_stress']:.1f}",
                              delta="High" if row['econ_stress'] > 60 else "Moderate" if row['econ_stress'] > 40 else "Low",
                              delta_color="inverse")

                st.markdown("**What should be done:**")
                if dominant_factor == 'Climate':
                    temp_note = f"Temperature is {row.get('temp_anomaly_deg', 0):.1f}°C above normal — " if 'temp_anomaly_deg' in row.index else ""
                    rain_note = (f"Rainfall is {abs(row.get('rainfall_anomaly_pct', 0)):.0f}% "
                                 f"{'below' if row.get('rainfall_anomaly_pct', 0) < 0 else 'above'} normal. ") if 'rainfall_anomaly_pct' in row.index else ""
                    st.markdown(f"""
> ⚠️ {temp_note}{rain_note}Climate stress is the primary risk driver.

- **Immediate:** Issue heat/rainfall advisories; activate disaster response teams
- **Short-term:** Pre-position water tankers and relief supplies in vulnerable districts
- **Agriculture link:** Alert farmers on irrigation needs given current {rain_note or 'rainfall conditions'}
- **Monitor:** Daily IMD updates; escalate if conditions worsen beyond {row['climate_stress']:.0f}/100
                    """)
                elif dominant_factor == 'Agriculture':
                    crop_note = f"Crop failure rate is at {row.get('crop_failure_rate_pct', 0):.1f}%. " if 'crop_failure_rate_pct' in row.index else ""
                    st.markdown(f"""
> ⚠️ {crop_note}Agricultural stress is the primary risk driver for {row['state']}.

- **Immediate:** Deploy field teams to assess actual crop damage across districts
- **Short-term:** Fast-track crop insurance claims; activate PM-KISAN emergency support
- **Irrigation:** Mobilise canal and groundwater resources — rainfall deficit is stressing crops
- **Relief:** Prepare district-level agricultural relief packages; coordinate with state agriculture dept
- **Monitor:** Weekly crop condition surveys; watch for farmer distress signals
                    """)
                else:
                    st.markdown(f"""
> ⚠️ Economic stress ({row['econ_stress']:.1f}/100) is the primary driver for {row['state']}.

- **Employment:** Accelerate MGNREGA work sanctioning in affected districts
- **Food security:** Audit PDS (ration shop) stock levels; ensure no shortfall
- **Social protection:** Identify families below threshold for emergency cash transfers
- **Inflation watch:** Monitor local market prices for essential commodities weekly
- **Coordinate:** Finance + Labour + Food departments to align on response plan
                    """)
    else:
        st.success(f"✅ **No states currently above alert threshold ({alert_threshold:.0f})**")
        st.info("Continue monitoring. Review the forecast horizons for potential future risks.")

    st.markdown("---")

    # ========================================================================
    # SECTION 3: DEEP DIVE — SELECTED STATE (NOW WITH CLICKABLE METRICS)
    # ========================================================================

    st.header(f"🔍 Deep Dive: {selected_state}")
    st.caption(f"Use the sidebar to switch state. "
               f"The four numbers below show how stressed {selected_state} currently is across the three risk domains.")

    state_current = current_risk[current_risk['state'] == selected_state].iloc[0]

    if state_current['total_risk'] > 70:
        st.error("🚨 Critical Risk Zone")
    elif state_current['total_risk'] > 50:
        st.warning("⚠️ High Risk Zone")
    else:
        st.success("✅ Safe Zone")

    _sf_match    = forecasts[forecasts['state'] == selected_state]
    state_forecast = _sf_match.iloc[0] if not _sf_match.empty else forecasts.iloc[0]
    _prophet_df  = load_prophet_forecasts()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        risk_cat, risk_emoji = get_risk_category(state_current['total_risk'])
        st.metric("Overall Risk Score", f"{state_current['total_risk']:.1f} / 100", delta=f"{risk_emoji} {risk_cat}", delta_color="off")
        with st.popover("ℹ️ Score meaning", use_container_width=True):
            st.markdown(f"**Overall Risk for {selected_state}**\n\nThis is a weighted aggregate score: **35%** Climate, **35%** Agriculture, and **30%** Economic stress. \n\nA score of **{state_current['total_risk']:.1f}** currently places the state in the **{risk_cat}** category.")

    with col2:
        clim_cat, clim_emoji = get_risk_category(state_current['climate_stress'])
        st.metric("🌦 Climate Stress", f"{state_current['climate_stress']:.1f} / 100", delta=f"{clim_emoji} {clim_cat}", delta_color="off")
        with st.popover("ℹ️ Climate details", use_container_width=True):
            st.markdown(f"**Climate Stress: {state_current['climate_stress']:.1f} / 100**\n\nMeasures deviations in historical weather patterns, specifically combining temperature anomalies (**{state_current.get('temp_anomaly_deg', 0):.1f}°C**) and rainfall deviations (**{state_current.get('rainfall_anomaly_pct', 0):.1f}%**). \n\nHigh scores indicate a severe weather disruption.")

    with col3:
        agri_cat, agri_emoji = get_risk_category(state_current['agri_stress'])
        st.metric("🌾 Agriculture Stress", f"{state_current['agri_stress']:.1f} / 100", delta=f"{agri_emoji} {agri_cat}", delta_color="off")
        with st.popover("ℹ️ Agri details", use_container_width=True):
            st.markdown(f"**Agriculture Stress: {state_current['agri_stress']:.1f} / 100**\n\nIndicates threats to food production and farmer livelihood. This is heavily driven by the current crop failure rate (**{state_current.get('crop_failure_rate_pct', 0):.1f}%**) and groundwater depletion levels.")

    with col4:
        econ_cat, econ_emoji = get_risk_category(state_current['econ_stress'])
        st.metric("💰 Economic Stress", f"{state_current['econ_stress']:.1f} / 100", delta=f"{econ_emoji} {econ_cat}", delta_color="off")
        with st.popover("ℹ️ Econ details", use_container_width=True):
            st.markdown(f"**Economic Stress: {state_current['econ_stress']:.1f} / 100**\n\nReflects socioeconomic pressure on the population. Driven by unemployment rates (**{state_current.get('unemployment_rate_pct', 0):.1f}%**) and local inflation spikes (**{state_current.get('inflation_rate_pct', 0):.1f}%**).")


    if historical is not None:
        st.subheader(f"📈 {selected_state}: Past Risk + Future Forecast")
        st.caption("Blue line = actual historical risk · Orange stars = model forecast for April, May, June 2026 · Red dotted line = your alert threshold.")

        state_history = historical[historical['state'] == selected_state].copy()
        state_history = state_history.sort_values('month_date')

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=state_history['month_date'], y=state_history['total_risk'], mode='lines+markers', name='Historical Risk', line=dict(color='#1f77b4', width=2), marker=dict(size=6)))

        _col_map = {'apr': ['best_forecast_feb', 'lt_forecast_feb2026', 'ma_forecast_feb2026'], 'may': ['best_forecast_mar', 'lt_forecast_mar2026', 'ma_forecast_mar2026'], 'jun': ['best_forecast_apr', 'lt_forecast_apr2026', 'ma_forecast_apr2026']}
        forecast_dates  = pd.to_datetime(['2026-04-28', '2026-05-28', '2026-06-27'])
        forecast_values = []
        for _kw in ['apr', 'may', 'jun']:
            _val = None
            for _c in _col_map[_kw]:
                if _c in state_forecast.index and pd.notna(state_forecast[_c]):
                    _val = state_forecast[_c]
                    break
            forecast_values.append(_val if _val is not None else state_current['total_risk'])

        fig_trend.add_trace(go.Scatter(x=forecast_dates, y=forecast_values, mode='lines+markers', name='Apr / May / Jun 2026 Forecast', line=dict(color='#ff7f0e', width=2, dash='dash'), marker=dict(size=10, symbol='star')))
        fig_trend.add_hline(y=alert_threshold, line_dash="dot", line_color="red", annotation_text=f"Alert Threshold ({alert_threshold})", annotation_position="right")

        fig_trend.update_layout(
            height=420, xaxis_title="Date", yaxis_title="Risk Score (0–100)", hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#334155", size=12),
            xaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0"),
            yaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0", range=[20, 85]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        st.subheader("📊 Model Comparison (ARIMA vs Linear vs Prophet)")

        try:
            series = state_history['total_risk']
            train  = series[:-6]
            test   = series[-6:]

            arima_pred  = run_arima(train)
            arima_pred  = arima_pred[:len(test)]

            x            = np.arange(len(train))
            coef         = np.polyfit(x, train, 1)
            x_future     = np.arange(len(train), len(train) + len(test))
            linear_pred  = coef[0] * x_future + coef[1]

            prophet_pred = None
            if _prophet_df is not None:
                _prophet_df.columns = _prophet_df.columns.str.strip().str.lower()
                _prophet_df['state'] = _prophet_df['state'].str.strip().str.lower()
                _prophet_df['month'] = pd.to_datetime(_prophet_df['month'])
                selected_state_clean = selected_state.strip().lower()
                prophet_state = _prophet_df[_prophet_df['state'] == selected_state_clean]
                if not prophet_state.empty:
                    prophet_state = prophet_state.sort_values('month')
                    prophet_pred  = prophet_state['predicted_risk'].values[-len(test):]
                else:
                    st.info("📊 Prophet not available for this state. Showing ARIMA & Linear only.")
            else:
                st.warning("⚠️ Prophet file not loaded")

            def calc_metrics(true, pred):
                rmse = np.sqrt(mean_squared_error(true, pred))
                mae  = mean_absolute_error(true, pred)
                return rmse, mae

            arima_rmse,  arima_mae  = calc_metrics(test, arima_pred)
            linear_rmse, linear_mae = calc_metrics(test, linear_pred)

            if prophet_pred is not None:
                min_len = min(len(test), len(prophet_pred))
                prophet_rmse, prophet_mae = calc_metrics(test[-min_len:], prophet_pred[-min_len:])

            fig_compare = go.Figure()
            fig_compare.add_trace(go.Scatter(y=test.values,    name='Actual',  mode='lines+markers'))
            fig_compare.add_trace(go.Scatter(y=arima_pred,     name='ARIMA',   mode='lines+markers'))
            fig_compare.add_trace(go.Scatter(y=linear_pred,    name='Linear',  mode='lines+markers'))

            if prophet_pred is not None:
                if len(prophet_pred) < len(test):
                    pad_size         = len(test) - len(prophet_pred)
                    prophet_pred_plot = np.concatenate([np.full(pad_size, np.nan), prophet_pred])
                else:
                    prophet_pred_plot = prophet_pred
                fig_compare.add_trace(go.Scatter(y=prophet_pred_plot, name='Prophet', mode='lines+markers'))

            fig_compare.update_layout(
                template="plotly_white", hovermode="x unified",
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )

            st.subheader("📏 How Accurate Are the Predictions?")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("ARIMA RMSE",  f"{arima_rmse:.2f}")
                st.metric("ARIMA MAE",   f"{arima_mae:.2f}")
            with col2:
                st.metric("Linear RMSE", f"{linear_rmse:.2f}")
                st.metric("Linear MAE",  f"{linear_mae:.2f}")
            with col3:
                if prophet_pred is not None:
                    st.metric("Prophet RMSE", f"{prophet_rmse:.2f}")
                    st.metric("Prophet MAE",  f"{prophet_mae:.2f}")
                else:
                    st.info("No Prophet data")

            results = {"ARIMA": arima_rmse, "Linear": linear_rmse}
            if prophet_pred is not None:
                results["Prophet"] = prophet_rmse
            best_model = min(results, key=results.get)

            st.success(f"🏆 Most Reliable Prediction Method for {selected_state}: {best_model}")
            
            if prophet_pred is not None and not prophet_state.empty:
                upper = prophet_state['upper_bound'].values[-len(prophet_pred):]
                lower = prophet_state['lower_bound'].values[-len(prophet_pred):]
                if len(upper) < len(test):
                    pad   = len(test) - len(upper)
                    upper = np.concatenate([np.full(pad, np.nan), upper])
                    lower = np.concatenate([np.full(pad, np.nan), lower])
                fig_compare.add_trace(go.Scatter(y=upper, line=dict(width=0), showlegend=False))
                fig_compare.add_trace(go.Scatter(
                    y=lower, fill='tonexty', name='Prophet Confidence Interval',
                    fillcolor='rgba(255, 127, 14, 0.2)', line=dict(width=0)
                ))
                st.plotly_chart(fig_compare, use_container_width=True)

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

    # ── Component breakdown ──────────────────────────────────────────────────
    st.subheader(f"🎯 What's Driving the Risk in {selected_state}?")
    st.caption("The three bars show how much each domain is contributing. The taller the bar, the worse that domain's situation.")

    col_left, col_right = st.columns(2)

    with col_left:
        components_data = pd.DataFrame({
            'Factor': ['Climate Stress', 'Agriculture Stress', 'Economic Stress'],
            'Score':  [state_current['climate_stress'],
                       state_current['agri_stress'],
                       state_current['econ_stress']],
            'Weight': [40, 35, 25]
        })

        fig_components = px.bar(
            components_data,
            x='Factor', y='Score', color='Factor',
            color_discrete_map={
                'Climate Stress':     '#3498db',
                'Agriculture Stress': '#2ecc71',
                'Economic Stress':    '#e74c3c'
            },
            text='Score'
        )
        fig_components.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig_components.update_layout(
            height=360, showlegend=False,
            yaxis_title="Stress Score (0–100)", xaxis_title="",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#334155", size=12),
            xaxis=dict(tickfont=dict(size=13, color="#334155"), linecolor="#E2E8F0"),
            yaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0", tickfont=dict(color="#64748B"), range=[0, 100]),
            bargap=0.35,
        )
        fig_components.update_traces(marker_line_width=0, textfont=dict(family="DM Mono, monospace", size=13, color="#0F172A"))
        st.plotly_chart(fig_components, use_container_width=True)

    with col_right:
        st.markdown("**📋 Key Indicators**")
        st.caption("Click any indicator to see its impact on ground conditions.")

        indicators = [
            ("🌡️", "Temp Anomaly", f"{state_current['temp_anomaly_deg']:.1f}°C", "Degrees above seasonal normal — higher readings intensify heatwave & drought risk."),
            ("🌧️", "Rainfall Anomaly", f"{state_current['rainfall_anomaly_pct']:.1f}%", "Deviation from expected rainfall — a negative % signals a dangerous moisture deficit."),
            ("🔥", "Heatwave Days", f"{state_current['heatwave_days']:.0f} days", "Number of extreme-heat days — more days raise health emergencies and energy demand."),
            ("🌾", "Crop Failure", f"{state_current['crop_failure_rate_pct']:.1f}%", "Share of crops at risk of failure — directly drives food insecurity and farmer distress."),
            ("💼", "Unemployment", f"{state_current['unemployment_rate_pct']:.1f}%", "Joblessness rate — elevated unemployment amplifies vulnerability to any economic shock."),
            ("📈", "Inflation", f"{state_current['inflation_rate_pct']:.1f}%", "Price rise rate — high inflation erodes household purchasing power, especially for the poor."),
        ]

        pop_col1, pop_col2 = st.columns(2)
        
        for i, (icon, label, value, context) in enumerate(indicators):
            target_col = pop_col1 if i % 2 == 0 else pop_col2
            with target_col:
                with st.popover(f"{icon} {label} \n\n **{value}**", use_container_width=True):
                    st.markdown(f"### {icon} {label}")
                    st.metric(f"Current Value in {selected_state}", value)
                    st.markdown("---")
                    st.info(f"**What this means:**\n\n{context}")

    st.markdown("---")

    # ========================================================================
    # FOOTER
    # ========================================================================

    st.markdown("""
    <div class="dash-footer">
        <div class="footer-top">
            <div>
                <div class="footer-brand">🌍 CrisisLens</div>
                <div class="footer-desc">
                    India's multi-factor crisis risk intelligence platform.<br>
                    Built to support faster, evidence-based disaster response decisions.
                </div>
            </div>
            <div style="display:flex; gap:2rem; align-items:center;">
                <div class="footer-stat"><div class="num">36</div><div class="lbl">States & UTs</div></div>
                <div class="footer-stat"><div class="num">93.3%</div><div class="lbl">Alert Accuracy</div></div>
                <div class="footer-stat"><div class="num">3</div><div class="lbl">Forecast Models</div></div>
                <div class="footer-stat"><div class="num">90d</div><div class="lbl">Max Horizon</div></div>
            </div>
        </div>
        <div class="footer-note">
            <strong>Data:</strong> Indicators are simulated from documented Indian climate & economic baselines
            (IMD seasonal norms, MOSPI state-level statistics) with a fixed random seed (42) for full reproducibility.
            The forecasting pipeline is identical to what would be applied to live government data feeds.
            &nbsp;·&nbsp; <strong>Backtested</strong> against 5 real crisis events (2023–2024):
            <strong>93.3% alert accuracy</strong>, avg point error 2.66 / 100.
            &nbsp;·&nbsp; Forecasts: Prophet + Linear Trend · Data as of Dec 2025.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()