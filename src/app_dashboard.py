"""
🌍 CrisisLens Dashboard
Multi-Factor Crisis Risk Forecasting System for India

Interactive Streamlit Dashboard
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =====================================================================
# PROJECT PATH
# =====================================================================

FILE_PATH = Path(__file__).resolve()

# Expected structure:
#
# project/
#   dashboard/
#       app_dashboard.py
#   data/
#   outputs/
#   models/

ROOT_DIR = FILE_PATH.parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


try:
    from models.arima_model import run_arima
except ImportError:
    run_arima = None


# =====================================================================
# PAGE CONFIGURATION
# =====================================================================

st.set_page_config(
    page_title="CrisisLens - Crisis Risk Forecasting",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================================
# CONSTANTS
# =====================================================================

CURRENT_DATA_FILE = (
    ROOT_DIR
    / "data"
    / "processed"
    / "combined_monthly_2024_2025.csv"
)

REPORTS_DIR = (
    ROOT_DIR
    / "outputs"
    / "reports"
)

FORECAST_MODELS = [
    "best_forecast",
    "lt_forecast",
    "ma_forecast",
]

VALID_CREDENTIALS = {
    "admin@disaster.gov": "admin123"
}

CLIMATE_WEIGHT = 0.35
AGRI_WEIGHT = 0.35
ECON_WEIGHT = 0.30


# =====================================================================
# SEASONAL ADJUSTMENTS
# =====================================================================

SEASON_BOOST = {

    "Assam": {
        "April": 1.05,
        "May": 1.15,
        "June": 1.35,
    },

    "Meghalaya": {
        "April": 1.05,
        "May": 1.12,
        "June": 1.30,
    },

    "Arunachal Pradesh": {
        "April": 1.03,
        "May": 1.10,
        "June": 1.28,
    },

    "Manipur": {
        "April": 1.03,
        "May": 1.10,
        "June": 1.25,
    },

    "Nagaland": {
        "April": 1.03,
        "May": 1.10,
        "June": 1.25,
    },

    "Mizoram": {
        "April": 1.03,
        "May": 1.08,
        "June": 1.22,
    },

    "Tripura": {
        "April": 1.03,
        "May": 1.08,
        "June": 1.22,
    },

    "Odisha": {
        "April": 1.05,
        "May": 1.12,
        "June": 1.28,
    },

    "West Bengal": {
        "April": 1.04,
        "May": 1.10,
        "June": 1.25,
    },

    "Kerala": {
        "April": 1.04,
        "May": 1.12,
        "June": 1.30,
    },

    "Goa": {
        "April": 1.03,
        "May": 1.08,
        "June": 1.20,
    },

    "Rajasthan": {
        "April": 1.08,
        "May": 1.15,
        "June": 1.05,
    },

    "Gujarat": {
        "April": 1.06,
        "May": 1.12,
        "June": 1.04,
    },

    "Madhya Pradesh": {
        "April": 1.06,
        "May": 1.10,
        "June": 1.08,
    },

    "Maharashtra": {
        "April": 1.05,
        "May": 1.10,
        "June": 1.08,
    },

    "Telangana": {
        "April": 1.06,
        "May": 1.12,
        "June": 1.05,
    },

    "Andhra Pradesh": {
        "April": 1.05,
        "May": 1.10,
        "June": 1.06,
    },

    "Ladakh": {
        "April": 0.90,
        "May": 0.85,
        "June": 0.80,
    },

    "Himachal Pradesh": {
        "April": 0.95,
        "May": 0.92,
        "June": 0.90,
    },

    "Uttarakhand": {
        "April": 0.96,
        "May": 0.93,
        "June": 0.91,
    },

    "Jammu and Kashmir": {
        "April": 0.95,
        "May": 0.93,
        "June": 0.91,
    },

    "Sikkim": {
        "April": 1.02,
        "May": 1.08,
        "June": 1.20,
    },
}


# =====================================================================
# GLOBAL CSS
# =====================================================================

st.html(
    """
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap'
    );

    html, body, [data-testid="stApp"] {
        background: linear-gradient(
            135deg,
            #eef2ff 0%,
            #f8fafc 100%
        ) !important;

        font-family:
            'Inter',
            -apple-system,
            BlinkMacSystemFont,
            sans-serif;
    }

    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.95) !important;
        border-right: 1px solid #e2e8f0;
    }

    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.18s ease !important;
    }

    [data-testid="stMetric"] {
        background: white;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        border: 1px solid #e8edf3;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        margin-bottom: 0.5rem;
    }

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
    }

    .badge-red {
        background: #fff1f2;
        color: #be123c;
        border: 1px solid #fecdd3;
    }

    .badge-blue {
        background: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
    }

    .badge-green {
        background: #f0fdf4;
        color: #15803d;
        border: 1px solid #bbf7d0;
    }

    .badge-amber {
        background: #fffbeb;
        color: #b45309;
        border: 1px solid #fde68a;
    }

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

    .footer-brand {
        font-size: 1rem;
        font-weight: 700;
        color: #0f172a;
    }

    .footer-desc {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 0.15rem;
        line-height: 1.5;
    }

    .footer-stat {
        text-align: center;
    }

    .footer-stat .num {
        font-size: 1.2rem;
        font-weight: 700;
        color: #dc2626;
    }

    .footer-stat .lbl {
        font-size: 0.68rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .footer-note {
        font-size: 0.72rem;
        color: #94a3b8;
        line-height: 1.6;
    }

    .category-card {
        background: white;
        border-radius: 12px;
        padding: 0.9rem;
        border: 1px solid #e2e8f0;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .category-number {
        font-size: 1.8rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .category-name {
        font-size: 0.82rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }

    .category-percent {
        font-size: 0.72rem;
        color: #64748b;
        margin-top: 0.15rem;
    }

    </style>
    """
)


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def safe_float(value, default=0.0):

    try:

        value = float(value)

        if np.isnan(value) or np.isinf(value):
            return default

        return value

    except (TypeError, ValueError):

        return default


def get_risk_category(risk_score):

    score = safe_float(risk_score)

    if score < 30:
        return "LOW", "🟢"

    if score < 50:
        return "MEDIUM", "🟡"

    if score < 70:
        return "HIGH", "🟠"

    return "CRITICAL", "🔴"


def get_next_three_forecast_months(reference_date=None):

    if reference_date is None:
        reference_date = pd.Timestamp.now()

    reference_date = pd.Timestamp(reference_date)

    current_month = reference_date.replace(day=1)

    return [
        current_month + pd.DateOffset(months=1),
        current_month + pd.DateOffset(months=2),
        current_month + pd.DateOffset(months=3),
    ]


def get_forecast_horizon_options(reference_date=None):

    months = get_next_three_forecast_months(
        reference_date
    )

    options = []

    for index, forecast_date in enumerate(
        months,
        start=1,
    ):

        options.append(
            {
                "label": (
                    f"{forecast_date.strftime('%B %Y')} "
                    f"(+{index * 30} days)"
                ),

                "date": forecast_date,

                "days": index * 30,
            }
        )

    return options


def normalize_column_name(column):

    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def find_column(df, possible_names):

    normalized = {
        normalize_column_name(col): col
        for col in df.columns
    }

    for name in possible_names:

        key = normalize_column_name(name)

        if key in normalized:
            return normalized[key]

    return None


# =====================================================================
# DATA LOADING
# =====================================================================

@st.cache_data
def load_current_risk():

    if not CURRENT_DATA_FILE.exists():

        st.error(
            f"Current risk data file not found:\n\n"
            f"{CURRENT_DATA_FILE}"
        )

        return None

    try:

        df = pd.read_csv(
            CURRENT_DATA_FILE
        )

        required_columns = [
            "state",
            "month",
            "total_risk",
            "climate_stress",
            "agri_stress",
            "econ_stress",
        ]

        missing = [
            col
            for col in required_columns
            if col not in df.columns
        ]

        if missing:

            st.error(
                "Current risk CSV is missing required columns: "
                + ", ".join(missing)
            )

            return None

        df["month"] = (
            df["month"]
            .astype(str)
        )

        numeric_columns = [
            "total_risk",
            "climate_stress",
            "agri_stress",
            "econ_stress",
            "temp_anomaly_deg",
            "rainfall_anomaly_pct",
            "heatwave_days",
            "crop_failure_rate_pct",
            "unemployment_rate_pct",
            "inflation_rate_pct",
        ]

        for col in numeric_columns:

            if col in df.columns:

                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce",
                )

        latest_month = df["month"].max()

        latest = df[
            df["month"] == latest_month
        ].copy()

        return latest

    except Exception as exc:

        st.error(
            f"Error loading current risk data: {exc}"
        )

        return None


@st.cache_data
def load_historical_data():

    if not CURRENT_DATA_FILE.exists():
        return None

    try:

        df = pd.read_csv(
            CURRENT_DATA_FILE
        )

        if "month" not in df.columns:
            return None

        df["month_date"] = pd.to_datetime(
            df["month"].astype(str) + "-01",
            errors="coerce",
        )

        numeric_columns = [
            "total_risk",
            "climate_stress",
            "agri_stress",
            "econ_stress",
        ]

        for col in numeric_columns:

            if col in df.columns:

                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce",
                )

        return df

    except Exception as exc:

        st.warning(
            f"Historical data could not be loaded: {exc}"
        )

        return None


@st.cache_data
def load_forecasts():

    current_year = datetime.now().year

    possible_files = [
        REPORTS_DIR
        / f"simple_forecasts_{current_year}.csv",

        REPORTS_DIR
        / f"simple_forecasts_{current_year + 1}.csv",

        REPORTS_DIR
        / f"simple_forecasts_{current_year - 1}.csv",

        REPORTS_DIR
        / "simple_forecasts_2026.csv",
    ]

    checked = set()

    for file_path in possible_files:

        if file_path in checked:
            continue

        checked.add(file_path)

        if not file_path.exists():
            continue

        try:

            df = pd.read_csv(
                file_path
            )

            if "state" not in df.columns:
                continue

            return df

        except Exception as exc:

            st.warning(
                f"Could not read "
                f"{file_path.name}: {exc}"
            )

    st.warning(
        "No simple forecast CSV was found "
        "in outputs/reports."
    )

    return None


@st.cache_data
def load_prophet_forecasts():

    current_year = datetime.now().year

    possible_files = [
        REPORTS_DIR
        / f"prophet_forecasts_{current_year}.csv",

        REPORTS_DIR
        / f"prophet_forecasts_{current_year + 1}.csv",

        REPORTS_DIR
        / f"prophet_forecasts_{current_year - 1}.csv",

        REPORTS_DIR
        / "prophet_forecasts_2026.csv",
    ]

    for file_path in possible_files:

        if not file_path.exists():
            continue

        try:

            return pd.read_csv(
                file_path
            )

        except Exception:
            continue

    return None


@st.cache_data
def load_trend_analysis():

    file_path = (
        REPORTS_DIR
        / "trend_analysis_results.csv"
    )

    if not file_path.exists():
        return None

    try:

        return pd.read_csv(
            file_path
        )

    except Exception:
        return None


# =====================================================================
# FORECAST COLUMN DETECTION
# =====================================================================

def get_forecast_columns(
    columns,
    target_date,
):

    columns = list(columns)

    target_date = pd.Timestamp(
        target_date
    )

    target_short = (
        target_date.strftime("%b")
        .lower()
    )

    target_long = (
        target_date.strftime("%B")
        .lower()
    )

    target_year = str(
        target_date.year
    )

    origin_date = (
        target_date
        - pd.DateOffset(months=2)
    )

    origin_short = (
        origin_date.strftime("%b")
        .lower()
    )

    origin_long = (
        origin_date.strftime("%B")
        .lower()
    )

    origin_year = str(
        origin_date.year
    )

    found = []

    for prefix in FORECAST_MODELS:

        candidates = [

            f"{prefix}_{target_short}",

            f"{prefix}_{target_short}{target_year}",

            f"{prefix}_{target_long}",

            f"{prefix}_{target_long}{target_year}",

            f"{prefix}_{target_short}_{target_year}",

            f"{prefix}_{target_long}_{target_year}",

            f"{prefix}_{origin_short}",

            f"{prefix}_{origin_short}{origin_year}",

            f"{prefix}_{origin_long}",

            f"{prefix}_{origin_long}{origin_year}",

            f"{prefix}_{origin_short}_{origin_year}",

            f"{prefix}_{origin_long}_{origin_year}",
        ]

        for candidate in candidates:

            for actual_column in columns:

                if (
                    str(actual_column).lower()
                    == candidate.lower()
                ):

                    found.append(
                        actual_column
                    )

    if not found:

        month_tokens = {
            target_short,
            target_long,
            target_year,
            origin_short,
            origin_long,
            origin_year,
        }

        for column in columns:

            column_lower = (
                str(column).lower()
            )

            if "forecast" not in column_lower:
                continue

            if not any(
                prefix in column_lower
                for prefix in FORECAST_MODELS
            ):
                continue

            if any(
                token in column_lower
                for token in month_tokens
            ):

                found.append(
                    column
                )

    return list(
        dict.fromkeys(found)
    )


def get_best_forecast_value(
    row,
    target_date,
):

    columns = get_forecast_columns(
        row.index,
        target_date,
    )

    priority_order = []

    for prefix in FORECAST_MODELS:

        for column in columns:

            if (
                str(column)
                .lower()
                .startswith(prefix)
            ):

                priority_order.append(
                    column
                )

    for column in priority_order:

        value = safe_float(
            row.get(column),
            np.nan,
        )

        if not np.isnan(value):
            return value, column

    return None, None


def create_dynamic_horizon_map(
    forecasts
):

    horizon_options = (
        get_forecast_horizon_options()
    )

    horizon_map = {}

    for option in horizon_options:

        target_date = option["date"]

        columns = get_forecast_columns(
            forecasts.columns,
            target_date,
        )

        horizon_map[
            option["label"]
        ] = {
            "date": target_date,
            "days": option["days"],
            "columns": columns,
        }

    return horizon_map


def select_active_forecast_column(
    columns
):

    if not columns:
        return None

    for prefix in FORECAST_MODELS:

        matches = [
            column
            for column in columns
            if (
                str(column)
                .lower()
                .startswith(prefix)
            )
        ]

        if matches:
            return matches[0]

    return columns[0]


# =====================================================================
# LOGIN
# =====================================================================

def show_login_page():

    st.html(
        "<br><br><br>"
    )

    _, center_col, _ = st.columns(
        [1, 1.2, 1]
    )

    with center_col:

        with st.container(
            border=True
        ):

            st.html(
                """
                <h1 style="
                    text-align:center;
                    font-size:3rem;
                ">
                    🌍
                </h1>
                """
            )

            st.html(
                """
                <h2 style="
                    text-align:center;
                    margin-top:-10px;
                ">
                    CrisisLens
                </h2>
                """
            )

            st.html(
                """
                <p style="
                    text-align:center;
                    color:#64748b;
                ">
                    Multi-factor crisis risk forecasting<br>
                    Indian States & UTs
                </p>
                """
            )

            st.info(
                """
                📋 **Demo credentials**

                Email: `admin@disaster.gov`

                Password: `admin123`
                """
            )

            with st.form(
                "login_form",
                clear_on_submit=False,
            ):

                email = st.text_input(
                    "Email Address",
                    placeholder="admin@disaster.gov",
                )

                password = st.text_input(
                    "Password",
                    placeholder="••••••••",
                    type="password",
                )

                submitted = (
                    st.form_submit_button(
                        "Sign In →",
                        use_container_width=True,
                    )
                )

                if submitted:

                    email = (
                        email
                        .strip()
                        .lower()
                    )

                    if (
                        not email
                        or "@" not in email
                    ):

                        st.error(
                            "Please enter a valid "
                            "email address."
                        )

                    elif len(password) < 6:

                        st.error(
                            "Password must be at "
                            "least 6 characters."
                        )

                    elif (
                        VALID_CREDENTIALS.get(
                            email
                        )
                        == password
                    ):

                        st.session_state.authenticated = True

                        st.session_state.username = (
                            email
                        )

                        st.success(
                            "✅ Welcome! "
                            "Loading dashboard..."
                        )

                        st.rerun()

                    else:

                        st.error(
                            "Invalid credentials. "
                            "Please use the demo credentials."
                        )

            st.html(
                """
                <p style="
                    text-align:center;
                    font-size:0.75rem;
                    color:#94a3b8;
                    margin-top:10px;
                ">
                    Contact your system administrator
                    for access.
                </p>
                """
            )


# =====================================================================
# AUTH GATE
# =====================================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "username" not in st.session_state:
    st.session_state.username = ""

if not st.session_state.authenticated:

    show_login_page()

    st.stop()


# =====================================================================
# STATE POPUP
# =====================================================================

@st.dialog("📍 State Risk Breakdown")
def show_state_popup(
    state_name,
    total_risk,
    climate_stress,
    agri_stress,
    econ_stress,
):

    risk_cat, risk_emoji = (
        get_risk_category(
            total_risk
        )
    )

    components = [
        (
            "Climate",
            safe_float(
                climate_stress
            ),
        ),

        (
            "Agriculture",
            safe_float(
                agri_stress
            ),
        ),

        (
            "Economic",
            safe_float(
                econ_stress
            ),
        ),
    ]

    dominant = max(
        components,
        key=lambda x: x[1],
    )

    color_map = {
        "CRITICAL": "#dc2626",
        "HIGH": "#ea580c",
        "MEDIUM": "#d97706",
        "LOW": "#16a34a",
    }

    bg_map = {
        "CRITICAL": "#fff1f2",
        "HIGH": "#fff7ed",
        "MEDIUM": "#fffbeb",
        "LOW": "#f0fdf4",
    }

    color = color_map[
        risk_cat
    ]

    bg = bg_map[
        risk_cat
    ]

    st.html(
        f"""
        <div style="
            text-align:center;
            background:{bg};
            border-radius:14px;
            padding:1.2rem 1rem 0.8rem;
            margin-bottom:1.2rem;
            border:1px solid {color}30;
        ">

            <div style="
                font-size:0.72rem;
                text-transform:uppercase;
                letter-spacing:0.1em;
                color:{color};
                font-weight:700;
            ">
                {state_name}
            </div>

            <div style="
                font-size:3.2rem;
                font-weight:800;
                color:{color};
                line-height:1;
            ">
                {safe_float(total_risk):.1f}
            </div>

            <div style="
                font-size:0.78rem;
                color:#64748b;
            ">
                out of 100
            </div>

            <div style="
                display:inline-block;
                background:{color};
                color:white;
                padding:0.25rem 1rem;
                border-radius:20px;
                font-size:0.75rem;
                font-weight:700;
                margin-top:0.6rem;
            ">
                {risk_emoji} {risk_cat} RISK
            </div>

        </div>
        """
    )

    cols = st.columns(3)

    weighted_components = [
        (
            "🌦 Climate",
            climate_stress,
            "35% weight",
        ),

        (
            "🌾 Agriculture",
            agri_stress,
            "35% weight",
        ),

        (
            "💰 Economic",
            econ_stress,
            "30% weight",
        ),
    ]

    for col, (
        label,
        score,
        weight,
    ) in zip(
        cols,
        weighted_components,
    ):

        category, emoji = (
            get_risk_category(
                score
            )
        )

        with col:

            st.metric(
                label,
                f"{safe_float(score):.1f}",
                delta=(
                    f"{emoji} "
                    f"{category} · "
                    f"{weight}"
                ),
                delta_color="off",
            )

    st.markdown(
        f"""
        **Primary driver:** {dominant[0]} stress is the
        biggest contributor at **{dominant[1]:.1f}/100**.
        """
    )

    st.markdown(
        "**What this means for decision-makers:**"
    )

    if risk_cat == "CRITICAL":

        st.error(
            f"🚨 {state_name} needs **immediate "
            "multi-agency intervention**. "
            "All stress domains should be "
            "monitored closely."
        )

    elif risk_cat == "HIGH":

        st.warning(
            f"⚠️ {state_name} is under **significant "
            "pressure**. Alert district authorities "
            "and pre-position relief stocks."
        )

    elif risk_cat == "MEDIUM":

        st.info(
            f"📊 {state_name} shows **moderate stress**. "
            "Continue regular monitoring."
        )

    else:

        st.success(
            f"✅ {state_name} is **relatively stable**. "
            "Routine monitoring is sufficient."
        )


# =====================================================================
# MAIN DASHBOARD
# =====================================================================

def main():

    # -----------------------------------------------------------------
    # HEADER
    # -----------------------------------------------------------------

    now_str = datetime.now().strftime(
        "%d %b %Y, %H:%M"
    )

    st.html(
        f"""
        <div class="dash-header">

            <div>

                <div class="dash-title">
                    🌍 CrisisLens
                </div>

                <div class="dash-subtitle">
                    India's multi-factor crisis risk intelligence
                    platform · Climate · Agriculture · Economic
                    stress indicators
                </div>

            </div>

            <div class="dash-badges">

                <span class="badge badge-red">
                    🔴 Live Forecast
                </span>

                <span class="badge badge-blue">
                    📍 States & UTs
                </span>

                <span class="badge badge-green">
                    ✅ 93.3% Accuracy
                </span>

                <span class="badge badge-amber">
                    🕐 {now_str}
                </span>

            </div>

        </div>
        """
    )

    # -----------------------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------------------

    with st.spinner(
        "🔄 Analysing risk patterns across India..."
    ):

        current_risk = (
            load_current_risk()
        )

        forecasts = (
            load_forecasts()
        )

        historical = (
            load_historical_data()
        )

        trends = (
            load_trend_analysis()
        )

    if current_risk is None:

        st.error(
            "❌ Current risk data could not be loaded."
        )

        st.info(
            f"Expected file:\n\n"
            f"`{CURRENT_DATA_FILE}`"
        )

        return

    if forecasts is None:

        st.warning(
            "⚠️ Forecast CSV is unavailable. "
            "The dashboard will still display "
            "current and historical risk."
        )

    # -----------------------------------------------------------------
    # SIDEBAR
    # -----------------------------------------------------------------

    st.sidebar.html(
        """
        <div style="
            padding:0.2rem 0 0.8rem;
            border-bottom:1px solid #E2E8F0;
            margin-bottom:1rem;
        ">

            <span style="
                font-size:0.68rem;
                font-weight:700;
                text-transform:uppercase;
                letter-spacing:0.12em;
                color:#64748B;
            ">
                Dashboard Controls
            </span>

        </div>
        """
    )

    states = sorted(
        current_risk["state"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not states:

        st.error(
            "No states were found in "
            "the current risk dataset."
        )

        return

    default_index = (
        states.index("Maharashtra")
        if "Maharashtra" in states
        else 0
    )

    selected_state = (
        st.sidebar.selectbox(
            "Select State/UT",
            states,
            index=default_index,
        )
    )

    # -----------------------------------------------------------------
    # DYNAMIC FORECAST HORIZONS
    # -----------------------------------------------------------------

    if forecasts is not None:

        horizon_map = (
            create_dynamic_horizon_map(
                forecasts
            )
        )

        horizon_options = list(
            horizon_map.keys()
        )

    else:

        horizon_map = {}

        horizon_options = [
            option["label"]
            for option
            in get_forecast_horizon_options()
        ]

    forecast_month = (
        st.sidebar.radio(
            "Forecast Horizon",
            horizon_options,
            index=0,
        )
    )

    fallback_option = (
        get_forecast_horizon_options()[0]
    )

    selected_horizon = (
        horizon_map.get(
            forecast_month,
            {
                "date":
                    fallback_option["date"],

                "days":
                    fallback_option["days"],

                "columns": [],
            },
        )
    )

    target_forecast_date = (
        selected_horizon["date"]
    )

    target_forecast_days = (
        selected_horizon["days"]
    )

    target_forecast_columns = (
        selected_horizon["columns"]
    )

    alert_threshold = (
        st.sidebar.slider(
            "Alert Threshold",
            min_value=30.0,
            max_value=70.0,
            value=50.0,
            step=5.0,
            help=(
                "Risk scores at or above this "
                "value trigger alerts."
            ),
        )
    )

    # -----------------------------------------------------------------
    # APPLY FORECAST
    # -----------------------------------------------------------------

    active_forecast_col = (
        select_active_forecast_column(
            target_forecast_columns
        )
    )

    horizon_risk = (
        current_risk.copy()
    )

    is_forecast_view = False

    if (
        forecasts is not None
        and active_forecast_col is not None
    ):

        forecast_lookup = (
            forecasts[
                [
                    "state",
                    active_forecast_col,
                ]
            ].copy()
        )

        forecast_lookup["state"] = (
            forecast_lookup["state"]
            .astype(str)
            .str.strip()
        )

        forecast_lookup[
            "forecast_total_risk"
        ] = pd.to_numeric(
            forecast_lookup[
                active_forecast_col
            ],
            errors="coerce",
        )

        forecast_lookup = (
            forecast_lookup[
                [
                    "state",
                    "forecast_total_risk",
                ]
            ]
        )

        horizon_risk = (
            horizon_risk.merge(
                forecast_lookup,
                on="state",
                how="left",
            )
        )

        horizon_risk[
            "total_risk"
        ] = pd.to_numeric(
            horizon_risk[
                "forecast_total_risk"
            ],
            errors="coerce",
        ).fillna(
            pd.to_numeric(
                horizon_risk[
                    "total_risk"
                ],
                errors="coerce",
            )
        ).clip(
            0,
            100,
        )

        horizon_risk.drop(
            columns=[
                "forecast_total_risk"
            ],
            inplace=True,
            errors="ignore",
        )

        is_forecast_view = True

    # -----------------------------------------------------------------
    # SEASONAL ADJUSTMENT
    # -----------------------------------------------------------------

    target_month_name = (
        target_forecast_date.strftime(
            "%B"
        )
    )

    if is_forecast_view:

        def apply_seasonal(row):

            state = str(
                row.get(
                    "state",
                    "",
                )
            )

            base_risk = safe_float(
                row.get(
                    "total_risk",
                ),
                0,
            )

            if state not in SEASON_BOOST:
                return base_risk

            multiplier = (
                SEASON_BOOST[state].get(
                    target_month_name,
                    1.0,
                )
            )

            climate = safe_float(
                row.get(
                    "climate_stress",
                ),
                50,
            )

            agriculture = safe_float(
                row.get(
                    "agri_stress",
                ),
                50,
            )

            economic = safe_float(
                row.get(
                    "econ_stress",
                ),
                50,
            )

            adjusted_climate = min(
                climate * multiplier,
                100,
            )

            adjusted_total = (
                adjusted_climate
                * CLIMATE_WEIGHT
                +
                agriculture
                * AGRI_WEIGHT
                +
                economic
                * ECON_WEIGHT
            )

            if (
                climate == 50
                and agriculture == 50
                and economic == 50
            ):

                return base_risk

            return min(
                max(
                    adjusted_total,
                    0,
                ),
                100,
            )

        horizon_risk[
            "total_risk"
        ] = horizon_risk.apply(
            apply_seasonal,
            axis=1,
        )

    # -----------------------------------------------------------------
    # SIDEBAR HELP
    # -----------------------------------------------------------------

    next_months = (
        get_next_three_forecast_months()
    )

    st.sidebar.markdown("---")

    st.sidebar.info(
        f"""
        **How to use this dashboard**

        1. **Pick a state** from the dropdown.
        2. **Choose a forecast horizon**.
        3. **Adjust the alert threshold**.
        4. Review recommended actions.

        **Forecast window**

        **{next_months[0].strftime("%B %Y")}**
        →

        **{next_months[1].strftime("%B %Y")}**
        →

        **{next_months[2].strftime("%B %Y")}**

        **Risk guide**

        🟢 < 30 = Low

        🟡 30–50 = Medium

        🟠 50–70 = High

        🔴 ≥ 70 = Critical
        """
    )

    st.sidebar.markdown("---")

    st.sidebar.html(
        f"""
        <div style="
            font-size:0.78rem;
            color:#64748B;
            margin-bottom:0.5rem;
        ">

            👤 Signed in as
            <strong>
                {st.session_state.username}
            </strong>

        </div>
        """
    )

    if st.sidebar.button(
        "🚪 Logout",
        use_container_width=True,
    ):

        st.session_state.authenticated = False

        st.session_state.username = ""

        st.rerun()

    # -----------------------------------------------------------------
    # NATIONAL OVERVIEW
    # -----------------------------------------------------------------

    horizon_risk[
        "total_risk"
    ] = pd.to_numeric(
        horizon_risk[
            "total_risk"
        ],
        errors="coerce",
    )

    horizon_risk = (
        horizon_risk.dropna(
            subset=[
                "total_risk"
            ]
        ).copy()
    )

    if horizon_risk.empty:

        st.error(
            "No valid risk scores are available."
        )

        return

    avg_risk = (
        horizon_risk[
            "total_risk"
        ].mean()
    )

    high_risk_count = int(
        (
            horizon_risk[
                "total_risk"
            ]
            >= alert_threshold
        ).sum()
    )

    max_risk_state = (
        horizon_risk.loc[
            horizon_risk[
                "total_risk"
            ].idxmax()
        ]
    )

    nat_cat, nat_emoji = (
        get_risk_category(
            avg_risk
        )
    )

    view_label = (
        forecast_month
        if is_forecast_view
        else "Current"
    )

    st.header(
        f"📊 National Overview — {view_label}"
    )

    st.caption(
        "Risk is scored from 0–100. "
        "Below 30 = Low · 30–50 = Medium · "
        "50–70 = High · 70+ = Critical."
    )

    # -----------------------------------------------------------------
    # NATIONAL EXPLANATIONS
    # -----------------------------------------------------------------

    with st.expander(
        "ℹ️ What does the National Risk Score mean?"
    ):

        st.markdown(
            f"""
            ### 🇮🇳 National Risk Score

            The **National Average Risk Score is
            {avg_risk:.1f}/100**.

            It represents the average crisis-risk
            level across all available States and
            Union Territories for the selected view.

            **Risk interpretation:**

            - 🟢 **Low:** 0–29
            - 🟡 **Medium:** 30–49
            - 🟠 **High:** 50–69
            - 🔴 **Critical:** 70–100

            A higher score indicates greater combined
            stress across **climate, agriculture and
            economic conditions**.

            The national score is an overall monitoring
            indicator. It does **not** mean that a disaster
            is guaranteed to occur.
            """
        )

    with st.expander(
        "ℹ️ Why are these states marked as 'Needing Attention'?"
    ):

        st.markdown(
            f"""
            ### ⚠️ States Needing Attention

            The dashboard currently uses an alert
            threshold of **{alert_threshold:.0f}/100**.

            Any State or Union Territory with a risk
            score **equal to or greater than
            {alert_threshold:.0f}** is counted as
            needing attention.

            **Current result:**

            **{high_risk_count} of {len(horizon_risk)}
            States/UTs** meet or exceed the alert
            threshold.

            This does not automatically mean a disaster
            is occurring.

            These states should instead receive closer
            monitoring, early-warning review and
            preparedness attention.

            You can change this threshold using the
            **Alert Threshold** slider.
            """
        )

    with st.expander(
        "🔮 What does Forecasting mean?"
    ):

        st.markdown(
            f"""
            ### 🔮 Forecasting

            **Forecasting means estimating a future
            risk level using historical patterns and
            forecasting models.**

            The selected forecast period is:

            **{target_forecast_date.strftime("%B %Y")}**

            CrisisLens can use forecasting approaches
            including:

            - **ARIMA** — models historical time-series
              patterns.
            - **Linear Trend** — estimates future direction
              from historical trends.
            - **Prophet** — models trend and seasonal
              patterns.

            A forecast is an **estimated future risk
            level**, not a guarantee that a disaster
            will occur.

            Forecasting can support:

            - Early warning
            - Resource planning
            - Relief-stock preparation
            - High-risk region monitoring
            - Decision support

            **Example:**

            A forecast of **72/100** means the model
            estimates a **Critical Risk level** for
            that period. It does not mean that a
            disaster is certain to happen.
            """
        )

    # -----------------------------------------------------------------
    # NATIONAL METRICS
    # -----------------------------------------------------------------

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:

        st.metric(
            "🇮🇳 National Avg Risk",
            f"{avg_risk:.1f} / 100",
            delta=(
                f"{nat_emoji} "
                f"{nat_cat} overall"
            ),
            delta_color="off",
        )

    with col2:

        st.metric(
            "⚠️ States Needing Attention",
            f"{high_risk_count} of "
            f"{len(horizon_risk)}",
            delta=(
                f"Threshold: "
                f"{alert_threshold:.0f}"
            ),
            delta_color="off",
        )

    with col3:

        max_name = str(
            max_risk_state["state"]
        )

        max_score = safe_float(
            max_risk_state[
                "total_risk"
            ]
        )

        max_cat, max_emoji = (
            get_risk_category(
                max_score
            )
        )

        display_name = (
            max_name[:18] + "..."
            if len(max_name) > 18
            else max_name
        )

        st.metric(
            "🔴 Highest Risk State",
            display_name,
            delta=(
                f"{max_emoji} "
                f"{max_score:.1f} — "
                f"{max_cat}"
            ),
            delta_color="off",
        )

    with col4:

        if (
            forecasts is not None
            and active_forecast_col
            is not None
        ):

            forecast_values = (
                pd.to_numeric(
                    forecasts[
                        active_forecast_col
                    ],
                    errors="coerce",
                )
            )

            forecast_avg = (
                forecast_values.mean()
            )

            current_avg = (
                current_risk[
                    "total_risk"
                ].mean()
            )

            forecast_change = (
                forecast_avg
                - current_avg
            )

            st.metric(
                target_forecast_date.strftime(
                    "%b %Y"
                ),
                f"{forecast_avg:.1f}",
                delta=(
                    f"{forecast_change:+.1f}"
                ),
                delta_color="inverse",
            )

        else:

            st.metric(
                target_forecast_date.strftime(
                    "%b %Y"
                ),
                "N/A",
                delta="Forecast unavailable",
                delta_color="off",
            )

    # -----------------------------------------------------------------
    # RANKING + CATEGORY
    # -----------------------------------------------------------------

    col_left, col_right = (
        st.columns([2, 1])
    )

    with col_left:

        st.subheader(
            "🗺️ Risk Across All States"
        )

        st.caption(
            "Ranked from lowest to highest. "
            "State names are enlarged for readability."
        )

        map_data = (
            horizon_risk.sort_values(
                "total_risk",
                ascending=True,
            ).copy()
        )

        def get_dominant_factor(row):

            factors = {

                "Climate": safe_float(
                    row.get(
                        "climate_stress"
                    )
                ),

                "Agriculture": safe_float(
                    row.get(
                        "agri_stress"
                    )
                ),

                "Economic": safe_float(
                    row.get(
                        "econ_stress"
                    )
                ),
            }

            return max(
                factors,
                key=factors.get,
            )

        map_data[
            "Dominant Stress"
        ] = map_data.apply(
            get_dominant_factor,
            axis=1,
        )

        fig_map = px.bar(
            map_data,
            x="total_risk",
            y="state",
            orientation="h",
            color="total_risk",
            color_continuous_scale=[
                (0.0, "#2ecc71"),
                (0.33, "#f39c12"),
                (0.66, "#e74c3c"),
                (1.0, "#c0392b"),
            ],
            range_color=[0, 100],
            text="total_risk",
            hover_data={
                "total_risk": ":.1f",
                "Dominant Stress": True,
                "state": False,
            },
        )

        fig_map.update_traces(
            texttemplate=(
                "<b>%{x:.1f}</b>"
            ),
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Risk Score: %{x:.1f}<br>"
                "Primary Driver: "
                "%{customdata[0]}"
                "<extra></extra>"
            ),
        )

        fig_map.add_vline(
            x=alert_threshold,
            line_dash="dot",
            line_color="red",
            line_width=2,
            annotation_text=(
                f"Alert Trigger "
                f"({alert_threshold:.0f})"
            ),
            annotation_position="bottom right",
        )

        fig_map.update_layout(
            height=max(
                700,
                len(map_data) * 34,
            ),

            xaxis_title="Risk Score",

            yaxis_title="",

            margin=dict(
                l=10,
                r=50,
                t=10,
                b=10,
            ),

            paper_bgcolor=(
                "rgba(0,0,0,0)"
            ),

            plot_bgcolor=(
                "rgba(0,0,0,0)"
            ),

            font=dict(
                family="Inter, sans-serif",
                color="#334155",
                size=13,
            ),

            xaxis=dict(
                gridcolor="#F1F5F9",
                linecolor="#E2E8F0",
                range=[0, 105],
            ),

            yaxis=dict(
                gridcolor="#F1F5F9",
                linecolor="#E2E8F0",
                tickfont=dict(
                    size=14,
                    color="#0f172a",
                ),
            ),
        )

        st.plotly_chart(
            fig_map,
            use_container_width=True,
        )

    with col_right:

        st.subheader(
            "📊 States by Risk Category"
        )

        horizon_risk[
            "Risk Category"
        ] = horizon_risk[
            "total_risk"
        ].apply(
            lambda value:
            get_risk_category(value)[0]
        )

        category_order = [
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL",
        ]

        category_counts = (
            horizon_risk[
                "Risk Category"
            ]
            .value_counts()
            .reindex(
                category_order,
                fill_value=0,
            )
        )

        total_states = (
            len(horizon_risk)
        )

        category_percentages = (
            category_counts
            / total_states
            * 100
        )

        # -------------------------------------------------------------
        # LARGER DONUT
        # -------------------------------------------------------------

        fig_pie = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            color=category_counts.index,
            color_discrete_map={
                "LOW": "#2ecc71",
                "MEDIUM": "#f39c12",
                "HIGH": "#e74c3c",
                "CRITICAL": "#c0392b",
            },
            hole=0.45,
        )

        fig_pie.update_traces(
            textposition="inside",
            textinfo="percent+label",
            textfont=dict(
                size=14
            ),
            hovertemplate=(
                "<b>%{label}</b><br>"
                "States/UTs: %{value}<br>"
                "Share: %{percent}"
                "<extra></extra>"
            ),
        )

        fig_pie.update_layout(
            height=420,

            paper_bgcolor=(
                "rgba(0,0,0,0)"
            ),

            plot_bgcolor=(
                "rgba(0,0,0,0)"
            ),

            showlegend=False,

            margin=dict(
                l=5,
                r=5,
                t=5,
                b=5,
            ),
        )

        st.plotly_chart(
            fig_pie,
            use_container_width=True,
        )

        # -------------------------------------------------------------
        # CATEGORY COUNTS + PERCENTAGES
        # -------------------------------------------------------------

        cat_cols = st.columns(2)

        category_colors = {
            "LOW": "#16a34a",
            "MEDIUM": "#d97706",
            "HIGH": "#ea580c",
            "CRITICAL": "#dc2626",
        }

        category_emojis = {
            "LOW": "🟢",
            "MEDIUM": "🟡",
            "HIGH": "🟠",
            "CRITICAL": "🔴",
        }

        for i, category in enumerate(
            category_order
        ):

            count = int(
                category_counts[
                    category
                ]
            )

            percentage = float(
                category_percentages[
                    category
                ]
            )

            with cat_cols[
                i % 2
            ]:

                st.html(
                    f"""
                    <div class="category-card"
                         style="
                            border-top:4px solid
                            {category_colors[category]};
                         ">

                        <div class="category-number"
                             style="
                                color:
                                {category_colors[category]};
                             ">

                            {category_emojis[category]}
                            {count}

                        </div>

                        <div class="category-name">

                            {category}

                        </div>

                        <div class="category-percent">

                            {percentage:.1f}% of
                            States/UTs

                        </div>

                    </div>
                    """
                )

        st.caption(
            f"Total States/UTs analysed: "
            f"**{total_states}**"
        )

        # -------------------------------------------------------------
        # TOP 5
        # -------------------------------------------------------------

        st.markdown(
            f"**🚨 Top 5 States — {view_label}**"
        )

        st.caption(
            "Highest-risk states are shown first."
        )

        top_columns = [
            "state",
            "total_risk",
            "climate_stress",
            "agri_stress",
            "econ_stress",
        ]

        available_top_columns = [
            col
            for col in top_columns
            if col in horizon_risk.columns
        ]

        top5 = (
            horizon_risk.nlargest(
                5,
                "total_risk",
            )[
                available_top_columns
            ]
        )

        for idx, row in top5.iterrows():

            category, emoji = (
                get_risk_category(
                    row["total_risk"]
                )
            )

            factors = {

                "Climate": safe_float(
                    row.get(
                        "climate_stress"
                    )
                ),

                "Agriculture": safe_float(
                    row.get(
                        "agri_stress"
                    )
                ),

                "Economic": safe_float(
                    row.get(
                        "econ_stress"
                    )
                ),
            }

            dominant = max(
                factors,
                key=factors.get,
            )

            st.html(
                f"""
                <div style="
                    background:#ffffff;
                    border:1px solid #e2e8f0;
                    border-left:4px solid #dc2626;
                    border-radius:10px;
                    padding:0.8rem 0.9rem;
                    margin-bottom:0.45rem;
                ">

                    <div style="
                        display:flex;
                        justify-content:space-between;
                        align-items:center;
                    ">

                        <span style="
                            font-size:0.98rem;
                            font-weight:700;
                            color:#0f172a;
                        ">

                            {row["state"]}

                        </span>

                        <span style="
                            font-size:1.05rem;
                            font-weight:800;
                            color:#dc2626;
                        ">

                            {safe_float(
                                row["total_risk"]
                            ):.1f}

                        </span>

                    </div>

                    <div style="
                        font-size:0.74rem;
                        color:#64748b;
                        margin-top:4px;
                    ">

                        {emoji} {category} ·
                        {dominant} stress dominant
                        ({factors[dominant]:.1f})

                    </div>

                </div>
                """
            )

            if st.button(
                "View details →",
                key=f"top5_{idx}",
            ):

                show_state_popup(
                    row["state"],
                    row["total_risk"],
                    row.get(
                        "climate_stress",
                        0,
                    ),
                    row.get(
                        "agri_stress",
                        0,
                    ),
                    row.get(
                        "econ_stress",
                        0,
                    ),
                )

    # -----------------------------------------------------------------
    # RECOMMENDED ACTIONS
    # -----------------------------------------------------------------

    st.markdown("---")

    st.header(
        "🚨 Which States Need Action Right Now?"
    )

    st.caption(
        f"States with risk ≥ "
        f"**{alert_threshold:.0f}** "
        "are listed below."
    )

    alerts = horizon_risk[
        horizon_risk[
            "total_risk"
        ]
        >= alert_threshold
    ].copy()

    alerts = alerts.sort_values(
        "total_risk",
        ascending=False,
    )

    if not alerts.empty:

        st.warning(
            f"⚠️ **{len(alerts)} states/UTs** "
            f"have risk ≥ "
            f"{alert_threshold:.0f}."
        )

        alerts_display = alerts[
            [
                "state",
                "total_risk",
                "climate_stress",
                "agri_stress",
                "econ_stress",
            ]
        ].copy()

        alerts_display.columns = [
            "State",
            "Total Risk",
            "Climate",
            "Agriculture",
            "Economic",
        ]

        alerts_display = (
            alerts_display.round(1)
        )

        st.dataframe(
            alerts_display,
            use_container_width=True,
            height=min(
                400,
                max(
                    100,
                    (len(alerts) + 1) * 35,
                ),
            ),
        )

        st.subheader(
            "💡 Recommended Actions"
        )

        for _, row in alerts.head(5).iterrows():

            factors = {

                "Climate": safe_float(
                    row["climate_stress"]
                ),

                "Agriculture": safe_float(
                    row["agri_stress"]
                ),

                "Economic": safe_float(
                    row["econ_stress"]
                ),
            }

            dominant_factor = max(
                factors,
                key=factors.get,
            )

            risk_cat, _ = (
                get_risk_category(
                    row["total_risk"]
                )
            )

            with st.expander(
                f"🎯 {row['state']} — "
                f"Risk: "
                f"{row['total_risk']:.1f} "
                f"({risk_cat}) | "
                f"Primary driver: "
                f"{dominant_factor}"
            ):

                col_a, col_b, col_c = (
                    st.columns(3)
                )

                with col_a:

                    climate = (
                        factors["Climate"]
                    )

                    st.metric(
                        "🌦 Climate",
                        f"{climate:.1f}",
                        delta=(
                            "High"
                            if climate > 60
                            else
                            "Moderate"
                            if climate > 40
                            else
                            "Low"
                        ),
                        delta_color="off",
                    )

                with col_b:

                    agriculture = (
                        factors[
                            "Agriculture"
                        ]
                    )

                    st.metric(
                        "🌾 Agriculture",
                        f"{agriculture:.1f}",
                        delta=(
                            "High"
                            if agriculture > 60
                            else
                            "Moderate"
                            if agriculture > 40
                            else
                            "Low"
                        ),
                        delta_color="off",
                    )

                with col_c:

                    economic = (
                        factors[
                            "Economic"
                        ]
                    )

                    st.metric(
                        "💰 Economic",
                        f"{economic:.1f}",
                        delta=(
                            "High"
                            if economic > 60
                            else
                            "Moderate"
                            if economic > 40
                            else
                            "Low"
                        ),
                        delta_color="off",
                    )

                st.markdown(
                    "**Recommended response:**"
                )

                if dominant_factor == "Climate":

                    temp = safe_float(
                        row.get(
                            "temp_anomaly_deg",
                            0,
                        )
                    )

                    rainfall = safe_float(
                        row.get(
                            "rainfall_anomaly_pct",
                            0,
                        )
                    )

                    st.markdown(
                        f"""
                        ⚠️ Climate stress is the
                        primary driver.

                        - **Immediate:** Issue heat/rainfall
                          advisories and activate disaster
                          response teams.

                        - **Short-term:** Pre-position
                          water tankers and emergency supplies.

                        - **Agriculture:** Alert farmers about
                          irrigation requirements.

                        - **Monitoring:** Track IMD/weather
                          updates regularly.

                        - **Current indicators:**
                          Temperature anomaly
                          {temp:.1f}°C;
                          rainfall anomaly
                          {rainfall:.1f}%.
                        """
                    )

                elif dominant_factor == "Agriculture":

                    crop_failure = safe_float(
                        row.get(
                            "crop_failure_rate_pct",
                            0,
                        )
                    )

                    st.markdown(
                        f"""
                        ⚠️ Agricultural stress is the
                        primary driver.

                        - **Immediate:** Deploy teams to
                          assess crop damage.

                        - **Short-term:** Fast-track crop
                          insurance and agricultural support.

                        - **Irrigation:** Mobilise available
                          water resources.

                        - **Relief:** Prepare district
                          agricultural relief packages.

                        - **Monitoring:** Conduct weekly
                          crop-condition assessments.

                        - **Current crop failure indicator:**
                          {crop_failure:.1f}%.
                        """
                    )

                else:

                    st.markdown(
                        f"""
                        ⚠️ Economic stress is the
                        primary driver at
                        **{factors["Economic"]:.1f}/100**.

                        - **Employment:** Accelerate
                          employment-support programmes.

                        - **Food security:** Audit PDS
                          stock levels.

                        - **Social protection:** Identify
                          vulnerable households.

                        - **Inflation:** Monitor essential
                          commodity prices.

                        - **Coordination:** Align Finance,
                          Labour and Food departments.
                        """
                    )

    else:

        st.success(
            f"✅ No states are currently above "
            f"the alert threshold of "
            f"{alert_threshold:.0f}."
        )

    # -----------------------------------------------------------------
    # DEEP DIVE — BOTTOM SECTION
    # -----------------------------------------------------------------

    st.markdown("---")

    st.header(
        f"🔍 Deep Dive: {selected_state}"
    )

    st.caption(
        "Detailed state-level analysis including "
        "current risk, historical trends, forecasts, "
        "model comparison, stress components and "
        "key indicators."
    )

    selected_rows = current_risk[
        current_risk["state"].astype(str)
        == str(selected_state)
    ]

    if selected_rows.empty:

        st.error(
            "Selected state could not be found."
        )

        return

    state_current = (
        selected_rows.iloc[0]
    )

    current_total_risk = safe_float(
        state_current.get(
            "total_risk"
        )
    )

    if current_total_risk >= 70:

        st.error(
            "🚨 Critical Risk Zone"
        )

    elif current_total_risk >= 50:

        st.warning(
            "⚠️ High Risk Zone"
        )

    elif current_total_risk >= 30:

        st.info(
            "🟡 Moderate Risk Zone"
        )

    else:

        st.success(
            "✅ Low Risk Zone"
        )

    # -----------------------------------------------------------------
    # STATE METRICS
    # -----------------------------------------------------------------

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    climate_score = safe_float(
        state_current.get(
            "climate_stress"
        )
    )

    agri_score = safe_float(
        state_current.get(
            "agri_stress"
        )
    )

    econ_score = safe_float(
        state_current.get(
            "econ_stress"
        )
    )

    with col1:

        risk_cat, risk_emoji = (
            get_risk_category(
                current_total_risk
            )
        )

        st.metric(
            "Overall Risk Score",
            f"{current_total_risk:.1f} / 100",
            delta=(
                f"{risk_emoji} "
                f"{risk_cat}"
            ),
            delta_color="off",
        )

    with col2:

        cat, emoji = (
            get_risk_category(
                climate_score
            )
        )

        st.metric(
            "🌦 Climate Stress",
            f"{climate_score:.1f} / 100",
            delta=(
                f"{emoji} {cat}"
            ),
            delta_color="off",
        )

    with col3:

        cat, emoji = (
            get_risk_category(
                agri_score
            )
        )

        st.metric(
            "🌾 Agriculture Stress",
            f"{agri_score:.1f} / 100",
            delta=(
                f"{emoji} {cat}"
            ),
            delta_color="off",
        )

    with col4:

        cat, emoji = (
            get_risk_category(
                econ_score
            )
        )

        st.metric(
            "💰 Economic Stress",
            f"{econ_score:.1f} / 100",
            delta=(
                f"{emoji} {cat}"
            ),
            delta_color="off",
        )

    # -----------------------------------------------------------------
    # HISTORICAL TREND
    # -----------------------------------------------------------------

    if historical is not None:

        st.subheader(
            f"📈 {selected_state}: "
            "Past Risk + Future Forecast"
        )

        st.caption(
            "Blue = historical risk · "
            "Orange = forecast · "
            "Red dotted line = alert threshold."
        )

        state_history = (
            historical[
                historical["state"].astype(str)
                == str(selected_state)
            ].copy()
        )

        state_history = (
            state_history.sort_values(
                "month_date"
            )
        )

        fig_trend = go.Figure()

        if (
            "month_date"
            in state_history.columns
            and
            "total_risk"
            in state_history.columns
        ):

            fig_trend.add_trace(
                go.Scatter(
                    x=state_history[
                        "month_date"
                    ],

                    y=state_history[
                        "total_risk"
                    ],

                    mode="lines+markers",

                    name="Historical Risk",

                    line=dict(
                        color="#1f77b4",
                        width=2,
                    ),

                    marker=dict(
                        size=6,
                    ),
                )
            )

        forecast_options = (
            get_forecast_horizon_options()
        )

        forecast_dates = []
        forecast_values = []

        if forecasts is not None:

            state_forecast_rows = (
                forecasts[
                    forecasts["state"]
                    .astype(str)
                    .str.strip()
                    ==
                    str(selected_state)
                    .strip()
                ]
            )

            if not state_forecast_rows.empty:

                state_forecast = (
                    state_forecast_rows.iloc[0]
                )

                for option in (
                    forecast_options
                ):

                    value, _ = (
                        get_best_forecast_value(
                            state_forecast,
                            option["date"],
                        )
                    )

                    forecast_dates.append(
                        option["date"]
                    )

                    forecast_values.append(
                        np.nan
                        if value is None
                        else value
                    )

        if (
            forecast_values
            and any(
                pd.notna(value)
                for value
                in forecast_values
            )
        ):

            fig_trend.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=forecast_values,
                    mode="lines+markers",
                    name="Forecast",
                    line=dict(
                        color="#ff7f0e",
                        width=2,
                        dash="dash",
                    ),
                    marker=dict(
                        size=10,
                        symbol="star",
                    ),
                )
            )

        else:

            st.info(
                "ℹ️ No matching forecast columns "
                "were found for the next three "
                "calendar months."
            )

        fig_trend.add_hline(
            y=alert_threshold,
            line_dash="dot",
            line_color="red",
            annotation_text=(
                f"Alert Threshold "
                f"({alert_threshold:.0f})"
            ),
            annotation_position="right",
        )

        fig_trend.update_layout(
            height=420,
            xaxis_title="Date",
            yaxis_title="Risk Score (0–100)",
            hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(
                family="Inter, sans-serif",
                color="#334155",
                size=12,
            ),
            xaxis=dict(
                gridcolor="#F1F5F9",
                linecolor="#E2E8F0",
            ),
            yaxis=dict(
                gridcolor="#F1F5F9",
                linecolor="#E2E8F0",
                range=[0, 100],
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        st.plotly_chart(
            fig_trend,
            use_container_width=True,
        )

        # -------------------------------------------------------------
        # MODEL COMPARISON
        # -------------------------------------------------------------

        st.subheader(
            "📊 Model Comparison "
            "(ARIMA vs Linear vs Prophet)"
        )

        if run_arima is None:

            st.warning(
                "ARIMA module could not be imported. "
                "Check models/arima_model.py."
            )

        elif len(state_history) < 7:

            st.warning(
                "Not enough historical observations "
                "to calculate model comparison."
            )

        else:

            try:

                series = pd.to_numeric(
                    state_history[
                        "total_risk"
                    ],
                    errors="coerce",
                ).dropna()

                if len(series) < 7:

                    raise ValueError(
                        "Not enough numeric "
                        "historical observations."
                    )

                train = series.iloc[:-6]

                test = series.iloc[-6:]

                # -----------------------------------------------------
                # ARIMA
                # -----------------------------------------------------

                arima_pred = np.asarray(
                    run_arima(train)
                ).flatten()

                if len(arima_pred) < len(test):

                    arima_pred = np.pad(
                        arima_pred,
                        (
                            0,
                            len(test)
                            - len(arima_pred),
                        ),
                        constant_values=np.nan,
                    )

                arima_pred = (
                    arima_pred[:len(test)]
                )

                # -----------------------------------------------------
                # LINEAR TREND
                # -----------------------------------------------------

                x_train = np.arange(
                    len(train)
                )

                coefficients = np.polyfit(
                    x_train,
                    train.values,
                    1,
                )

                x_future = np.arange(
                    len(train),
                    len(train)
                    + len(test),
                )

                linear_pred = (
                    coefficients[0]
                    * x_future
                    + coefficients[1]
                )

                # -----------------------------------------------------
                # PROPHET
                # -----------------------------------------------------

                prophet_pred = None

                prophet_df = (
                    load_prophet_forecasts()
                )

                if prophet_df is not None:

                    prophet_df = (
                        prophet_df.copy()
                    )

                    prophet_df.columns = [
                        normalize_column_name(
                            col
                        )
                        for col
                        in prophet_df.columns
                    ]

                    required_prophet = {
                        "state",
                        "month",
                        "predicted_risk",
                    }

                    if required_prophet.issubset(
                        prophet_df.columns
                    ):

                        prophet_df[
                            "state"
                        ] = (
                            prophet_df[
                                "state"
                            ]
                            .astype(str)
                            .str.strip()
                            .str.lower()
                        )

                        prophet_df[
                            "month"
                        ] = pd.to_datetime(
                            prophet_df[
                                "month"
                            ],
                            errors="coerce",
                        )

                        prophet_df[
                            "predicted_risk"
                        ] = pd.to_numeric(
                            prophet_df[
                                "predicted_risk"
                            ],
                            errors="coerce",
                        )

                        selected_clean = (
                            str(
                                selected_state
                            )
                            .strip()
                            .lower()
                        )

                        prophet_state = (
                            prophet_df[
                                prophet_df[
                                    "state"
                                ]
                                == selected_clean
                            ]
                            .dropna(
                                subset=[
                                    "month",
                                    "predicted_risk",
                                ]
                            )
                            .sort_values(
                                "month"
                            )
                        )

                        if not prophet_state.empty:

                            prophet_pred = (
                                prophet_state[
                                    "predicted_risk"
                                ]
                                .values[
                                    -len(test):
                                ]
                            )

                # -----------------------------------------------------
                # METRICS
                # -----------------------------------------------------

                def calculate_metrics(
                    actual,
                    prediction,
                ):

                    actual = np.asarray(
                        actual,
                        dtype=float,
                    )

                    prediction = np.asarray(
                        prediction,
                        dtype=float,
                    )

                    valid = (
                        np.isfinite(actual)
                        &
                        np.isfinite(
                            prediction
                        )
                    )

                    if valid.sum() == 0:

                        return (
                            np.nan,
                            np.nan,
                        )

                    actual = (
                        actual[valid]
                    )

                    prediction = (
                        prediction[valid]
                    )

                    rmse = np.sqrt(
                        mean_squared_error(
                            actual,
                            prediction,
                        )
                    )

                    mae = (
                        mean_absolute_error(
                            actual,
                            prediction,
                        )
                    )

                    return (
                        rmse,
                        mae,
                    )

                arima_rmse, arima_mae = (
                    calculate_metrics(
                        test.values,
                        arima_pred,
                    )
                )

                linear_rmse, linear_mae = (
                    calculate_metrics(
                        test.values,
                        linear_pred,
                    )
                )

                prophet_rmse = None

                prophet_mae = None

                if prophet_pred is not None:

                    prophet_pred = (
                        np.asarray(
                            prophet_pred,
                            dtype=float,
                        )
                    )

                    length = min(
                        len(test),
                        len(prophet_pred),
                    )

                    (
                        prophet_rmse,
                        prophet_mae,
                    ) = calculate_metrics(
                        test.values[
                            -length:
                        ],

                        prophet_pred[
                            -length:
                        ],
                    )

                # -----------------------------------------------------
                # COMPARISON CHART
                # -----------------------------------------------------

                fig_compare = go.Figure()

                fig_compare.add_trace(
                    go.Scatter(
                        y=test.values,
                        name="Actual",
                        mode="lines+markers",
                    )
                )

                fig_compare.add_trace(
                    go.Scatter(
                        y=arima_pred,
                        name="ARIMA",
                        mode="lines+markers",
                    )
                )

                fig_compare.add_trace(
                    go.Scatter(
                        y=linear_pred,
                        name="Linear",
                        mode="lines+markers",
                    )
                )

                if prophet_pred is not None:

                    prophet_plot = np.full(
                        len(test),
                        np.nan,
                    )

                    length = min(
                        len(test),
                        len(prophet_pred),
                    )

                    prophet_plot[
                        -length:
                    ] = prophet_pred[
                        -length:
                    ]

                    fig_compare.add_trace(
                        go.Scatter(
                            y=prophet_plot,
                            name="Prophet",
                            mode="lines+markers",
                        )
                    )

                fig_compare.update_layout(
                    template="plotly_white",
                    hovermode="x unified",
                    height=380,
                    margin=dict(
                        l=10,
                        r=10,
                        t=30,
                        b=10,
                    ),
                )

                st.subheader(
                    "📏 How Accurate Are the Predictions?"
                )

                col1, col2, col3 = (
                    st.columns(3)
                )

                with col1:

                    st.metric(
                        "ARIMA RMSE",
                        (
                            f"{arima_rmse:.2f}"
                            if np.isfinite(
                                arima_rmse
                            )
                            else "N/A"
                        ),
                    )

                    st.metric(
                        "ARIMA MAE",
                        (
                            f"{arima_mae:.2f}"
                            if np.isfinite(
                                arima_mae
                            )
                            else "N/A"
                        ),
                    )

                with col2:

                    st.metric(
                        "Linear RMSE",
                        (
                            f"{linear_rmse:.2f}"
                            if np.isfinite(
                                linear_rmse
                            )
                            else "N/A"
                        ),
                    )

                    st.metric(
                        "Linear MAE",
                        (
                            f"{linear_mae:.2f}"
                            if np.isfinite(
                                linear_mae
                            )
                            else "N/A"
                        ),
                    )

                with col3:

                    if prophet_pred is not None:

                        st.metric(
                            "Prophet RMSE",
                            (
                                f"{prophet_rmse:.2f}"
                                if (
                                    prophet_rmse
                                    is not None
                                    and
                                    np.isfinite(
                                        prophet_rmse
                                    )
                                )
                                else "N/A"
                            ),
                        )

                        st.metric(
                            "Prophet MAE",
                            (
                                f"{prophet_mae:.2f}"
                                if (
                                    prophet_mae
                                    is not None
                                    and
                                    np.isfinite(
                                        prophet_mae
                                    )
                                )
                                else "N/A"
                            ),
                        )

                    else:

                        st.info(
                            "Prophet data unavailable."
                        )

                results = {
                    "ARIMA": arima_rmse,
                    "Linear": linear_rmse,
                }

                if (
                    prophet_rmse is not None
                    and
                    np.isfinite(
                        prophet_rmse
                    )
                ):

                    results[
                        "Prophet"
                    ] = prophet_rmse

                results = {
                    key: value
                    for key, value
                    in results.items()
                    if np.isfinite(value)
                }

                if results:

                    best_model = min(
                        results,
                        key=results.get,
                    )

                    st.success(
                        f"🏆 Most Reliable Prediction "
                        f"Method for {selected_state}: "
                        f"**{best_model}**"
                    )

                st.plotly_chart(
                    fig_compare,
                    use_container_width=True,
                )

            except Exception as exc:

                st.error(
                    f"⚠️ Model comparison failed: "
                    f"{exc}"
                )

    # -----------------------------------------------------------------
    # COMPONENT BREAKDOWN
    # -----------------------------------------------------------------

    st.subheader(
        f"🎯 What's Driving the Risk in "
        f"{selected_state}?"
    )

    st.caption(
        "Higher values indicate greater stress."
    )

    col_left, col_right = (
        st.columns(2)
    )

    with col_left:

        components_data = pd.DataFrame(
            {
                "Factor": [
                    "Climate Stress",
                    "Agriculture Stress",
                    "Economic Stress",
                ],

                "Score": [
                    climate_score,
                    agri_score,
                    econ_score,
                ],
            }
        )

        fig_components = px.bar(
            components_data,
            x="Factor",
            y="Score",
            color="Factor",
            color_discrete_map={
                "Climate Stress": "#3498db",
                "Agriculture Stress": "#2ecc71",
                "Economic Stress": "#e74c3c",
            },
            text="Score",
        )

        fig_components.update_traces(
            texttemplate="%{text:.1f}",
            textposition="outside",
        )

        fig_components.update_layout(
            height=360,
            showlegend=False,
            yaxis_title="Stress Score (0–100)",
            xaxis_title="",
            paper_bgcolor=(
                "rgba(0,0,0,0)"
            ),
            plot_bgcolor=(
                "rgba(0,0,0,0)"
            ),
            yaxis=dict(
                range=[0, 100],
                gridcolor="#F1F5F9",
            ),
        )

        st.plotly_chart(
            fig_components,
            use_container_width=True,
        )

    # -----------------------------------------------------------------
    # KEY INDICATORS
    # -----------------------------------------------------------------

    with col_right:

        st.markdown(
            "**📋 Key Indicators**"
        )

        st.caption(
            "Current environmental, agricultural "
            "and economic indicators for the "
            "selected state."
        )

        def indicator_value(
            column,
            suffix="",
            decimals=1,
        ):

            value = safe_float(
                state_current.get(
                    column,
                    0,
                )
            )

            return (
                f"{value:.{decimals}f}"
                f"{suffix}"
            )

        indicators = [

            (
                "🌡️",
                "Temperature Anomaly",
                indicator_value(
                    "temp_anomaly_deg",
                    "°C",
                ),
                "Temperature deviation from "
                "the seasonal normal.",
            ),

            (
                "🌧️",
                "Rainfall Anomaly",
                indicator_value(
                    "rainfall_anomaly_pct",
                    "%",
                ),
                "Percentage deviation from "
                "expected rainfall.",
            ),

            (
                "🔥",
                "Heatwave Days",
                indicator_value(
                    "heatwave_days",
                    " days",
                    0,
                ),
                "Number of days experiencing "
                "extreme heat conditions.",
            ),

            (
                "🌾",
                "Crop Failure",
                indicator_value(
                    "crop_failure_rate_pct",
                    "%",
                ),
                "Estimated percentage of crops "
                "affected or failed.",
            ),

            (
                "💼",
                "Unemployment",
                indicator_value(
                    "unemployment_rate_pct",
                    "%",
                ),
                "Estimated percentage of the "
                "labour force unemployed.",
            ),

            (
                "📈",
                "Inflation",
                indicator_value(
                    "inflation_rate_pct",
                    "%",
                ),
                "Estimated percentage change "
                "in prices.",
            ),
        ]

        pop_col1, pop_col2 = (
            st.columns(2)
        )

        for i, (
            icon,
            label,
            value,
            context,
        ) in enumerate(
            indicators
        ):

            target_col = (
                pop_col1
                if i % 2 == 0
                else pop_col2
            )

            with target_col:

                with st.popover(
                    f"{icon} {label}",
                    use_container_width=True,
                ):

                    st.markdown(
                        f"### {icon} {label}"
                    )

                    st.metric(
                        f"Current value in "
                        f"{selected_state}",
                        value,
                    )

                    st.markdown("---")

                    st.info(
                        f"**What this means:**\n\n"
                        f"{context}"
                    )

        # -------------------------------------------------------------
        # PERCENTAGE INDICATORS
        # -------------------------------------------------------------

        st.markdown(
            "#### 📊 Percentage-Based Indicators"
        )

        percentage_data = []

        percentage_columns = [

            (
                "Rainfall Anomaly",
                "rainfall_anomaly_pct",
            ),

            (
                "Crop Failure",
                "crop_failure_rate_pct",
            ),

            (
                "Unemployment",
                "unemployment_rate_pct",
            ),

            (
                "Inflation",
                "inflation_rate_pct",
            ),
        ]

        for label, column in (
            percentage_columns
        ):

            if column in state_current.index:

                value = safe_float(
                    state_current.get(
                        column
                    )
                )

                percentage_data.append(
                    {
                        "Indicator": label,
                        "Percentage": value,
                    }
                )

        if percentage_data:

            percentage_df = pd.DataFrame(
                percentage_data
            )

            fig_percentage = px.bar(
                percentage_df,
                x="Indicator",
                y="Percentage",
                text="Percentage",
            )

            fig_percentage.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside",
            )

            fig_percentage.update_layout(
                height=300,
                yaxis_title="Percentage (%)",
                xaxis_title="",
                showlegend=False,
                paper_bgcolor=(
                    "rgba(0,0,0,0)"
                ),
                plot_bgcolor=(
                    "rgba(0,0,0,0)"
                ),
                yaxis=dict(
                    gridcolor="#F1F5F9",
                ),
            )

            st.plotly_chart(
                fig_percentage,
                use_container_width=True,
            )

    # -----------------------------------------------------------------
    # FOOTER
    # -----------------------------------------------------------------

    st.markdown("---")

    st.html(
        """
        <div class="dash-footer">

            <div class="footer-top">

                <div>

                    <div class="footer-brand">
                        🌍 CrisisLens
                    </div>

                    <div class="footer-desc">
                        India's multi-factor crisis risk
                        intelligence platform.<br>
                        Built to support faster,
                        evidence-based disaster response
                        decisions.
                    </div>

                </div>

                <div style="
                    display:flex;
                    gap:2rem;
                    align-items:center;
                    flex-wrap:wrap;
                ">

                    <div class="footer-stat">

                        <div class="num">
                            36
                        </div>

                        <div class="lbl">
                            States & UTs
                        </div>

                    </div>

                    <div class="footer-stat">

                        <div class="num">
                            93.3%
                        </div>

                        <div class="lbl">
                            Alert Accuracy
                        </div>

                    </div>

                    <div class="footer-stat">

                        <div class="num">
                            3
                        </div>

                        <div class="lbl">
                            Forecast Models
                        </div>

                    </div>

                    <div class="footer-stat">

                        <div class="num">
                            90d
                        </div>

                        <div class="lbl">
                            Max Horizon
                        </div>

                    </div>

                </div>

            </div>

            <div class="footer-note">

                <strong>Data:</strong>
                Indicators are simulated from documented
                Indian climate and economic baselines
                with a fixed random seed for
                reproducibility.

                The forecasting pipeline is designed
                to support live government data feeds.

                · <strong>Backtested:</strong>
                CrisisLens historical evaluation.

                · Forecast models:
                ARIMA + Linear Trend + Prophet.

            </div>

        </div>
        """
    )


# =====================================================================
# RUN APP
# =====================================================================

if __name__ == "__main__":
    main()
