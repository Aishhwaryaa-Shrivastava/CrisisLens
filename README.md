# 🌍 CrisisLens

## Multi-Factor Crisis Risk Assessment and Forecasting for India


![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange)
![Prophet](https://img.shields.io/badge/Prophet-Forecasting-purple)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange)
![CrisisLens Dashboard](docs/images/hero-dashboard.png)

CrisisLens is a research-oriented crisis intelligence platform that combines **climate, agricultural, and economic indicators** to estimate regional stress, analyze historical risk patterns, forecast future risk, and provide interpretable machine-learning insights through an interactive dashboard.

> **Project Status:** Research / Academic Prototype
> **Scope:** India — 36 regions comprising 28 States and 8 Union Territories
> **Primary Interface:** Streamlit Dashboard
> **Risk Scale:** 0–100

---

# 📌 Overview

Crisis situations are rarely caused by a single factor. Environmental stress can affect agricultural productivity, agricultural losses can increase economic pressure, and economic vulnerability can reduce the ability of communities to absorb shocks.

CrisisLens addresses this problem by integrating three major dimensions:

```text
Climate Stress
      │
      ├──────────────┐
      │              │
      ▼              ▼
Agricultural Stress ──────► Economic Stress
      │                       │
      └──────────┬────────────┘
                 ▼
        Composite Risk Score
                 │
        ┌────────┴────────┐
        ▼                 ▼
   Trend Analysis     Forecasting
        │                 │
        └────────┬────────┘
                 ▼
       ML + Explainability
                 │
                 ▼
        Streamlit Dashboard
```

The objective is not to replace official disaster-management or government early-warning systems. Instead, CrisisLens demonstrates how heterogeneous environmental, agricultural, and socioeconomic indicators can be combined into a unified analytical framework.

---

# 🎯 Problem Statement

Traditional risk-monitoring approaches often examine climate, agriculture, and economic indicators independently.

However, regional vulnerability may emerge from the interaction of several factors.

For example:

```text
Rainfall Deficit / Extreme Weather
              ↓
        Climate Stress
              ↓
       Crop Stress / Failure
              ↓
     Agricultural Pressure
              ↓
      Economic Vulnerability
              ↓
     Increased Regional Risk
```

CrisisLens attempts to capture this interaction through a multi-factor risk index and complementary forecasting and machine-learning workflows.

---

# ✨ Key Features

### 🌦️ Climate Risk Analysis

CrisisLens incorporates:

* Average temperature
* Total rainfall
* Temperature anomaly
* Rainfall anomaly
* Heatwave days
* Drought severity

### 🌾 Agricultural Risk Analysis

The agricultural component includes:

* Crop production
* Irrigation coverage
* Crop failure rate
* Fertilizer consumption
* Land under cultivation

### 💰 Economic Risk Analysis

The economic component incorporates:

* GDP estimate
* Unemployment rate
* Inflation rate
* Poverty rate
* Per-capita income

### 📊 Risk Assessment

* Composite regional risk score
* Climate stress
* Agricultural stress
* Economic stress
* Risk categorization
* State/UT comparison
* Risk-driver analysis

### 📈 Trend Analysis

* Historical risk trends
* Monthly risk changes
* Component-level trends
* State-level comparisons

### 🔮 Forecasting

The project includes:

* Persistence / moving-average forecasting
* Linear-trend forecasting
* ARIMA implementation
* Prophet forecasting
* Short-term future risk estimation
* Forecast comparison

### 🤖 Machine Learning

* Random Forest Regression
* Feature importance
* Prediction examples
* Model evaluation
* Correlation analysis

### 🧠 Explainable AI

* SHAP analysis
* Feature-importance visualization
* Risk-driver identification
* Prediction interpretation

### 🧪 Validation

* Time-based backtesting workflow
* 7-day, 14-day and 30-day lead-time experiments
* Forecast-vs-observation comparison
* Severity classification evaluation
* Alert-decision evaluation

### 🖥️ Interactive Dashboard

The Streamlit application provides:

* Regional risk overview
* State/UT selection
* Climate/agriculture/economic breakdown
* Historical trends
* Forecasts
* Risk categories
* Risk-driver interpretation
* Interactive visualizations

---

# 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │    Climate Data      │
                    │ Temperature          │
                    │ Rainfall             │
                    │ Drought              │
                    │ Heatwaves            │
                    └──────────┬───────────┘
                               │
                               │
┌──────────────────────┐       │       ┌──────────────────────┐
│  Agriculture Data    │───────┼───────│   Economic Data      │
│ Crop Production      │       │       │ GDP                  │
│ Crop Failure         │       │       │ Unemployment         │
│ Irrigation           │       │       │ Inflation            │
└──────────────────────┘       │       │ Poverty              │
                               │       └──────────────────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Data Cleaning &      │
                    │ Standardization      │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Data Integration     │
                    │ Daily → Monthly      │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Risk Scoring Engine  │
                    └──────────┬───────────┘
                               ▼
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
     ┌──────────────────┐             ┌──────────────────┐
     │ Trend Analysis   │             │ ML Prediction    │
     └────────┬─────────┘             └────────┬─────────┘
              │                                │
              ▼                                ▼
     ┌──────────────────┐             ┌──────────────────┐
     │ Forecasting      │             │ SHAP / Feature   │
     │ ARIMA / Prophet  │             │ Importance       │
     └────────┬─────────┘             └────────┬─────────┘
              │                                │
              └──────────────┬─────────────────┘
                             ▼
                  ┌──────────────────────┐
                  │ Validation /         │
                  │ Backtesting          │
                  └──────────┬───────────┘
                             ▼
                  ┌──────────────────────┐
                  │ Streamlit Dashboard  │
                  └──────────────────────┘
```

---

# 📊 Risk Methodology

## 1. Indicator Selection

The system uses three major categories.

| Category       | Example Indicators                                        |
| -------------- | --------------------------------------------------------- |
| 🌦️ Climate    | Temperature anomaly, rainfall anomaly, drought, heatwaves |
| 🌾 Agriculture | Crop failure, irrigation, crop production                 |
| 💰 Economic    | Unemployment, inflation, poverty                          |

---

## 2. Normalization

Indicators have different units and scales.

For example:

```text
Temperature      → °C
Rainfall         → mm
Crop failure     → %
GDP              → Crores
Unemployment     → %
```

Therefore, the pipeline normalizes indicators before combining them.

The final merged pipeline uses normalized values and percentile-based category stress scores on a **0–100 scale**.

---

## 3. Category Construction

### Climate Stress

Climate stress is constructed using:

* Temperature anomaly
* Rainfall anomaly
* Heatwave days
* Drought severity

### Agricultural Stress

Agricultural stress incorporates:

* Crop failure
* Irrigation conditions
* Drought-related agricultural pressure

### Economic Stress

Economic stress incorporates:

* Unemployment
* Inflation
* Poverty

---

## 4. Composite Risk Score

The current integrated pipeline uses:

| Component   |  Weight |
| ----------- | ------: |
| Climate     | **40%** |
| Agriculture | **35%** |
| Economic    | **25%** |

The composite risk is calculated conceptually as:

```text
Total Risk =
    0.40 × Climate Stress
  + 0.35 × Agricultural Stress
  + 0.25 × Economic Stress
```

The resulting score ranges from approximately **0 to 100**.

---

## 5. Risk Categories

The Streamlit dashboard uses the following thresholds:

| Risk Score | Category    |
| ---------: | ----------- |
|     `< 30` | 🟢 Low      |
| `30 – <50` | 🟡 Medium   |
| `50 – <70` | 🟠 High     |
|     `≥ 70` | 🔴 Critical |

> These thresholds are **prototype/heuristic thresholds** and should be calibrated against independently verified historical crisis outcomes before operational deployment.

---

# 📈 Results

CrisisLens produces analytical outputs covering regional risk, risk drivers, temporal trends, forecasting, explainability, and forecast validation.

All generated analytical figures are available in:

```text
outputs/figures/
```

The README references these figures directly rather than duplicating them into `docs/images/`.

---

## 📊 Risk Category Distribution

The composite risk framework categorizes observations into Low, Medium, High, and Critical risk levels.

![Risk Category Distribution](outputs/figures/week2_risk_category_distribution.png)

---

## 🔝 Top Risk States

The top-risk analysis ranks regions according to their calculated composite risk score.

![Top Risk States](outputs/figures/week2_top10_risk_states.png)

---

## 🧩 Risk Component Breakdown

The risk breakdown illustrates the contribution of climate, agricultural, and economic components to the overall risk score.

![Risk Breakdown](outputs/figures/week2_risk_breakdown_stacked.png)

---

## 🔗 Risk Correlation

The correlation analysis examines relationships among the major risk components.

![Risk Correlation](outputs/figures/week2_risk_correlation_heatmap.png)

---

## 📈 Historical Risk Trends

Trend analysis tracks how risk changes across high-risk regions over time.

![Risk Trends](outputs/figures/week3_trend_analysis_top_states.png)

---

# 🌦️ Climate and Agricultural Analysis

The project also investigates relationships between climate conditions and agricultural stress.

One of the key exploratory analyses examines drought severity in relation to crop failure.

![Drought vs Crop Failure](outputs/figures/week1_drought_vs_crop_failure.png)

This analysis supports the broader conceptual relationship:

```text
Climate Stress
      ↓
Drought / Rainfall Stress
      ↓
Agricultural Stress
      ↓
Crop Failure
      ↓
Increased Regional Risk
```

Additional exploratory figures remain available in:

```text
outputs/figures/
```

including:

```text
week1_data_quality_overview.png
week1_top_10_hottest_states.png
```

These are retained as supporting project outputs rather than being placed prominently in the main README.

---

# 🔮 Forecasting

CrisisLens evaluates multiple forecasting approaches, including:

* Persistence
* Moving average
* Linear trend
* ARIMA
* Prophet

The objective is to estimate future regional risk while comparing different forecasting strategies.

---

## Prophet Forecast

Prophet is used to model temporal trends and seasonality and provides forecast intervals.

![Prophet Forecast](outputs/figures/week3_prophet_forecasts_top5.png)

---

## Forecast Model Comparison

Different forecasting approaches are compared using forecast evaluation metrics.

![Forecast Model Comparison](outputs/figures/week3_model_comparison.png)

The project's reported prototype comparison is:

| Model          | Typical MAPE | Seasonality | Confidence Intervals | Complexity |
| -------------- | -----------: | ----------- | -------------------- | ---------- |
| Persistence    |       24–28% | No          | No                   | Very Low   |
| Moving Average |       19–22% | Partial     | No                   | Low        |
| Linear Trend   |       15–18% | No          | No                   | Low        |
| Prophet        |       11–14% | Yes         | Yes                  | Medium     |

> These are the project's reported/typical comparison ranges rather than independently reproduced benchmark results. They should be treated as prototype evaluation results.

---

# 🤖 Machine Learning

## Random Forest Regression

CrisisLens includes a Random Forest regression workflow to model the relationship between the available indicators and the composite `total_risk` index.

### Model Configuration

```text
Algorithm          : Random Forest Regressor
Number of Trees    : 100
Maximum Depth      : 10
Minimum Split      : 5
Minimum Leaf       : 2
Train/Test Split   : 80% / 20%
Random State       : 42
```

The model uses climate, agricultural, and economic indicators as input features.

### Important Interpretation

The current ML experiment predicts the **CrisisLens composite risk index**, which is itself constructed from the underlying risk indicators.

Therefore:

> The Random Forest experiment demonstrates nonlinear prediction and explainability of the constructed risk index. It should not be interpreted as an independently validated model of real-world crisis occurrence.

---

# 📊 Machine Learning Results

The current Random Forest experiment produced:

| Metric           |               Result |
| ---------------- | -------------------: |
| Records          |              **864** |
| Features         |               **16** |
| Training samples |              **691** |
| Test samples     |              **173** |
| Test R²          |            **0.996** |
| Test MAE         | **0.64 risk points** |

The high R² should be interpreted carefully because the target variable is a deterministic composite risk index derived from related input indicators.

---

# 🧠 Explainable AI

CrisisLens incorporates feature-importance analysis and SHAP to investigate which variables contribute most strongly to machine-learning predictions.

---

## Feature Importance

The current Random Forest experiment identified the following leading variables:

| Rank | Feature             | Importance |
| ---: | ------------------- | ---------: |
|    1 | Rainfall anomaly    | **68.29%** |
|    2 | Crop failure rate   | **14.67%** |
|    3 | Drought severity    | **14.60%** |
|    4 | Crop production     |  **1.47%** |
|    5 | Temperature anomaly |  **0.34%** |

![Feature Importance](outputs/figures/week4_feature_importance.png)

The current experiment indicates that rainfall anomaly, crop failure, and drought severity dominate the model's predictions.

These values describe **model feature importance**, not causal influence.

---

## Category-Level Feature Importance

The model experiment reported approximately:

| Category    | Importance |
| ----------- | ---------: |
| Climate     |  **83.5%** |
| Agriculture |  **16.2%** |
| Economic    |   **0.4%** |

These values describe **model feature importance**, not scientific causal influence.

They should therefore be interpreted as:

> Which variables were most useful to this model when reproducing the constructed risk index?

rather than:

> Which variables scientifically cause crises?

---

## SHAP Analysis

SHAP provides a more detailed view of how individual features influence model predictions.

The SHAP workflow helps answer:

```text
Why did the model produce this prediction?
```

The analysis follows:

```text
Input Features
      ↓
Random Forest
      ↓
Predicted Risk
      ↓
SHAP Values
      ↓
Positive / Negative Feature Contributions
```

![SHAP Summary](outputs/figures/week4_shap_summary.png)

The generated SHAP analysis is also available through:

```text
outputs/reports/
```

---

# 🧪 Validation & Backtesting

CrisisLens includes a time-based backtesting workflow that evaluates forecasts at:

* 7-day lead time
* 14-day lead time
* 30-day lead time

The workflow simulates how the system would behave when making predictions before a future observation.

```text
Historical Data
      ↓
Training Window
      ↓
Prediction Date
      ↓
7 / 14 / 30 Day Forecast
      ↓
Compare with Later Observation
      ↓
Calculate Error
      ↓
Repeat
```

The validation workflow evaluates:

* Mean Absolute Error
* Percentage error
* Severity classification
* Alert decisions
* Performance by lead time

---

## Forecast vs Reality

![Forecast Reality Check](outputs/figures/forecast_reality_check.png)

---

## Seasonal Validation

![Seasonal Validation](outputs/figures/seasonal_validation.png)

---

## Backtesting Validation

![Backtesting Validation](outputs/week4_backtesting_validation.png)

---

## Backtesting Summary

The current prototype backtesting experiment reports:

| Metric                           |             Result |
| -------------------------------- | -----------------: |
| Scenarios evaluated              |              **5** |
| Lead times                       | **7, 14, 30 days** |
| Average absolute error           |    **2.66 points** |
| Average percentage error         |          **5.37%** |
| Severity classification accuracy |          **66.7%** |
| Alert decision accuracy          |          **93.3%** |

### Performance by Lead Time

| Lead Time | Average Error | Severity Accuracy |
| --------: | ------------: | ----------------: |
|    7 days |      **1.93** |           **80%** |
|   14 days |      **2.36** |           **60%** |
|   30 days |      **3.70** |           **60%** |

The results show increasing average forecast error at longer prediction horizons.

> These are **prototype validation results**. They should not be interpreted as independent operational validation because the underlying project datasets include generated/simulated data and the current event definitions are limited.

---

# 📋 Validation Interpretation

The backtesting results demonstrate the intended validation workflow:

```text
Short Horizon
     │
     ▼
Lower Forecast Error
     │
     ▼
Higher Severity Accuracy
```

while:

```text
Longer Horizon
     │
     ▼
Greater Forecast Uncertainty
     │
     ▼
Higher Prediction Error
```

The results therefore demonstrate the importance of considering forecast horizon when interpreting crisis-risk predictions.

---

# 🖥️ Dashboard

CrisisLens provides an interactive Streamlit dashboard for exploring the analytical results.

The dashboard supports:

* Regional risk overview
* Risk score visualization
* Risk category identification
* Climate stress analysis
* Agricultural stress analysis
* Economic stress analysis
* State/UT comparison
* Historical risk trends
* Forecast visualization
* Forecast comparison
* Interactive charts
* Risk-driver interpretation

The dashboard is implemented in:

```text
src/app_dashboard.py
```

---

# 📸 Dashboard Screenshots

If screenshots of the actual Streamlit application are later added, they can be stored in:

```text
docs/images/
```

Recommended screenshots:

```text
docs/images/
├── dashboard-overview.png
├── state-risk-analysis.png
├── risk-forecast.png
├── risk-trends.png
└── model-explainability.png
```

They can then be embedded using:

```markdown
![CrisisLens Dashboard](docs/images/dashboard-overview.png)

![State Risk Analysis](docs/images/state-risk-analysis.png)

![Risk Forecast](docs/images/risk-forecast.png)

![Model Explainability](docs/images/model-explainability.png)
```

> These screenshots are optional. The analytical result figures already exist under `outputs/figures/` and are directly referenced by this README.

---

# 📂 Data

## Data Dimensions

CrisisLens works with three primary data domains.

### Climate

```text
avg_temperature
total_rainfall_mm
temp_anomaly_deg
rainfall_anomaly_pct
heatwave_days
drought_severity
```

### Agriculture

```text
crop_production_1000t
irrigation_coverage_pct
crop_failure_rate_pct
fertilizer_consumption_kg_per_ha
land_under_cultivation_1000ha
```

### Economic

```text
gdp_estimate_crores
unemployment_rate_pct
inflation_rate_pct
poverty_rate_pct
per_capita_income_inr
```

---

# ⚠️ Data Provenance

The current repository is an **academic research prototype**.

The available data-generation pipeline contains simulated/generated values designed to reproduce realistic-looking climate, agricultural, and economic patterns.

For example, the repository contains generators for:

```text
2024 agricultural data
2024 climate data
2024 economic data
2025 regional data
```

Therefore:

> **The current results should not be interpreted as official measurements of Indian state-level crisis conditions.**

For an operational system, the project should be connected to independently verified government, scientific, satellite, meteorological, agricultural, and economic datasets.

---

# 📅 Dataset Coverage

The integrated dataset contains approximately:

```text
36 regions
2024–2025 monthly observations
Climate indicators
Agricultural indicators
Economic indicators
Composite risk scores
```

The merged monthly dataset is:

```text
data/processed/combined_monthly_2024_2025.csv
```

The daily integrated dataset is:

```text
data/processed/combined_daily_2024_2025.csv
```

---

# 🔄 Data Pipeline

```text
Raw Data
   ↓
Data Cleaning
   ↓
Standardization
   ↓
Daily Integration
   ↓
Monthly Aggregation
   ↓
Indicator Normalization
   ↓
Category Stress Scores
   ↓
Composite Risk Score
   ↓
Trend Analysis
   ↓
Forecasting
   ↓
Machine Learning
   ↓
Explainability
   ↓
Validation
   ↓
Dashboard
```

---

# 🔐 Dashboard Authentication

The current dashboard contains a **demonstration authentication mechanism**.

It is intended only for prototype demonstration.

> **Never use the current hard-coded demonstration credentials in production.**

Production deployment should use:

* Password hashing
* Environment variables
* Secrets management
* Secure session management
* Role-based access control
* Credential rotation
* Multi-factor authentication where appropriate

---

# ⚡ Quick Start

## 1. Clone the Repository

```bash
git clone https://github.com/Aishhwaryaa-Shrivastava/CrisisLens.git
cd CrisisLens
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

The project's dependency file is located at:

```text
src/requirements.txt
```

Install using:

```bash
pip install -r src/requirements.txt
```

The main libraries include:

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
streamlit
prophet
shap
jupyter
notebook
openpyxl
requests
tqdm
```

---

# ▶️ Run the Dashboard

From the project root:

```bash
streamlit run src/app_dashboard.py
```

Streamlit will provide a local address, normally:

```text
http://localhost:8501
```

Open that address in a browser.

---

# 🧪 Running the Analytical Workflow

The notebooks are organized according to the project lifecycle.

| Notebook                                     | Purpose                   |
| -------------------------------------------- | ------------------------- |
| `01_data_exploration.ipynb`                  | Explore raw data          |
| `02_data_cleaning.ipynb`                     | Clean and prepare data    |
| `03_risk_assessment.ipynb`                   | Calculate risk components |
| `04_trend_analysis.ipynb`                    | Analyze historical trends |
| `05_simple_forecasting.ipynb`                | Baseline forecasting      |
| `06_prophet_forecasting_FIXED.ipynb`         | Prophet forecasting       |
| `07_feature_importance_explainability.ipynb` | Random Forest + SHAP      |
| `08_backtesting_validation.ipynb`            | Forecast validation       |

---

# 📁 Project Structure

```text
CrisisLens/
│
├── data/
│   ├── raw/
│   │   ├── agriculture_data_2024.csv
│   │   ├── climate_data_2024.csv
│   │   └── economic_data_2024.csv
│   │
│   └── processed/
│       ├── combined_daily_2024_2025.csv
│       ├── combined_monthly_2024_2025.csv
│       ├── combined_monthly_2024_2025_FIXED.csv
│       ├── master_dataset.csv
│       └── master_dataset_with_risks.csv
│
├── models/
│   └── arima_model.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_risk_assessment.ipynb
│   ├── 04_trend_analysis.ipynb
│   ├── 05_simple_forecasting.ipynb
│   ├── 06_prophet_forecasting_FIXED.ipynb
│   ├── 07_feature_importance_explainability.ipynb
│   └── 08_backtesting_validation.ipynb
│
├── outputs/
│   ├── figures/
│   │   ├── week1_data_quality_overview.png
│   │   ├── week1_top_10_hottest_states.png
│   │   ├── week1_drought_vs_crop_failure.png
│   │   ├── week2_risk_category_distribution.png
│   │   ├── week2_top10_risk_states.png
│   │   ├── week2_risk_breakdown_stacked.png
│   │   ├── week2_risk_correlation_heatmap.png
│   │   ├── week3_trend_analysis_top_states.png
│   │   ├── week3_simple_forecast_top5.png
│   │   ├── week3_prophet_forecasts_top5.png
│   │   ├── week3_model_comparison.png
│   │   ├── week4_feature_importance.png
│   │   ├── week4_shap_summary.png
│   │   ├── week4_correlation_matrix.png
│   │   ├── forecast_reality_check.png
│   │   └── seasonal_validation.png
│   │
│   ├── reports/
│   │   ├── risk_assessment_report.txt
│   │   ├── state_risk_summary.csv
│   │   ├── high_risk_states.csv
│   │   ├── simple_forecasts_2026.csv
│   │   ├── prophet_forecasts_2026.csv
│   │   ├── model_comparison.csv
│   │   ├── forecast_validation_results.csv
│   │   ├── feature_importance.csv
│   │   └── prediction_examples.csv
│   │
│   ├── backtesting_results.csv
│   ├── week4_backtesting_report.txt
│   └── week4_backtesting_validation.png
│
├── src/
│   ├── app_dashboard.py
│   ├── download_2025_data.py
│   ├── merge_data.py
│   ├── fix_dashboard.py
│   ├── fix_monthly_data.py
│   ├── test_prophet.py
│   └── requirements.txt
│
├── download_data.py
├── LICENSE
└── README.md
```

> Development/debugging files and Jupyter checkpoint directories may be removed from the final submission repository for a cleaner academic presentation.

---

# 🛠️ Technology Stack

| Technology           | Purpose                      |
| -------------------- | ---------------------------- |
| **Python**           | Core development             |
| **Pandas**           | Data manipulation            |
| **NumPy**            | Numerical computation        |
| **SciPy**            | Scientific computation       |
| **Matplotlib**       | Visualization                |
| **Seaborn**          | Statistical visualization    |
| **Plotly**           | Interactive visualization    |
| **Scikit-learn**     | Machine learning             |
| **Statsmodels**      | Statistical modeling / ARIMA |
| **Prophet**          | Time-series forecasting      |
| **SHAP**             | Explainable AI               |
| **Streamlit**        | Interactive dashboard        |
| **Jupyter Notebook** | Experimentation and analysis |
| **Power BI**         | Additional reporting         |

---

# 📊 Project Outputs

## Risk Assessment

```text
outputs/reports/risk_assessment_report.txt
outputs/reports/state_risk_summary.csv
outputs/reports/high_risk_states.csv
```

## Forecasting

```text
outputs/reports/simple_forecasts_2026.csv
outputs/reports/prophet_forecasts_2026.csv
outputs/reports/model_comparison.csv
outputs/reports/forecast_validation_results.csv
```

## Machine Learning

```text
outputs/reports/feature_importance.csv
outputs/reports/prediction_examples.csv
```

## Validation

```text
outputs/backtesting_results.csv
outputs/week4_backtesting_report.txt
outputs/week4_backtesting_validation.png
```

---

# 📌 Key Findings from the Current Prototype

## 1. Climate indicators dominate the current ML experiment

Rainfall anomaly is the most important feature in the Random Forest experiment, followed by:

```text
Rainfall anomaly
      ↓
Crop failure
      ↓
Drought severity
```

This is consistent with the project's intended climate–agriculture interaction.

---

## 2. The risk model is multi-dimensional

The system does not rely on climate indicators alone.

The final risk framework incorporates:

```text
Climate
+
Agriculture
+
Economy
```

---

## 3. Forecast uncertainty increases with prediction horizon

The backtesting experiment shows larger average errors at longer lead times:

```text
7 days   → 1.93 points
14 days  → 2.36 points
30 days  → 3.70 points
```

This demonstrates why longer-horizon forecasts should be interpreted with greater uncertainty.

---

## 4. Explainability is integrated into the workflow

CrisisLens does not only generate a risk score.

It also investigates:

```text
Which features influence the model?
Which category dominates?
Why did a particular prediction occur?
```

---

# ⚠️ Limitations

CrisisLens is currently an **academic/research prototype** and has important limitations.

## 1. Simulated Data

The current data-generation pipeline contains simulated values.

Consequently, the model's performance cannot be treated as evidence of real-world operational accuracy.

## 2. Risk Weights Are Prototype Assumptions

The current weighting scheme:

```text
Climate       40%
Agriculture   35%
Economic      25%
```

is a methodological prototype and requires validation against independently observed outcomes.

## 3. Risk Categories Are Heuristic

The thresholds:

```text
0–30       Low
30–50      Medium
50–70      High
70–100     Critical
```

should be calibrated using historical crisis outcomes before operational use.

## 4. ML Target Construction

The current Random Forest model predicts the constructed `total_risk` index.

Because the target is derived from the same underlying indicators used as model inputs, the very high R² should not be interpreted as evidence that the model can independently predict real-world crisis events.

## 5. Limited Historical Coverage

The current integrated dataset provides a relatively short historical period.

Longer historical datasets would improve:

* Seasonal modeling
* Trend detection
* Forecast validation
* Extreme-event analysis

## 6. Event Validation

The current backtesting workflow uses a limited number of predefined event scenarios.

A stronger evaluation would use a large, independently verified historical crisis-event dataset.

## 7. State-Level Aggregation

State-level analysis can hide important local variation.

District-level and sub-district-level modeling would provide more detailed risk information.

## 8. Correlation Is Not Causation

Feature importance and SHAP explain model behavior.

They do not establish causal relationships between indicators and crisis events.

---

# 🔮 Future Work

### Data

* [ ] Integrate verified government datasets
* [ ] Integrate meteorological observations
* [ ] Integrate satellite-derived indicators
* [ ] Integrate agricultural statistics
* [ ] Integrate socioeconomic datasets
* [ ] Automate data ingestion
* [ ] Improve data-quality monitoring

### Risk Modeling

* [ ] Empirically calibrate category weights
* [ ] Calibrate risk thresholds against historical events
* [ ] Add district-level risk assessment
* [ ] Add uncertainty estimation
* [ ] Develop probabilistic risk scores

### Machine Learning

* [ ] Add Gradient Boosting
* [ ] Add XGBoost / LightGBM
* [ ] Compare multiple ML algorithms
* [ ] Introduce temporal validation
* [ ] Introduce independently observed crisis labels
* [ ] Develop true crisis-event classification models

### Forecasting

* [ ] Improve ARIMA tuning
* [ ] Improve Prophet tuning
* [ ] Add probabilistic forecasting
* [ ] Add confidence / credible intervals
* [ ] Add longer forecast horizons
* [ ] Compare against stronger time-series baselines

### Explainable AI

* [ ] Improve local SHAP explanations
* [ ] Add dashboard-level explanations
* [ ] Add feature-interaction analysis
* [ ] Distinguish predictive importance from causal interpretation

### Deployment

* [ ] Replace demonstration authentication
* [ ] Add secure secrets management
* [ ] Add role-based access control
* [ ] Add automated model retraining
* [ ] Add CI/CD
* [ ] Add unit and integration tests
* [ ] Containerize the application
* [ ] Deploy the dashboard to a cloud platform

---

# 🎓 Academic Significance

CrisisLens demonstrates the integration of several data-science disciplines within a single analytical system:

```text
Data Engineering
       +
Exploratory Data Analysis
       +
Feature Engineering
       +
Risk Modeling
       +
Machine Learning
       +
Time-Series Forecasting
       +
Explainable AI
       +
Data Visualization
       +
Interactive Application Development
```

The project therefore serves as a practical example of how heterogeneous datasets can be transformed into a unified analytical decision-support framework.

---

# 🔬 Research Contribution

The primary contribution of CrisisLens is the design of a **multi-factor regional risk framework** that combines:

1. Climate stress
2. Agricultural stress
3. Economic vulnerability
4. Composite risk scoring
5. Temporal forecasting
6. Machine-learning-based risk modeling
7. Explainable AI
8. Backtesting and validation
9. Interactive visualization

The framework can be extended with verified real-world data and historical crisis labels for more rigorous research.

---

# 📚 References

The implementation makes use of established methodologies and software libraries including:

* Random Forest regression
* SHAP explainability
* ARIMA time-series modeling
* Prophet forecasting
* Min-Max normalization
* Time-series backtesting
* Interactive visualization with Plotly and Streamlit

For an academic submission, the final project report should provide formal citations for:

* Random Forest methodology
* SHAP
* Prophet
* ARIMA
* The datasets used
* Any government or scientific data sources added in future versions

---

# 🤝 Contributing

Contributions are welcome.

## 1. Fork the Repository

Create a personal copy of the project.

## 2. Create a Feature Branch

```bash
git checkout -b feature/your-feature
```

## 3. Implement and Test Your Changes

Ensure that the dashboard and analytical workflow continue to function correctly.

## 4. Commit Your Changes

```bash
git add .
git commit -m "Add new feature"
```

## 5. Push the Branch

```bash
git push origin feature/your-feature
```

## 6. Submit a Pull Request

Describe:

* What was changed
* Why it was changed
* How it was tested
* Any limitations introduced

---

# 📜 License

CrisisLens is released under the **MIT License**.

See the `LICENSE` file for the complete license terms.

---

# 📬 Contact

**Project:** CrisisLens

**Repository:** [CrisisLens](https://github.com/Aishhwaryaa-Shrivastava/CrisisLens)

For questions, suggestions, academic discussion, or collaboration, please open an issue or submit a pull request.

---

# ⭐ Project Summary

CrisisLens demonstrates how environmental, agricultural, and economic indicators can be integrated into a unified regional risk-analysis pipeline.

The complete workflow is:

```text
              DATA
               │
      ┌────────┼────────┐
      ▼        ▼        ▼
   Climate  Agriculture Economy
      │        │        │
      └────────┼────────┘
               ▼
       Data Integration
               │
               ▼
       Risk Score Engine
               │
       ┌───────┼────────┐
       ▼       ▼        ▼
     Trends  Forecast   ML
                       │
                       ▼
                     SHAP
                       │
       ┌───────────────┘
       ▼
   Validation
       │
       ▼
   Dashboard
```

### CrisisLens

> **From multi-source data to interpretable regional crisis intelligence.**

---

# ⚠️ Academic Prototype Disclaimer

CrisisLens is a **research and academic prototype**.

Its current datasets include simulated/generated data, and its risk scores and forecasts are analytical estimates rather than official warnings.

The system must **not** be used as a substitute for government disaster-management systems, meteorological warnings, agricultural advisories, emergency services, or official early-warning mechanisms.

Before operational deployment, the framework requires:

* Verified real-world data
* Independent historical crisis labels
* Rigorous temporal validation
* Empirical calibration of risk weights
* Uncertainty estimation
* Independent model evaluation
* Security and reliability testing
* Domain-expert review

---

# 📬 Contact

**Project:** CrisisLens

**Repository:** https://github.com/Aishhwaryaa-Shrivastava/CrisisLens

For questions, suggestions, or collaboration, open an issue or submit a pull request.
