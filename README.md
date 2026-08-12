# 🌍 CrisisLens — Multi-Factor Crisis Risk Forecasting System

> **An AI-powered crisis intelligence and forecasting platform that combines climate, agricultural, and economic indicators to assess and forecast regional crisis risk across India.**

---

## 📌 Overview

**CrisisLens** is a data-driven crisis risk assessment and forecasting platform designed to identify regions that may be vulnerable to emerging climate, agricultural, and economic stress.

The system combines multiple dimensions of risk into a unified analytical framework and provides:

* 📊 Regional crisis-risk assessment
* 🌦️ Climate stress analysis
* 🌾 Agricultural stress analysis
* 💰 Economic vulnerability analysis
* 📈 Historical trend analysis
* 🔮 Future risk forecasting
* 🤖 Machine-learning-based risk prediction
* 🧠 Explainable AI using SHAP
* 🧪 Forecast validation and backtesting
* 🖥️ Interactive Streamlit dashboard

The goal is to transform complex multi-source datasets into actionable insights that can support **risk monitoring, early warning, planning, and decision-making**.

> ⚠️ **Project Status:** CrisisLens is currently a research/prototype system. Some datasets and generated outputs may use simulated data and should not be treated as operational government or emergency-warning information.

---

# 🎯 Problem Statement

Crisis events are rarely caused by a single factor.

For example:

```text
Extreme Weather
      ↓
Agricultural Stress
      ↓
Economic Pressure
      ↓
Increased Regional Vulnerability
      ↓
Potential Crisis
```

Traditional monitoring systems often examine climate, agriculture, or economic indicators separately.

**CrisisLens combines these dimensions into one framework** to provide a more comprehensive view of regional risk.

---

# 🚀 Key Features

## 🌦️ Climate Risk Analysis

CrisisLens analyzes climate-related indicators such as:

* Temperature
* Rainfall
* Rainfall anomalies
* Drought indicators
* Heatwave conditions
* Climate stress

---

## 🌾 Agricultural Risk Analysis

The platform evaluates agricultural vulnerability using indicators such as:

* Crop production
* Crop failure
* Cultivated area
* Irrigation coverage
* Agricultural stress

---

## 💰 Economic Risk Analysis

Economic vulnerability is incorporated through indicators such as:

* GDP
* Income
* Inflation
* Unemployment
* Poverty
* Economic stress

---

# 📊 Multi-Factor Risk Score

The system combines multiple stress components into an overall crisis-risk score.

### Current Dashboard Weights

| Component   | Weight |
| ----------- | -----: |
| Climate     |    35% |
| Agriculture |    35% |
| Economic    |    30% |

### Risk Categories

|        Score | Category    |
| -----------: | ----------- |
|       `< 30` | 🟢 Low      |
| `30 – 49.99` | 🟡 Medium   |
| `50 – 69.99` | 🟠 High     |
|       `≥ 70` | 🔴 Critical |

> **Note:** These weights and thresholds represent the current prototype methodology and should be empirically validated before operational deployment.

---

# 🤖 Machine Learning

CrisisLens uses machine learning to investigate relationships between environmental, agricultural, and economic indicators and overall crisis risk.

### Machine Learning Components

* Random Forest
* Feature importance analysis
* SHAP explainability
* Prediction examples
* Model evaluation
* Correlation analysis

---

# 🧠 Explainable AI

An important component of CrisisLens is **model interpretability**.

Instead of only predicting risk, the system attempts to answer:

> **"Why is this region considered high risk?"**

## Feature Importance

The feature-importance workflow identifies variables that contribute most strongly to model predictions.

```text
Feature
   ↓
Machine Learning Model
   ↓
Feature Importance
   ↓
Identify Major Risk Drivers
```

## SHAP

SHAP is used to understand how individual features influence model predictions.

### SHAP Summary

![SHAP Summary](outputs/figures/week4_shap_summary.png)

---

# 🔮 Forecasting

CrisisLens includes several forecasting approaches.

## Forecasting Models

* Simple baseline forecasting
* ARIMA
* Prophet

Forecasting horizons include:

* 30 days
* 60 days
* 90 days

The project compares forecasting approaches and evaluates predictions against historical observations.

---

# 🧪 Model Validation & Backtesting

The project includes a dedicated backtesting workflow to evaluate forecasting performance against historical observations.

### Important Outputs

```text
outputs/reports/model_comparison.csv
outputs/reports/forecast_validation_results.csv
outputs/reports/backtesting_results.csv
outputs/reports/week4_backtesting_report.txt
```

### Validation Visualizations

```text
outputs/figures/forecast_reality_check.png
outputs/figures/seasonal_validation.png
outputs/figures/week4_backtesting_validation.png
```

These outputs help evaluate whether the forecasting models are able to reproduce historical patterns.

---

# 🖥️ Project Demo

## 🎬 Demo Video

### YouTube

Replace `YOUR_VIDEO_ID` with the actual YouTube video ID:

```markdown
[![CrisisLens Demo](https://img.youtube.com/vi/YOUR_VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)
```

### GitHub Video

If the demo is uploaded directly to GitHub, add the generated GitHub video asset URL:

```text
https://github.com/Aishhwaryaa-Shrivastava/CrisisLens/assets/YOUR_VIDEO_ID
```

---

# 📸 Dashboard Screenshots

Create the following directory:

```text
docs/
└── images/
```

Recommended structure:

```text
docs/
└── images/
    ├── dashboard-overview.png
    ├── state-risk-analysis.png
    ├── risk-forecast.png
    ├── risk-trends.png
    └── model-explainability.png
```

## Dashboard Overview

![CrisisLens Dashboard](docs/images/dashboard-overview.png)

## State Risk Analysis

![State Risk Analysis](docs/images/state-risk-analysis.png)

## Risk Forecast

![Risk Forecast](docs/images/risk-forecast.png)

## Risk Trends

![Risk Trends](docs/images/risk-trends.png)

## Model Explainability

![Model Explainability](docs/images/model-explainability.png)

---

# 📊 Project Outputs & ML Results

Generated figures are stored in:

```text
outputs/figures/
```

---

## 🌡️ Data Quality Analysis

The data-quality analysis provides an overview of the datasets used by CrisisLens.

### Data Quality Overview

![Data Quality Overview](outputs/figures/week1_data_quality_overview.png)

### 🔥 Top 10 Hottest States

This visualization compares temperature-related conditions across states.

![Top 10 Hottest States](outputs/figures/week1_top_10_hottest_states.png)

### 🌾 Drought vs Crop Failure

This analysis investigates the relationship between drought conditions and agricultural outcomes.

![Drought vs Crop Failure](outputs/figures/week1_drought_vs_crop_failure.png)

---

# 📈 Risk Analysis

## Risk Category Distribution

This visualization shows the distribution of regions across different crisis-risk categories.

![Risk Category Distribution](outputs/figures/week2_risk_category_distribution.png)

## Top 10 Risk States

This analysis identifies the highest-risk states according to the calculated risk score.

![Top 10 Risk States](outputs/figures/week2_top10_risk_states.png)

## Risk Correlation Heatmap

The correlation analysis helps identify relationships between different risk indicators.

![Risk Correlation Heatmap](outputs/figures/week2_risk_correlation_heatmap.png)

## Risk Breakdown

This visualization breaks down the contribution of different risk components.

![Risk Breakdown](outputs/figures/week2_risk_breakdown_stacked.png)

---

# 📉 Trend Analysis

## Top Risk States Over Time

This visualization tracks how risk changes across the highest-risk states.

![Top Risk States Over Time](outputs/figures/week3_trend_analysis_top_states.png)

## Monthly Risk Change

Monthly changes are analyzed to identify regions experiencing increasing or decreasing risk.

![Monthly Risk Change](outputs/figures/week3_monthly_change_rates.png)

## Component Trends

This visualization compares the evolution of individual risk components over time.

![Component Trends](outputs/figures/week3_component_trends_top5.png)

---

# 🔮 Forecasting Results

## Simple Forecast

Baseline forecasting is used as a reference for comparing more advanced forecasting methods.

![Simple Forecast](outputs/figures/week3_simple_forecast_top5.png)

## Prophet Forecast

Prophet is used to model temporal patterns and generate future risk estimates.

![Prophet Forecast](outputs/figures/week3_prophet_forecasts_top5.png)

## Model Comparison

Different forecasting approaches are compared using validation metrics.

![Model Comparison](outputs/figures/week3_model_comparison.png)

---

# 🧠 ML Model Explainability

## Feature Importance

Feature importance analysis helps identify which variables contribute most strongly to model predictions.

![Feature Importance](outputs/figures/week4_feature_importance.png)

## SHAP Summary

SHAP analysis provides a more detailed explanation of how individual features influence predictions.

![SHAP Summary](outputs/figures/week4_shap_summary.png)

## Correlation Matrix

The correlation matrix provides an overview of relationships between model features.

![Correlation Matrix](outputs/figures/week4_correlation_matrix.png)

---

# 🧪 Forecast Validation

## Forecast vs Reality

Forecast predictions are compared with observed historical values.

![Forecast vs Reality](outputs/figures/forecast_reality_check.png)

## Seasonal Validation

The forecasting pipeline is evaluated against historical seasonal patterns.

![Seasonal Validation](outputs/figures/seasonal_validation.png)

## Backtesting

Backtesting evaluates forecasting performance using historical observations.

![Backtesting Validation](outputs/figures/week4_backtesting_validation.png)

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
│   │   ├── week2_risk_correlation_heatmap.png
│   │   ├── week2_risk_breakdown_stacked.png
│   │   ├── week3_trend_analysis_top_states.png
│   │   ├── week3_component_trends_top5.png
│   │   ├── week3_monthly_change_rates.png
│   │   ├── week3_simple_forecast_top5.png
│   │   ├── week3_prophet_forecasts_top5.png
│   │   ├── week3_model_comparison.png
│   │   ├── week4_feature_importance.png
│   │   ├── week4_shap_summary.png
│   │   ├── week4_correlation_matrix.png
│   │   ├── forecast_reality_check.png
│   │   ├── seasonal_validation.png
│   │   └── ...
│   │
│   └── reports/
│       ├── risk_assessment_report.txt
│       ├── feature_importance.csv
│       ├── model_comparison.csv
│       ├── forecast_validation_results.csv
│       ├── prediction_examples.csv
│       ├── state_risk_summary.csv
│       ├── high_risk_states.csv
│       ├── prophet_forecasts_2026.csv
│       ├── simple_forecasts_2026.csv
│       ├── backtesting_results.csv
│       └── week4_backtesting_report.txt
│
├── src/
│   ├── app_dashboard.py
│   ├── merge_data.py
│   ├── download_2025_data.py
│   ├── fix_monthly_data.py
│   ├── fix_dashboard.py
│   ├── test_prophet.py
│   └── 08_backtesting_validation_DEBUG.py
│
├── download_data.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

# 🛠️ Technologies Used

| Technology           | Purpose                        |
| -------------------- | ------------------------------ |
| **Python**           | Core programming language      |
| **Pandas**           | Data processing                |
| **NumPy**            | Numerical computation          |
| **SciPy**            | Scientific computing           |
| **Matplotlib**       | Data visualization             |
| **Seaborn**          | Statistical visualization      |
| **Plotly**           | Interactive visualization      |
| **Scikit-learn**     | Machine learning               |
| **Statsmodels**      | Statistical modeling and ARIMA |
| **Prophet**          | Time-series forecasting        |
| **SHAP**             | Model explainability           |
| **Streamlit**        | Interactive dashboard          |
| **Jupyter Notebook** | Data science experimentation   |
| **Power BI**         | Additional reporting           |

---

# ⚙️ Installation

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

The `requirements.txt` file is located in the project root:

```bash
pip install -r requirements.txt
```

If the requirements file is unavailable, install the main dependencies manually:

```bash
pip install pandas numpy scipy matplotlib seaborn plotly scikit-learn statsmodels prophet shap streamlit jupyter
```

---

# ▶️ Run the Dashboard

From the project root:

```bash
streamlit run src/app_dashboard.py
```

The dashboard should normally be available at:

```text
http://localhost:8501
```

Open the address in your browser.

---

# 🔐 Authentication

The current prototype may include demonstration authentication.

> ⚠️ **Do not commit real passwords, API keys, tokens, or production credentials to GitHub.**

For production deployment, authentication should use secure mechanisms such as:

* Password hashing
* Environment variables or a secrets manager
* Role-based access control
* Secure session management
* Credential rotation

If a real credential has previously been committed to the repository, it should be revoked or changed immediately.

---

# 📊 Data Pipeline

The overall data pipeline is:

```text
Climate Data
     │
     ├─────────────┐
     │             │
Agriculture Data  Economic Data
     │             │
     └──────┬──────┘
            ↓
      Data Cleaning
            ↓
      Data Integration
            ↓
     Master Dataset
            ↓
     Risk Assessment
            ↓
   ┌────────┴─────────┐
   ↓                  ↓
Trend Analysis    ML Prediction
   │                  │
   └────────┬─────────┘
            ↓
       Forecasting
            ↓
      Model Validation
            ↓
      Explainable AI
            ↓
   Streamlit Dashboard
```

---

# 📚 Notebook Workflow

The notebooks are designed to follow the analytical pipeline.

### 01 — Data Exploration

```text
notebooks/01_data_exploration.ipynb
```

Explores the raw datasets and their distributions.

### 02 — Data Cleaning

```text
notebooks/02_data_cleaning.ipynb
```

Cleans and prepares the datasets.

### 03 — Risk Assessment

```text
notebooks/03_risk_assessment.ipynb
```

Calculates regional risk scores.

### 04 — Trend Analysis

```text
notebooks/04_trend_analysis.ipynb
```

Analyzes historical risk trends.

### 05 — Simple Forecasting

```text
notebooks/05_simple_forecasting.ipynb
```

Creates baseline forecasts.

### 06 — Prophet Forecasting

```text
notebooks/06_prophet_forecasting_FIXED.ipynb
```

Generates Prophet-based forecasts.

### 07 — Explainability

```text
notebooks/07_feature_importance_explainability.ipynb
```

Performs feature importance and SHAP analysis.

### 08 — Backtesting

```text
notebooks/08_backtesting_validation.ipynb
```

Evaluates forecasting performance.

---

# 📦 Important Output Files

## Risk Analysis

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
```

## Machine Learning

```text
outputs/reports/feature_importance.csv
outputs/reports/prediction_examples.csv
```

## Validation

```text
outputs/reports/forecast_validation_results.csv
outputs/reports/backtesting_results.csv
outputs/reports/week4_backtesting_report.txt
```

---

# ⚠️ Limitations

CrisisLens is currently a **research/prototype system**.

Important limitations include:

* The current data-generation pipeline includes simulated data.
* Forecast accuracy depends on the quality and amount of historical data.
* Models require additional validation using independent real-world datasets.
* Risk-weight methodology should be empirically validated and standardized.
* The current dashboard authentication is designed for demonstration.
* Predictions should be treated as analytical estimates rather than guaranteed outcomes.
* The system is not intended to replace official emergency-management or government early-warning systems.

For real-world deployment, the system should use verified data sources and undergo extensive validation against historical crisis events.

---

# 🔮 Future Improvements

* [ ] Integrate live government/open-data sources
* [ ] Add automated data ingestion
* [ ] Add district-level risk analysis
* [ ] Improve forecast uncertainty estimation
* [ ] Add real-time crisis alerts
* [ ] Add automated model retraining
* [ ] Improve authentication and authorization
* [ ] Add CI/CD
* [ ] Add unit and integration tests
* [ ] Deploy dashboard to the cloud
* [ ] Add additional ML models
* [ ] Add probabilistic forecasting
* [ ] Add real-world historical crisis-event validation

---

# 🤝 Contributing

Contributions are welcome!

## 1. Fork the Repository

Use GitHub's **Fork** button to create your own copy of the repository.

Then clone your fork:

```bash
git clone https://github.com/YOUR_USERNAME/CrisisLens.git
cd CrisisLens
```

## 2. Create a Feature Branch

```bash
git checkout -b feature/your-feature
```

## 3. Make Your Changes

Implement and test your changes.

## 4. Commit

```bash
git add .
git commit -m "Add new feature"
```

## 5. Push

```bash
git push origin feature/your-feature
```

## 6. Open a Pull Request

Open a pull request on GitHub and describe the changes you made.

---

# 📜 License

CrisisLens is licensed under the **MIT License**.

You are free to use, modify, distribute, and sublicense this project, subject to the terms of the MIT License.

See the [LICENSE](LICENSE) file for the complete license text.

---

# 👨‍💻 Project Summary

**CrisisLens** combines:

```text
Data Engineering
       +
Data Science
       +
Machine Learning
       +
Time-Series Forecasting
       +
Explainable AI
       +
Interactive Visualization
       =
Crisis Intelligence Platform
```

The project demonstrates how multi-dimensional environmental, agricultural, and economic data can be combined with machine learning and forecasting techniques to build an analytical crisis-monitoring system.

---

# ⭐ If You Find This Project Useful

If you find CrisisLens useful:

* ⭐ Star the repository
* 🍴 Fork the project
* 🐛 Report issues
* 💡 Suggest improvements
* 🤝 Contribute

---

# 📬 Contact

**Project:** CrisisLens

**Repository:** [Aishhwaryaa-Shrivastava/CrisisLens](https://github.com/Aishhwaryaa-Shrivastava/CrisisLens)

For questions, suggestions, or collaboration, open an issue or submit a pull request.

