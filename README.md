# 🚨 CrisisLens — Multi-Factor Crisis Risk Forecasting System for India

> An AI-based early warning and risk forecasting system for 36 Indian States and Union Territories.

CrisisLens is a web-based crisis risk forecasting and monitoring system designed to help identify potential crises before they become severe. The system combines climate, agriculture, and economic indicators to calculate a state-level risk score from 0 to 100, then uses forecasting models to estimate future risk for the next 30, 60, and 90 days.

Instead of only showing what is happening now, CrisisLens answers:
> **"What could happen next, and where should we focus attention?"**

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Our Solution](#-our-solution)
- [Key Features](#-key-features)
- [How CrisisLens Works](#-how-crisislens-works)
- [Risk Scoring](#-risk-scoring)
- [Prediction Models](#-prediction-models)
- [Explainable AI](#-explainable-ai)
- [Dashboard](#-dashboard)
- [System Architecture](#-system-architecture)
- [Data Sources](#-data-sources)
- [Technology Stack](#-technology-stack)
- [Hardware & Software Requirements](#-hardware--software-requirements)
- [Installation](#-installation)
- [Running the Application](#-running-the-application)
- [Project Progress & Insights](#-project-progress--insights)
- [Academic Context & Team](#-academic-context--team)
- [Disclaimer](#-disclaimer)

---

## 🔎 Overview

India experiences different types of crises, including floods, droughts, climate stress, agricultural losses, economic stress, and price instability. These problems are often analyzed separately. **CrisisLens** brings these multiple indicators together into a single state-level risk assessment system.

### Core Architecture Flow
```text
Climate Data      ┐
Agriculture Data  ┼─► Data Processing ─► Risk Score (0-100) ─► Forecasting Models ─► 30/60/90 Day Forecast ─► Streamlit Dashboard
Economic Data     ┘

❗ Problem Statement

Current crisis-monitoring approaches are largely reactive—action is taken after significant damage has already occurred.

Major Challenges
Fragmented Data: Data is distributed across multiple separate sources.
Isolated Analysis: Different crisis factors are usually analyzed independently.
Lack of Forecasting: Future risk is difficult to understand from raw, real-time data alone.
Complex Outputs: Decision-makers need simple, interpretable information rather than complex multi-variable metrics.The Key Gap: We know what is happening now, but we lack proactive systems to understand what may happen in the next 30 to 90 days.

💡 Our Solution

CrisisLens follows a multi-factor approach that consolidates:
Climate Data
Agriculture Data
Economic Data
These indicators are processed into a normalized 0–100 risk score. The system then uses forecasting models to estimate future trends and risk levels for 30 days, 60 days, and 90 days in advance.

✨ Key Features
🗺️ State-Level Risk Monitoring: Analyze crisis risk across all 36 Indian States and Union Territories.
📊 Multi-Factor Risk Score: Combines climate, agriculture, and economic stress into a single score.
🔮 Future Risk Forecasting: Projects potential risk for 30, 60, and 90 days.
🤖 Multiple Forecasting Models: Supports ARIMA, Prophet, Linear Regression, and optional LSTM integration.
🧠 Explainable Predictions: Shows the contribution of individual factors behind a region's risk score.
📈 Interactive Dashboard: Built using Streamlit and Plotly for real-time visualization of trends, alerts, and forecasts.

⚙️ How CrisisLens Works

DATA COLLECTION ──► DATA PROCESSING ──► RISK CALCULATION ──► FORECASTING ──► EXPLAINABILITY ──► DASHBOARD

Data Collection: Data is gathered from public sources covering climate, agriculture, and economic conditions.
Data Processing: Datasets are cleaned, transformed, and combined.
Risk Scoring: Indicators are converted into a composite risk score between 0 and 100.
Forecasting: Time-series models analyze historical trends to project future conditions.
Explainability: Contributing factors are extracted so users can understand the drivers behind a score.
Visualization: Output is displayed on the Streamlit dashboard for non-technical decision-makers.

🌡️ Risk Scoring
CrisisLens uses a simple 0–100 risk scale to monitor state health:
Score RangeRisk LevelIndicatorAction / Status0 – 29Low / Normal🟢Everything is fine30 – 49Moderate Stress🟡Watch and monitor50 – 69High Risk🟠Serious problem developing70 – 100Critical / Crisis🔴High threat — immediate intervention required🔮 Prediction ModelsCrisisLens leverages multiple modeling approaches to process time-series trends:ARIMA: Used for time-series forecasting and evaluating baseline trends.Prophet: Captures seasonality patterns and non-linear multi-period trends.Linear Regression: Identifies foundational directional trends across data points.LSTM (Optional): Evaluated for long-term sequence dependency forecasting.The system compares forecasting approaches to evaluate performance across targets.🧠 Explainable AIPrediction alone is not enough; decision-makers must understand why a region is flagged. CrisisLens breaks down contributing factors:PlaintextState Risk Score: 72 (🔴 CRITICAL)

Climate Stress       ████████████████  (High)
Agriculture Stress   █████████████     (Medium-High)
Economic Stress      █████████         (Medium)
📊 DashboardThe CrisisLens dashboard is built using Streamlit and interactive Plotly graphs.Plaintext┌──────────────────────────────────────────────────────────┐
│                      CRISISLENS                          │
│        State-Level Risk Forecasting Dashboard             │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Select State: [ Select Indian State ▼ ]                 │
│                                                          │
│  Current Risk Score: 72        🔴 CRITICAL               │
│                                                          │
│  Factors: Climate (High) | Agri (Med-High) | Econ (Med)  │
│                                                          │
│  ───────────── 30 / 60 / 90 Day Forecast ─────────────   │
│                                                          │
│       Current      30-Day      60-Day      90-Day        │
│         72  ────►    75 ────►    81  ────►   78          │
│                                                          │
│             [ Interactive Plotly Visualizations ]        │
└──────────────────────────────────────────────────────────┘
🏗️ System ArchitecturePlaintext[ Data Sources: Climate, Agri, Econ ] 
                 │
                 ▼
     [ Data Processing Pipeline ]
                 │
                 ▼
    [ Multi-Factor Risk Calculator ]
                 │
                 ▼
      [ Forecasting Engine ]
    (ARIMA / Prophet / Regression)
                 │
                 ▼
    [ Explainability Framework ]
                 │
                 ▼
     [ Streamlit Web Dashboard ]
🗃️ Data SourcesData is integrated across key government and public domains:🌦️ Climate: India Meteorological Department (IMD)🌾 Agriculture: Government agricultural and crop yields datasets💰 Economy: Reserve Bank of India (RBI) and economic indicators🛠️ Technology StackTechnologyPurposePython 3.10+Programming LanguageStreamlitInteractive Dashboard UIPandas & NumPyData Processing & Numerical ComputationPlotlyData VisualizationScikit-LearnMachine Learning ModelingProphetTime-Series Trend ForecastingStatsmodelsARIMA Time-Series ModelingVS CodeDevelopment Environment💻 Hardware & Software RequirementsHardwareProcessor: Intel i3 / i5 or higherRAM: Minimum 4 GBStorage: 20 GB free spaceOS: 64-bit Operating SystemPeripherals: Keyboard, Monitor, and active Internet connectionSoftwarePython 3.10 or higherModern web browser (Chrome, Edge, Firefox)🚀 Installation1. Clone the RepositoryBashgit clone [https://github.com/Aishhwaryaa-Shrivastava/CrisisLens.git](https://github.com/Aishhwaryaa-Shrivastava/CrisisLens.git)
cd CrisisLens
2. Set Up a Virtual EnvironmentBash# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
3. Install DependenciesBashpip install -r requirements.txt
▶️ Running the ApplicationExecute the Streamlit application using:Bashstreamlit run app_dashboard.py
After running, open the browser URL provided in the terminal (typically http://localhost:8501).📈 Project Progress & InsightsProgress[x] Data collection pipeline setup.[x] Integration of climate, agricultural, and economic data sources.[x] Initial multi-factor risk-scoring structure.[ ] Model refinement and automated threshold alert deployment.Key InsightsMulti-Factor Synergy: Crises are rarely isolated; climate stress often directly drives agricultural and economic risk.Seasonal Dependencies: Models must heavily account for seasonal weather variations in India.Horizon Uncertainty: Forecast accuracy naturally decreases as projection windows lengthen from 30 to 90 days.🎓 Academic Context & TeamProject Title: CrisisLens: Multi-Factor Crisis Risk Forecasting System for IndiaProject Type: Capstone ProjectInstitution: Bhilai Institute of TechnologyLead Developer: Aishwarya ShrivastavaMentorship: Faculty Members of Bhilai Institute of Technology⚠️ DisclaimerCrisisLens is an academic capstone research project and decision-support prototype. Predictions produced by this system should be evaluated alongside official government emergency management systems and real-time warnings.
