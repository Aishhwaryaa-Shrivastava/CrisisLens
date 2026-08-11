import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Create data directories if they don't exist
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

print("=" * 60)
print("CrisisLens Data Collection - Day 1")
print("=" * 60)

# Indian states list
INDIAN_STATES = [
    'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
    'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand',
    'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
    'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab',
    'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura',
    'Uttar Pradesh', 'Uttarakhand', 'West Bengal'
]

print(f"\n✓ Loaded {len(INDIAN_STATES)} Indian states")

# ============================================
# DATASET 1: Climate Data (Simulated based on real patterns)
# ============================================
print("\n[1/3] Generating climate dataset...")

# We'll create realistic sample data based on actual Indian climate patterns
# In Week 2, we'll replace this with real IMD API data

climate_data = []
np.random.seed(42)  # For reproducibility

# Generate 12 months of data for each state (Jan 2024 - Dec 2024)
start_date = datetime(2024, 1, 1)

for state in INDIAN_STATES:
    # Base climate characteristics by region
    if state in ['Rajasthan', 'Gujarat', 'Haryana', 'Punjab']:
        # Hot, dry states
        base_temp = 32
        base_rainfall = 400
    elif state in ['Kerala', 'Goa', 'Karnataka', 'Tamil Nadu']:
        # Coastal states - moderate temp, high rainfall
        base_temp = 28
        base_rainfall = 2500
    elif state in ['Assam', 'Meghalaya', 'Tripura', 'Mizoram']:
        # Northeast - high rainfall
        base_temp = 24
        base_rainfall = 2800
    elif state in ['Himachal Pradesh', 'Uttarakhand', 'Sikkim']:
        # Mountain states - cold, moderate rainfall
        base_temp = 15
        base_rainfall = 1200
    else:
        # Central/Eastern states
        base_temp = 27
        base_rainfall = 1100
    
    for month in range(12):
        date = start_date + timedelta(days=30 * month)
        
        # Add seasonal variation
        temp_seasonal = 5 * np.sin((month - 2) * np.pi / 6)  # Peak in May-Jun
        rainfall_seasonal = base_rainfall / 12 * (1 + 2 * np.sin((month - 5) * np.pi / 3))
        
        # Add random variation
        temp = base_temp + temp_seasonal + np.random.normal(0, 2)
        rainfall = max(0, rainfall_seasonal * (1 + np.random.normal(0, 0.3)))
        
        # Calculate anomalies (deviation from normal)
        temp_anomaly = temp - base_temp
        rainfall_anomaly = (rainfall - base_rainfall/12) / (base_rainfall/12) * 100
        
        climate_data.append({
            'state': state,
            'date': date.strftime('%Y-%m-%d'),
            'month': date.strftime('%Y-%m'),
            'avg_temperature': round(temp, 1),
            'total_rainfall_mm': round(rainfall, 1),
            'temp_anomaly_deg': round(temp_anomaly, 2),
            'rainfall_anomaly_pct': round(rainfall_anomaly, 1),
            'heatwave_days': max(0, int(np.random.normal(2, 3))) if temp > 40 else 0,
            'drought_severity': max(0, min(10, int((100 - rainfall_anomaly) / 10))) if rainfall_anomaly < -20 else 0
        })

df_climate = pd.DataFrame(climate_data)
df_climate.to_csv('data/raw/climate_data_2024.csv', index=False)
print(f"   ✓ Saved climate_data_2024.csv")
print(f"   ✓ {len(df_climate)} records ({len(INDIAN_STATES)} states × 12 months)")

# ============================================
# DATASET 2: Agricultural Data
# ============================================
print("\n[2/3] Generating agricultural dataset...")

agri_data = []

for state in INDIAN_STATES:
    # Base agricultural characteristics
    if state in ['Punjab', 'Haryana', 'Uttar Pradesh']:
        # Major grain producers
        base_production = 25000  # thousand tonnes
        irrigation_pct = 85
    elif state in ['Maharashtra', 'Karnataka', 'Madhya Pradesh']:
        # Mixed agriculture
        base_production = 18000
        irrigation_pct = 45
    elif state in ['Rajasthan', 'Gujarat']:
        # Water-stressed states
        base_production = 12000
        irrigation_pct = 35
    else:
        # Other states
        base_production = 10000
        irrigation_pct = 55
    
    for month in range(12):
        date = start_date + timedelta(days=30 * month)
        
        # Kharif (Jun-Oct) and Rabi (Nov-Apr) seasons
        if month in [5, 6, 7, 8, 9]:  # Kharif
            production_factor = 1.5
        elif month in [10, 11, 0, 1, 2, 3]:  # Rabi
            production_factor = 1.2
        else:
            production_factor = 0.5  # Off-season
        
        production = base_production * production_factor * (1 + np.random.normal(0, 0.2))
        
        # Crop failure rate (higher if drought)
        climate_month = df_climate[(df_climate['state'] == state) & 
                                   (df_climate['month'] == date.strftime('%Y-%m'))]
        if not climate_month.empty:
            drought = climate_month['drought_severity'].values[0]
            crop_failure_rate = min(50, drought * 2 + np.random.normal(5, 3))
        else:
            crop_failure_rate = np.random.normal(5, 3)
        
        agri_data.append({
            'state': state,
            'date': date.strftime('%Y-%m-%d'),
            'month': date.strftime('%Y-%m'),
            'crop_production_1000t': round(production, 0),
            'irrigation_coverage_pct': round(irrigation_pct + np.random.normal(0, 5), 1),
            'crop_failure_rate_pct': max(0, round(crop_failure_rate, 1)),
            'fertilizer_consumption_kg_per_ha': round(100 + np.random.normal(0, 20), 1),
            'land_under_cultivation_1000ha': round(base_production / 2 + np.random.normal(0, 500), 0)
        })

df_agri = pd.DataFrame(agri_data)
df_agri.to_csv('data/raw/agriculture_data_2024.csv', index=False)
print(f"   ✓ Saved agriculture_data_2024.csv")
print(f"   ✓ {len(df_agri)} records ({len(INDIAN_STATES)} states × 12 months)")

# ============================================
# DATASET 3: Economic Indicators
# ============================================
print("\n[3/3] Generating economic indicators dataset...")

econ_data = []

for state in INDIAN_STATES:
    # Base economic characteristics
    if state in ['Maharashtra', 'Tamil Nadu', 'Karnataka', 'Gujarat']:
        # Developed states
        base_gdp = 500000  # Crores
        base_unemployment = 4.5
        base_inflation = 5.2
    elif state in ['Uttar Pradesh', 'Bihar', 'Jharkhand', 'Odisha']:
        # Developing states
        base_gdp = 200000
        base_unemployment = 8.5
        base_inflation = 6.5
    else:
        # Other states
        base_gdp = 300000
        base_unemployment = 6.0
        base_inflation = 5.8
    
    for month in range(12):
        date = start_date + timedelta(days=30 * month)
        
        # Economic trends (gradual changes over year)
        gdp_growth = base_gdp * (1.06 ** (month/12))  # 6% annual growth
        unemployment = base_unemployment + np.random.normal(0, 0.5)
        inflation = base_inflation + np.sin(month * np.pi / 6) + np.random.normal(0, 0.3)
        
        econ_data.append({
            'state': state,
            'date': date.strftime('%Y-%m-%d'),
            'month': date.strftime('%Y-%m'),
            'gdp_estimate_crores': round(gdp_growth, 0),
            'unemployment_rate_pct': max(0, round(unemployment, 2)),
            'inflation_rate_pct': round(inflation, 2),
            'poverty_rate_pct': max(0, round(base_unemployment * 1.5 + np.random.normal(0, 2), 1)),
            'per_capita_income_inr': round(gdp_growth * 10000 / 50, 0)  # Rough estimate
        })

df_econ = pd.DataFrame(econ_data)
df_econ.to_csv('data/raw/economic_data_2024.csv', index=False)
print(f"   ✓ Saved economic_data_2024.csv")
print(f"   ✓ {len(df_econ)} records ({len(INDIAN_STATES)} states × 12 months)")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 60)
print("✓ DATA COLLECTION COMPLETE!")
print("=" * 60)
print("\nFiles created in data/raw/:")
print("  1. climate_data_2024.csv       - Temperature, rainfall, drought severity")
print("  2. agriculture_data_2024.csv   - Crop production, irrigation, failures")
print("  3. economic_data_2024.csv      - GDP, unemployment, inflation")
print("\nDataset Overview:")
print(f"  • Time Period: Jan 2024 - Dec 2024")
print(f"  • States Covered: {len(INDIAN_STATES)}")
print(f"  • Total Records: {len(df_climate) + len(df_agri) + len(df_econ)}")
print("\nNext Steps:")
print("  1. Run: jupyter notebook")
print("  2. Create new notebook: 01_data_exploration.ipynb")
print("  3. Start exploring your data!")
print("\n" + "=" * 60)

# Quick data preview
print("\nQuick Preview - Climate Data (First 5 rows):")
print(df_climate.head())
print("\nQuick Preview - Agriculture Data (First 5 rows):")
print(df_agri.head())
print("\nQuick Preview - Economic Data (First 5 rows):")
print(df_econ.head())