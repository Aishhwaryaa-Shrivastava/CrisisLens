"""
CrisisLens: 2025 Data Generator for Indian States & Union Territories
Generates realistic 2025 data based on 2024 patterns with realistic variations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("🚀 CrisisLens 2025 Data Generator")
print("=" * 60)

# ============================================================================
# STEP 1: Load Your 2024 Data
# ============================================================================
print("\n📂 STEP 1: Loading 2024 baseline data...")

# You'll replace this with your actual file
# For now, I'll create structure based on your format
# REPLACE 'your_2024_data.csv' with your actual filename

try:
    data_2024 = pd.read_csv('data/processed/master_dataset_with_risks.csv')
    print(f"✅ Loaded 2024 data: {len(data_2024)} records")
except FileNotFoundError:
    print("⚠️  2024 data file not found. Creating sample structure...")
    # If file not found, we'll generate from scratch
    data_2024 = None

# ============================================================================
# STEP 2: Define All States + Union Territories (36 total)
# ============================================================================
print("\n📍 STEP 2: Setting up all 36 regions...")

ALL_REGIONS = [
    # 28 States
    'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
    'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand',
    'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
    'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab',
    'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura',
    'Uttar Pradesh', 'Uttarakhand', 'West Bengal',
    # 8 Union Territories
    'Andaman and Nicobar Islands', 'Chandigarh', 
    'Dadra and Nagar Haveli and Daman and Diu', 'Delhi',
    'Jammu and Kashmir', 'Ladakh', 'Lakshadweep', 'Puducherry'
]

print(f"✅ Configured {len(ALL_REGIONS)} regions (28 states + 8 UTs)")

# ============================================================================
# STEP 3: Generate 2025 Data with Realistic Patterns
# ============================================================================
print("\n🔮 STEP 3: Generating 2025 data...")

def generate_2025_data(baseline_2024=None):
    """
    Generate realistic 2025 data based on 2024 patterns
    If no baseline provided, creates data from scratch
    """
    
    # Date range: Jan 1, 2025 to Dec 31, 2025
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    all_data = []
    
    for region in ALL_REGIONS:
        print(f"  Processing: {region}...")
        
        # Get baseline stats (from 2024 if available)
        if baseline_2024 is not None and region in baseline_2024['state'].values:
            region_2024 = baseline_2024[baseline_2024['state'] == region]
            
            # Calculate 2024 averages
            avg_temp_base = region_2024['avg_temperature'].mean()
            avg_rain_base = region_2024['total_rainfall_mm'].mean()
            avg_crop_base = region_2024['crop_production_1000t'].mean()
            avg_gdp_base = region_2024['gdp_estimate_crores'].mean()
            avg_unemployment_base = region_2024['unemployment_rate_pct'].mean()
        else:
            # Default values for new regions (UTs not in 2024 data)
            avg_temp_base = 25.0 + np.random.uniform(-5, 5)
            avg_rain_base = 150.0 + np.random.uniform(-50, 100)
            avg_crop_base = 5000 + np.random.uniform(-2000, 3000)
            avg_gdp_base = 150000 + np.random.uniform(-50000, 100000)
            avg_unemployment_base = 6.5 + np.random.uniform(-2, 2)
        
        for date in dates:
            month = date.month
            day_of_year = date.timetuple().tm_yday
            
            # =============================================
            # CLIMATE INDICATORS (with realistic trends)
            # =============================================
            
            # Temperature: Seasonal variation + climate change trend
            seasonal_temp = 5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            temp_trend = 0.15  # Slight warming trend 2024→2025
            temp_noise = np.random.normal(0, 1.5)
            avg_temperature = avg_temp_base + seasonal_temp + temp_trend + temp_noise
            
            # Rainfall: Monsoon patterns
            if month in [6, 7, 8, 9]:  # Monsoon months
                rainfall_multiplier = 3.5
            elif month in [10, 11]:  # Post-monsoon
                rainfall_multiplier = 1.2
            else:  # Dry season
                rainfall_multiplier = 0.3
            
            total_rainfall_mm = max(0, avg_rain_base * rainfall_multiplier * 
                                    np.random.uniform(0.7, 1.3))
            
            # Temperature anomaly (vs long-term average)
            temp_anomaly_deg = np.random.uniform(-2, 4)
            
            # Rainfall anomaly
            rainfall_anomaly_pct = (total_rainfall_mm / avg_rain_base - 1) * 100 if avg_rain_base > 0 else 0
            
            # Heatwave days (more common Apr-Jun)
            if month in [4, 5, 6] and avg_temperature > 38:
                heatwave_days = np.random.randint(0, 8)
            else:
                heatwave_days = 0
            
            # Drought severity (0-10 scale)
            if total_rainfall_mm < avg_rain_base * 0.5:
                drought_severity = np.random.randint(5, 11)
            elif total_rainfall_mm < avg_rain_base * 0.75:
                drought_severity = np.random.randint(2, 6)
            else:
                drought_severity = 0
            
            # =============================================
            # AGRICULTURE INDICATORS
            # =============================================
            
            # Crop production (seasonal)
            if month in [10, 11, 12]:  # Kharif harvest
                crop_production_1000t = avg_crop_base * np.random.uniform(0.9, 1.2)
            elif month in [3, 4, 5]:  # Rabi harvest
                crop_production_1000t = avg_crop_base * np.random.uniform(0.7, 1.0)
            else:
                crop_production_1000t = avg_crop_base * 0.3 * np.random.uniform(0.8, 1.2)
            
            irrigation_coverage_pct = 45 + np.random.uniform(-5, 15)
            
            # Crop failure (increases with drought/heatwave)
            base_failure_rate = 3.5
            climate_impact = (drought_severity / 10) * 5 + (heatwave_days / 7) * 3
            crop_failure_rate_pct = min(25, base_failure_rate + climate_impact + 
                                       np.random.uniform(-1, 2))
            
            fertilizer_consumption_kg_per_ha = 100 + np.random.uniform(-20, 40)
            land_under_cultivation_1000ha = 5000 + np.random.uniform(-1000, 1000)
            
            # =============================================
            # ECONOMIC INDICATORS
            # =============================================
            
            # GDP (slight growth trend)
            gdp_estimate_crores = avg_gdp_base * (1 + 0.06/12) * (1 + np.random.uniform(-0.02, 0.02))
            
            # Unemployment (inverse correlation with GDP growth)
            unemployment_rate_pct = max(2, avg_unemployment_base + np.random.uniform(-0.5, 0.5))
            
            # Inflation (realistic 2025 range)
            inflation_rate_pct = 5.5 + np.random.uniform(-1, 1.5)
            
            # Poverty rate (slow-changing)
            poverty_rate_pct = 10 + np.random.uniform(-2, 3)
            
            # Per capita income
            per_capita_income_inr = 55000 + np.random.uniform(-10000, 20000)
            
            # =============================================
            # COMPOSITE STRESS SCORES
            # All three categories balanced on 0-20 scale.
            # Each term is capped at its stated max so no single
            # indicator can dominate the composite score.
            # =============================================

            # Climate stress (0-20 scale)
            # 4 indicators × max 5pts each = 20
            climate_stress = (
                min((temp_anomaly_deg + 2) / 6 * 5, 5) +       # Temperature   (0-5)
                min((abs(rainfall_anomaly_pct) / 100) * 5, 5) + # Rainfall      (0-5)
                min((heatwave_days / 7) * 5, 5) +               # Heatwave      (0-5)
                min((drought_severity / 10) * 5, 5)             # Drought       (0-5)
            )

            # Agriculture stress (0-20 scale)
            # FIX: crop_failure was worth 10pts (double others) — now 7pts
            # 3 indicators, balanced weights totalling 20
            agri_stress = (
                min((crop_failure_rate_pct / 25) * 7, 7) +          # Crop failure  (0-7)
                min(((100 - irrigation_coverage_pct) / 100) * 6, 6) + # Irrigation gap (0-6)
                min((drought_severity / 10) * 7, 7)                 # Drought       (0-7)
            )

            # Economic stress (0-20 scale)
            # 3 indicators × max ~6-7pts each = 20
            econ_stress = (
                min((unemployment_rate_pct / 10) * 7, 7) +  # Unemployment  (0-7)
                min((inflation_rate_pct / 10) * 7, 7) +     # Inflation     (0-7)
                min((poverty_rate_pct / 20) * 6, 6)         # Poverty       (0-6)
            )
            
            # =============================================
            # META FIELDS
            # =============================================
            
            year = 2025
            month_num = month
            quarter = (month - 1) // 3 + 1
            
            # Season classification
            if month in [12, 1, 2]:
                season = 'Winter'
            elif month in [3, 4, 5]:
                season = 'Summer'
            elif month in [6, 7, 8, 9]:
                season = 'Monsoon'
            else:
                season = 'Post-Monsoon'
            
            # Compile record
            record = {
                'state': region,
                'date': date.strftime('%d-%m-%Y'),
                'month': f"{year}-{month:02d}",
                'avg_temperature': round(avg_temperature, 1),
                'total_rainfall_mm': round(total_rainfall_mm, 1),
                'temp_anomaly_deg': round(temp_anomaly_deg, 2),
                'rainfall_anomaly_pct': round(rainfall_anomaly_pct, 1),
                'heatwave_days': heatwave_days,
                'drought_severity': drought_severity,
                'crop_production_1000t': round(crop_production_1000t, 0),
                'irrigation_coverage_pct': round(irrigation_coverage_pct, 1),
                'crop_failure_rate_pct': round(crop_failure_rate_pct, 1),
                'fertilizer_consumption_kg_per_ha': round(fertilizer_consumption_kg_per_ha, 1),
                'land_under_cultivation_1000ha': round(land_under_cultivation_1000ha, 0),
                'gdp_estimate_crores': round(gdp_estimate_crores, 0),
                'unemployment_rate_pct': round(unemployment_rate_pct, 2),
                'inflation_rate_pct': round(inflation_rate_pct, 2),
                'poverty_rate_pct': round(poverty_rate_pct, 1),
                'per_capita_income_inr': round(per_capita_income_inr, 0),
                'year': year,
                'month_num': month_num,
                'quarter': quarter,
                'season': season,
                'climate_stress': round(climate_stress, 2),
                'agri_stress': round(agri_stress, 2),
                'econ_stress': round(econ_stress, 2)
            }
            
            all_data.append(record)
    
    return pd.DataFrame(all_data)

# Generate the data
df_2025 = generate_2025_data(data_2024)

print(f"\n✅ Generated 2025 data:")
print(f"   - Total records: {len(df_2025):,}")
print(f"   - Date range: {df_2025['date'].min()} to {df_2025['date'].max()}")
print(f"   - Regions: {df_2025['state'].nunique()}")

# ============================================================================
# STEP 4: Save the Data
# ============================================================================
print("\n💾 STEP 4: Saving generated data...")

output_file = 'india_crisis_data_2025.csv'
df_2025.to_csv(output_file, index=False)

print(f"✅ Saved to: {output_file}")
print(f"   File size: {len(df_2025) * len(df_2025.columns) / 1024:.1f} KB")

# ============================================================================
# STEP 5: Generate Summary Statistics
# ============================================================================
print("\n📊 STEP 5: Data Summary")
print("=" * 60)

# Sample records
print("\n📋 Sample Records (First 3):")
print(df_2025.head(3).to_string(index=False))

print("\n\n📈 Summary Statistics by Region (Top 10 High-Risk States):")
monthly_avg = df_2025.groupby('state').agg({
    'climate_stress': 'mean',
    'agri_stress': 'mean',
    'econ_stress': 'mean'
}).reset_index()

monthly_avg['total_risk'] = (
    monthly_avg['climate_stress'] + 
    monthly_avg['agri_stress'] + 
    monthly_avg['econ_stress']
)

top_risk = monthly_avg.nlargest(10, 'total_risk')
print(top_risk.to_string(index=False))

print("\n\n✅ DATA GENERATION COMPLETE!")
print("=" * 60)
print("\n📂 Next Steps:")
print("1. Review the generated file: india_crisis_data_2025.csv")
print("2. Combine with your 2024 data for training")
print("3. Ready to build forecasting models!")
print("\n🚀 Proceed to Week 3 Day 13 implementation!")
