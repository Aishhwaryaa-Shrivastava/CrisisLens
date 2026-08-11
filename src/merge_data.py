"""
CrisisLens: Data Merger - Combine 2024 + 2025 Data
Prepares complete dataset for forecasting model training
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("CrisisLens Data Merger")
print("=" * 60)

# ============================================================================
# STEP 1: Load Both Datasets
# ============================================================================
print("\nSTEP 1: Loading datasets...")

try:
    data_2024 = pd.read_csv('data/processed/master_dataset_with_risks.csv')
    print(f"Loaded 2024 data: {len(data_2024):,} records, {data_2024['state'].nunique()} states")
except FileNotFoundError:
    print("ERROR: 2024 data file not found!")
    exit(1)

try:
    data_2025 = pd.read_csv('india_crisis_data_2025.csv')
    print(f"Loaded 2025 data: {len(data_2025):,} records, {data_2025['state'].nunique()} regions")
except FileNotFoundError:
    print("ERROR: 2025 data not found! Run download_2025_data.py first.")
    exit(1)

# ============================================================================
# STEP 2: Add Missing Union Territories to 2024 Data
# ============================================================================
print("\nSTEP 2: Adding Union Territories to 2024 data...")

ALL_UTS = [
    'Andaman and Nicobar Islands', 'Chandigarh',
    'Dadra and Nagar Haveli and Daman and Diu', 'Delhi',
    'Jammu and Kashmir', 'Ladakh', 'Lakshadweep', 'Puducherry'
]

existing_states_2024 = data_2024['state'].unique()
missing_uts = [ut for ut in ALL_UTS if ut not in existing_states_2024]

if missing_uts:
    print(f"   Adding {len(missing_uts)} missing UTs...")
    ut_data_2024 = []
    for ut in missing_uts:
        if ut == 'Delhi':
            template_state = 'Haryana'
        elif ut in ['Jammu and Kashmir', 'Ladakh']:
            template_state = 'Himachal Pradesh'
        elif ut == 'Puducherry':
            template_state = 'Tamil Nadu'
        elif ut == 'Chandigarh':
            template_state = 'Punjab'
        else:
            template_state = data_2024['state'].iloc[0]

        template_data = data_2024[data_2024['state'] == template_state].copy()
        for _, row in template_data.iterrows():
            new_row = row.copy()
            new_row['state'] = ut
            if 'avg_temperature' in new_row:
                new_row['avg_temperature'] *= np.random.uniform(0.95, 1.05)
            if 'total_rainfall_mm' in new_row:
                new_row['total_rainfall_mm'] *= np.random.uniform(0.9, 1.1)
            if 'gdp_estimate_crores' in new_row:
                new_row['gdp_estimate_crores'] *= np.random.uniform(0.3, 0.7)
            ut_data_2024.append(new_row)

    data_2024 = pd.concat([data_2024, pd.DataFrame(ut_data_2024)], ignore_index=True)
    print(f"   Added {len(ut_data_2024)} records for UTs")

print(f"   2024 now has {data_2024['state'].nunique()} regions")

# ============================================================================
# STEP 3: Drop all pre-computed stress/risk scores — recompute from scratch
# The 2024 notebook scores (0-10 scale) and 2025 generator scores (0-20 scale)
# are incompatible. Raw indicators are compatible, so we recompute after merge.
# ============================================================================
print("\nSTEP 3: Dropping pre-computed stress scores (will recompute from raw indicators)...")

drop_cols = ['climate_stress', 'agri_stress', 'econ_stress',
             'climate_risk', 'agriculture_risk', 'economic_risk',
             'composite_risk_score', 'total_risk', 'risk_category']

for label, df in [('2024', data_2024), ('2025', data_2025)]:
    dropped = [c for c in drop_cols if c in df.columns]
    df.drop(columns=dropped, inplace=True)
    if dropped:
        print(f"   Dropped from {label}: {dropped}")

# ============================================================================
# STEP 4: Standardize dates and columns, then merge
# ============================================================================
print("\nSTEP 4: Standardizing and merging...")

data_2024['date'] = pd.to_datetime(data_2024['date'], format='%Y-%m-%d', errors='coerce')
data_2025['date'] = pd.to_datetime(data_2025['date'], format='%d-%m-%Y', errors='coerce')

print(f"   2024 dates: {data_2024['date'].min().date()} to {data_2024['date'].max().date()}  ({data_2024['date'].isna().sum()} nulls)")
print(f"   2025 dates: {data_2025['date'].min().date()} to {data_2025['date'].max().date()}  ({data_2025['date'].isna().sum()} nulls)")

# Keep only the raw indicator columns that exist in both datasets
RAW_COLS = [
    'state', 'date',
    'avg_temperature', 'total_rainfall_mm', 'temp_anomaly_deg',
    'rainfall_anomaly_pct', 'heatwave_days', 'drought_severity',
    'crop_production_1000t', 'irrigation_coverage_pct', 'crop_failure_rate_pct',
    'fertilizer_consumption_kg_per_ha', 'land_under_cultivation_1000ha',
    'gdp_estimate_crores', 'unemployment_rate_pct', 'inflation_rate_pct',
    'poverty_rate_pct', 'per_capita_income_inr'
]

cols_2024 = [c for c in RAW_COLS if c in data_2024.columns]
cols_2025 = [c for c in RAW_COLS if c in data_2025.columns]

combined = pd.concat([data_2024[cols_2024], data_2025[cols_2025]], ignore_index=True)
combined = combined.sort_values(['state', 'date']).reset_index(drop=True)
# Drop rows where date failed to parse
combined = combined[combined['date'].notna()].copy()

print(f"   Combined: {len(combined):,} records, {combined['state'].nunique()} regions")
print(f"   Date range: {combined['date'].min().date()} to {combined['date'].max().date()}")

# ============================================================================
# STEP 5: Monthly aggregation
# ============================================================================
print("\nSTEP 5: Monthly aggregation...")

combined['year_month'] = combined['date'].dt.to_period('M')

agg_cols = {c: 'mean' for c in RAW_COLS if c not in ['state', 'date']}
agg_cols['total_rainfall_mm'] = 'sum'
agg_cols['heatwave_days'] = 'sum'
agg_cols['crop_production_1000t'] = 'sum'
agg_cols = {k: v for k, v in agg_cols.items() if k in combined.columns}

monthly = combined.groupby(['state', 'year_month']).agg(agg_cols).reset_index()
monthly['month'] = monthly['year_month'].astype(str)
monthly = monthly.drop('year_month', axis=1)

print(f"   Monthly rows: {len(monthly):,}  ({len(monthly)//monthly['state'].nunique()} months/state)")

# ============================================================================
# STEP 6: Recompute stress scores from raw indicators (consistent formula)
# All indicators normalised across the FULL combined dataset so 2024 and
# 2025 are on the same scale. Weights match the original notebook design.
# ============================================================================
print("\nSTEP 6: Recomputing stress scores from raw indicators...")

def minmax(series):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(50.0, index=series.index)
    return (series - mn) / (mx - mn) * 100

def minmax_col(df, col):
    """Normalise a column globally across the full dataset."""
    return minmax(df[col])

# Normalise each raw indicator globally (0-100 across all states & months)
monthly['t_norm']   = minmax_col(monthly, 'temp_anomaly_deg')
monthly['r_norm']   = minmax(monthly['rainfall_anomaly_pct'].abs())
monthly['hw_norm']  = minmax_col(monthly, 'heatwave_days')
monthly['dr_norm']  = minmax_col(monthly, 'drought_severity')
monthly['cf_norm']  = minmax_col(monthly, 'crop_failure_rate_pct')
monthly['irr_norm'] = 100 - minmax_col(monthly, 'irrigation_coverage_pct')
monthly['ue_norm']  = minmax_col(monthly, 'unemployment_rate_pct')
monthly['inf_norm'] = minmax_col(monthly, 'inflation_rate_pct')
monthly['pov_norm'] = minmax_col(monthly, 'poverty_rate_pct')

# Compute raw composites
monthly['climate_raw'] = (
    monthly['t_norm']  * 0.25 +
    monthly['r_norm']  * 0.30 +
    monthly['hw_norm'] * 0.15 +
    monthly['dr_norm'] * 0.30
)
monthly['agri_raw'] = (
    monthly['cf_norm']  * 0.55 +
    monthly['irr_norm'] * 0.25 +
    monthly['dr_norm']  * 0.20
)
monthly['econ_raw'] = (
    monthly['ue_norm']  * 0.35 +
    monthly['inf_norm'] * 0.30 +
    monthly['pov_norm'] * 0.35
)

# Rank-based normalisation: convert each category to percentile ranks (0-100)
# This guarantees all three have identical distributions regardless of
# how much their raw indicators vary — so no single category can dominate
# just because it has more variance.
monthly['climate_stress'] = monthly['climate_raw'].rank(pct=True) * 100
monthly['agri_stress']    = monthly['agri_raw'].rank(pct=True) * 100
monthly['econ_stress']    = monthly['econ_raw'].rank(pct=True) * 100

# Drop temp columns
monthly.drop(columns=['t_norm','r_norm','hw_norm','dr_norm','cf_norm',
                       'irr_norm','ue_norm','inf_norm','pov_norm',
                       'climate_raw','agri_raw','econ_raw'], inplace=True)

# Total risk: weighted average of the three categories (sums to 1.0)
monthly['total_risk'] = (
    monthly['climate_stress'] * 0.40 +
    monthly['agri_stress']    * 0.35 +
    monthly['econ_stress']    * 0.25
)

print("   Stress score ranges:")
for col in ['climate_stress', 'agri_stress', 'econ_stress', 'total_risk']:
    print(f"   {col}: min={monthly[col].min():.1f}  max={monthly[col].max():.1f}  avg={monthly[col].mean():.1f}")

# Dominant driver check
def dominant_driver(row):
    return max(
        [('Climate', row['climate_stress']),
         ('Agri',    row['agri_stress']),
         ('Econ',    row['econ_stress'])],
        key=lambda x: x[1]
    )[0]

latest = monthly[monthly['month'] == monthly['month'].max()]
driver_counts = latest.apply(dominant_driver, axis=1).value_counts()
print(f"\n   Dominant driver (latest month): {driver_counts.to_dict()}")

# ============================================================================
# STEP 7: Save
# ============================================================================
print("\nSTEP 7: Saving...")

combined.to_csv('data/processed/combined_daily_2024_2025.csv', index=False)
print("   Saved: data/processed/combined_daily_2024_2025.csv")

monthly.to_csv('data/processed/combined_monthly_2024_2025.csv', index=False)
print("   Saved: data/processed/combined_monthly_2024_2025.csv")

# ============================================================================
# STEP 8: Quality report
# ============================================================================
print("\nSTEP 8: Quality Report")
print("=" * 60)

monthly['year'] = monthly['month'].str[:4]
print("\nCoverage by year:")
print(monthly.groupby('year')['state'].count().to_string())

print("\nTop 10 high-risk states (avg total_risk):")
top = monthly.groupby('state')['total_risk'].mean().nlargest(10)
for state, risk in top.items():
    print(f"   {state}: {risk:.1f}")

print("\nDone! combined_monthly_2024_2025.csv is ready.")
