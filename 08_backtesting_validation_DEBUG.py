"""
================================================================================
WEEK 4 - DAY 22-23: BACKTESTING & VALIDATION (DEBUGGED VERSION)
================================================================================
This version includes debugging to help identify and fix data issues
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("BACKTESTING: VALIDATING MODEL ON REAL HISTORICAL EVENTS (DEBUG MODE)")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD YOUR DATA
# ============================================================================
print("\n[1/6] Loading historical data...")

# First, let's try to load your actual data
try:
    # Try common file paths - UPDATE THIS to match your actual file location
    possible_paths = [
        '../data/processed/state_risk_data.csv',
        '../data/state_risk_data.csv',
        'data/processed/state_risk_data.csv',
        'state_risk_data.csv',
        '../outputs/state_risk_data.csv'
    ]
    
    data = None
    for path in possible_paths:
        try:
            data = pd.read_csv(path)
            print(f"✓ Found data at: {path}")
            break
        except FileNotFoundError:
            continue
    
    if data is None:
        raise FileNotFoundError("Could not find data file in common locations")
    
    # Check if date column exists and convert
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
    elif 'Date' in data.columns:
        data['date'] = pd.to_datetime(data['Date'])
    elif 'month' in data.columns and 'year' in data.columns:
        # Create date from year and month
        data['date'] = pd.to_datetime(data[['year', 'month']].assign(day=1))
    else:
        print("⚠ Warning: No date column found. Creating synthetic dates...")
        data['date'] = pd.date_range('2023-01-01', periods=len(data), freq='MS')
    
    print(f"✓ Loaded {len(data)} records")
    print(f"  Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"  Columns: {list(data.columns)}")
    
    # Check for required columns
    if 'state' not in data.columns and 'State' not in data.columns:
        print("⚠ Warning: No 'state' column found. Adding default states...")
        data['state'] = 'Default_State'
    elif 'State' in data.columns:
        data['state'] = data['State']
    
    # Check for total_risk column
    if 'total_risk' not in data.columns:
        print("⚠ Warning: No 'total_risk' column found.")
        # Try to find alternative risk column
        risk_cols = [col for col in data.columns if 'risk' in col.lower()]
        if risk_cols:
            data['total_risk'] = data[risk_cols[0]]
            print(f"  Using '{risk_cols[0]}' as total_risk")
        else:
            print("  Creating synthetic risk scores...")
            np.random.seed(42)
            data['total_risk'] = np.random.uniform(20, 80, len(data))
    
    print(f"\n✓ Data loaded successfully!")
    print(f"  States available: {data['state'].unique()[:10]}")  # Show first 10 states
    print(f"  Total risk range: {data['total_risk'].min():.2f} - {data['total_risk'].max():.2f}")

except Exception as e:
    print(f"⚠ Could not load actual data: {e}")
    print("Creating demonstration dataset...")
    
    # Create synthetic data for demonstration
    states = ['Punjab', 'Karnataka', 'Kerala', 'Maharashtra', 'Uttar Pradesh', 
              'Tamil Nadu', 'Rajasthan', 'Gujarat', 'West Bengal', 'Madhya Pradesh']
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='MS')
    
    np.random.seed(42)
    data_list = []
    for state in states:
        for date in dates:
            # Create somewhat realistic synthetic data
            month = date.month
            base_rainfall = -20 + 40 * np.sin(2 * np.pi * month / 12)  # Seasonal pattern
            
            data_list.append({
                'state': state,
                'date': date,
                'rainfall_anomaly_pct': base_rainfall + np.random.uniform(-15, 15),
                'temp_anomaly_deg': 1 + np.random.uniform(-1, 2),
                'crop_failure_rate_pct': max(0, 10 + np.random.uniform(-5, 15)),
                'drought_severity': max(0, 3 + np.random.uniform(-2, 5)),
                'unemployment_rate_pct': 6 + np.random.uniform(-2, 4),
                'heatwave_days': max(0, np.random.poisson(3)),
                'total_rainfall_mm': max(0, 100 + np.random.uniform(-50, 150)),
                'total_risk': 0  # Will calculate
            })
    
    data = pd.DataFrame(data_list)
    
    # Calculate total_risk as weighted sum
    data['total_risk'] = (
        abs(data['rainfall_anomaly_pct']) * 0.4 +
        data['temp_anomaly_deg'] * 5 +
        data['crop_failure_rate_pct'] * 0.8 +
        data['drought_severity'] * 3
    ) + np.random.uniform(-5, 5, len(data))
    
    # Clip to reasonable range
    data['total_risk'] = data['total_risk'].clip(15, 85)
    
    print(f"✓ Created synthetic dataset with {len(data)} records")
    print(f"  States: {', '.join(states)}")
    print(f"  Date range: {data['date'].min()} to {data['date'].max()}")

# ============================================================================
# STEP 2: DEFINE REAL CRISIS EVENTS TO TEST
# ============================================================================
print("\n[2/6] Defining crisis events for validation...")

# Get available states and dates from your data
available_states = data['state'].unique()
min_date = data['date'].min()
max_date = data['date'].max()

print(f"\nAvailable data:")
print(f"  States: {list(available_states)[:5]}... ({len(available_states)} total)")
print(f"  Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")

# Define events that match your data
# TODO: CUSTOMIZE THESE to match real events in your study period
crisis_events = [
    {
        'name': 'Karnataka Drought',
        'state': 'Karnataka' if 'Karnataka' in available_states else available_states[0],
        'event_date': '2023-06-15',
        'description': 'Severe drought affecting agricultural regions',
        'actual_severity': 'HIGH'
    },
    {
        'name': 'Kerala Monsoon Impact',
        'state': 'Kerala' if 'Kerala' in available_states else available_states[1] if len(available_states) > 1 else available_states[0],
        'event_date': '2023-08-20',
        'description': 'Heavy monsoon rainfall causing flood risk',
        'actual_severity': 'HIGH'
    },
    {
        'name': 'Punjab Summer Stress',
        'state': 'Punjab' if 'Punjab' in available_states else available_states[2] if len(available_states) > 2 else available_states[0],
        'event_date': '2024-05-10',
        'description': 'Extreme heat affecting wheat harvest',
        'actual_severity': 'MEDIUM'
    },
    {
        'name': 'Maharashtra Crop Stress',
        'state': 'Maharashtra' if 'Maharashtra' in available_states else available_states[3] if len(available_states) > 3 else available_states[0],
        'event_date': '2024-02-28',
        'description': 'Agricultural stress from weather anomalies',
        'actual_severity': 'MEDIUM'
    },
    {
        'name': 'Uttar Pradesh Crisis',
        'state': 'Uttar Pradesh' if 'Uttar Pradesh' in available_states else available_states[4] if len(available_states) > 4 else available_states[0],
        'event_date': '2023-11-05',
        'description': 'Multi-factor crisis affecting rural areas',
        'actual_severity': 'HIGH'
    }
]

# Filter events that are within our data range
valid_events = []
for event in crisis_events:
    event_date = pd.to_datetime(event['event_date'])
    if min_date <= event_date <= max_date:
        valid_events.append(event)
    else:
        print(f"⚠ Skipping {event['name']} - outside data range")

if len(valid_events) == 0:
    print("\n⚠ No events match your data range. Creating synthetic events...")
    # Create events within the data range
    mid_date = min_date + (max_date - min_date) / 2
    for i, state in enumerate(available_states[:5]):
        event_date = mid_date + timedelta(days=30*i)
        valid_events.append({
            'name': f'{state} Event {i+1}',
            'state': state,
            'event_date': event_date.strftime('%Y-%m-%d'),
            'description': f'Synthetic crisis event for {state}',
            'actual_severity': ['HIGH', 'MEDIUM', 'LOW'][i % 3]
        })

crisis_events = valid_events
print(f"\n✓ Using {len(crisis_events)} crisis events:")
for event in crisis_events:
    print(f"  • {event['name']} ({event['state']}, {event['event_date']})")

# ============================================================================
# STEP 3: BACKTESTING FUNCTION (WITH DEBUGGING)
# ============================================================================
print("\n[3/6] Setting up backtesting framework...")

def backtest_event(event, data, prediction_days_before=30):
    """
    Test if model would have predicted a crisis event
    """
    try:
        event_date = pd.to_datetime(event['event_date'])
        prediction_date = event_date - timedelta(days=prediction_days_before)
        
        # Debug: Check data availability
        train_data = data[data['date'] < prediction_date].copy()
        test_data = data[
            (data['date'] == event_date) & 
            (data['state'] == event['state'])
        ].copy()
        
        if len(test_data) == 0:
            # Try to find the closest date
            state_data = data[data['state'] == event['state']].copy()
            state_data['date_diff'] = abs((state_data['date'] - event_date).dt.days)
            closest = state_data.nsmallest(1, 'date_diff')
            
            if len(closest) > 0 and closest['date_diff'].iloc[0] < 15:  # Within 15 days
                test_data = closest
                actual_date = test_data['date'].iloc[0]
                print(f"    Using closest date {actual_date.strftime('%Y-%m-%d')} (±{closest['date_diff'].iloc[0]} days)")
            else:
                return {
                    'success': False,
                    'error': f'No data for {event["state"]} near {event_date.strftime("%Y-%m-%d")}'
                }
        
        if len(train_data) < 10:  # Need at least 10 samples to train
            return {
                'success': False,
                'error': f'Insufficient training data ({len(train_data)} samples)'
            }
        
        # Define features (exclude non-feature columns)
        exclude_cols = ['date', 'state', 'total_risk', 'month', 'year', 'State', 'Date']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        # Make sure we have some features
        if len(feature_cols) == 0:
            return {
                'success': False,
                'error': 'No feature columns found in data'
            }
        
        X_train = train_data[feature_cols].fillna(0)  # Fill NaN with 0
        y_train = train_data['total_risk']
        X_test = test_data[feature_cols].fillna(0)
        y_test = test_data['total_risk']
        
        # Train model
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=10)
        model.fit(X_train, y_train)
        
        # Make prediction
        predicted_risk = model.predict(X_test)[0]
        actual_risk = y_test.iloc[0]
        
        # Classify severity
        def classify_risk(score):
            if score >= 60:
                return 'HIGH'
            elif score >= 40:
                return 'MEDIUM'
            else:
                return 'LOW'
        
        predicted_severity = classify_risk(predicted_risk)
        actual_severity = classify_risk(actual_risk)
        
        # Calculate metrics
        error = abs(predicted_risk - actual_risk)
        error_pct = (error / actual_risk) * 100 if actual_risk > 0 else 0
        
        # Check if we got the severity right
        correct_severity = (predicted_severity == actual_severity)
        
        # Would we have issued an alert?
        alert_issued = predicted_risk >= 50
        should_have_alerted = actual_risk >= 50
        
        return {
            'success': True,
            'event_name': event['name'],
            'state': event['state'],
            'event_date': event_date.strftime('%Y-%m-%d'),
            'prediction_date': prediction_date.strftime('%Y-%m-%d'),
            'days_before': prediction_days_before,
            'predicted_risk': round(predicted_risk, 2),
            'actual_risk': round(actual_risk, 2),
            'absolute_error': round(error, 2),
            'percentage_error': round(error_pct, 2),
            'predicted_severity': predicted_severity,
            'actual_severity': actual_severity,
            'correct_severity': correct_severity,
            'alert_issued': alert_issued,
            'should_have_alerted': should_have_alerted,
            'alert_correct': alert_issued == should_have_alerted,
            'training_samples': len(train_data),
            'features_used': len(feature_cols)
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': f'Error: {str(e)}'
        }

print("✓ Backtesting framework ready")

# ============================================================================
# STEP 4: RUN BACKTESTS ON ALL EVENTS
# ============================================================================
print("\n[4/6] Running backtests on all crisis events...")

results = []
failed_tests = []

for i, event in enumerate(crisis_events, 1):
    print(f"\n  [{i}/{len(crisis_events)}] Testing: {event['name']}...")
    
    # Test at different lead times
    for days_before in [7, 14, 30]:
        result = backtest_event(event, data, prediction_days_before=days_before)
        
        if result['success']:
            results.append(result)
            if days_before == 30:  # Print details for 30-day prediction
                print(f"    ✓ Success! Predicted: {result['predicted_risk']:.1f}, Actual: {result['actual_risk']:.1f}")
                print(f"      Error: {result['absolute_error']:.1f} ({result['percentage_error']:.1f}%)")
                print(f"      Severity: {result['predicted_severity']} vs {result['actual_severity']} "
                      f"{'✓' if result['correct_severity'] else '✗'}")
        else:
            failed_tests.append((event['name'], days_before, result['error']))
            if days_before == 30:
                print(f"    ✗ Failed: {result['error']}")

if len(failed_tests) > 0:
    print(f"\n⚠ {len(failed_tests)} tests failed:")
    for name, days, error in failed_tests[:5]:  # Show first 5
        print(f"  • {name} ({days}d): {error}")

if len(results) == 0:
    print("\n❌ ERROR: No successful backtests!")
    print("\nPossible issues:")
    print("  1. Event dates don't match your data")
    print("  2. State names don't match exactly")
    print("  3. Insufficient historical data")
    print("\nPlease check your data and event definitions.")
    exit()

results_df = pd.DataFrame(results)
print(f"\n✓ Completed {len(results)} successful backtests!")

# ============================================================================
# STEP 5: CALCULATE PERFORMANCE METRICS
# ============================================================================
print("\n[5/6] Calculating validation metrics...")

# Check if we have the required columns
print(f"\nResults DataFrame columns: {list(results_df.columns)}")
print(f"Results DataFrame shape: {results_df.shape}")

# Overall accuracy metrics
overall_metrics = {
    'total_events_tested': len(crisis_events),
    'total_predictions': len(results_df),
    'successful_predictions': len(results_df),
    'failed_predictions': len(failed_tests),
    'avg_absolute_error': results_df['absolute_error'].mean(),
    'avg_percentage_error': results_df['percentage_error'].mean(),
    'median_absolute_error': results_df['absolute_error'].median(),
    'severity_accuracy': (results_df['correct_severity'].sum() / len(results_df)) * 100,
    'alert_accuracy': (results_df['alert_correct'].sum() / len(results_df)) * 100
}

# Performance by lead time
lead_time_performance = results_df.groupby('days_before').agg({
    'absolute_error': 'mean',
    'percentage_error': 'mean',
    'correct_severity': lambda x: (x.sum() / len(x)) * 100
}).round(2)

print("\n" + "=" * 80)
print("VALIDATION RESULTS")
print("=" * 80)

print(f"\nOVERALL PERFORMANCE:")
print(f"  • Events Tested: {overall_metrics['total_events_tested']}")
print(f"  • Successful Predictions: {overall_metrics['successful_predictions']}")
print(f"  • Failed Predictions: {overall_metrics['failed_predictions']}")
print(f"  • Average Error: {overall_metrics['avg_absolute_error']:.2f} risk points ({overall_metrics['avg_percentage_error']:.2f}%)")
print(f"  • Severity Classification Accuracy: {overall_metrics['severity_accuracy']:.1f}%")
print(f"  • Alert Decision Accuracy: {overall_metrics['alert_accuracy']:.1f}%")

print(f"\nPERFORMANCE BY LEAD TIME:")
print(lead_time_performance.to_string())

# Best and worst predictions
if len(results_df) > 0:
    best_prediction = results_df.loc[results_df['absolute_error'].idxmin()]
    worst_prediction = results_df.loc[results_df['absolute_error'].idxmax()]
    
    print(f"\nBEST PREDICTION:")
    print(f"  • Event: {best_prediction['event_name']}")
    print(f"  • Predicted: {best_prediction['predicted_risk']:.2f}, Actual: {best_prediction['actual_risk']:.2f}")
    print(f"  • Error: {best_prediction['absolute_error']:.2f} ({best_prediction['percentage_error']:.2f}%)")
    
    print(f"\nWORST PREDICTION:")
    print(f"  • Event: {worst_prediction['event_name']}")
    print(f"  • Predicted: {worst_prediction['predicted_risk']:.2f}, Actual: {worst_prediction['actual_risk']:.2f}")
    print(f"  • Error: {worst_prediction['absolute_error']:.2f} ({worst_prediction['percentage_error']:.2f}%)")

# Save results immediately
print("\nSaving results...")
results_df.to_csv('../outputs/backtesting_results.csv', index=False)
print("✓ Saved: outputs/backtesting_results.csv")

print("\n" + "=" * 80)
print("✅ BACKTESTING COMPLETE!")
print("=" * 80)
print("\nResults saved. You can now:")
print("  1. Review backtesting_results.csv for details")
print("  2. Use these metrics in your defense")
print("  3. Create visualizations if needed")
