# Quick Prophet Test
# Save this as: test_prophet.py
# Run: python test_prophet.py

print("Testing Prophet installation...")

try:
    from prophet import Prophet
    import pandas as pd
    import numpy as np
    
    print("✅ Prophet imported successfully!")
    
    # Create dummy data
    dates = pd.date_range('2024-01-01', periods=12, freq='MS')
    values = [50 + 10*np.sin(i/2) + np.random.normal(0, 2) for i in range(12)]
    
    df = pd.DataFrame({
        'ds': dates,
        'y': values
    })
    
    print("✅ Sample data created")
    
    # Try to fit a model
    model = Prophet()
    model.fit(df)
    
    print("✅ Model trained successfully!")
    
    # Make forecast
    future = model.make_future_dataframe(periods=3, freq='MS')
    forecast = model.predict(future)
    
    print("✅ Forecast generated successfully!")
    print(f"\nForecast for next 3 months:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(3))
    
    print("\n" + "="*60)
    print("🎉 Prophet is working perfectly!")
    print("="*60)
    print("\nYou can now run your 06_prophet_forecasting.ipynb notebook.")
    
except ImportError as e:
    print("❌ Prophet not installed properly")
    print(f"Error: {e}")
    print("\nTry installing with:")
    print("  pip install prophet")
    print("  OR")
    print("  conda install -c conda-forge prophet")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nThere might be an issue with your Prophet installation.")
