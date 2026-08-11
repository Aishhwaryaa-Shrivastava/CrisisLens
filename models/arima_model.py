from statsmodels.tsa.arima.model import ARIMA

def run_arima(data):
    model = ARIMA(data, order=(1,1,1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=10)
    return forecast