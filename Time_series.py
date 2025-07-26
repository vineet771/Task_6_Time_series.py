
import warnings
warnings.filterwarnings("ignore")  

from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

df = pd.read_csv('sales_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.asfreq('MS') 

plt.figure(figsize=(12, 6))
plt.plot(df['Sales'], label='Original Sales')
plt.title('Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()

df['Moving_Avg'] = df['Sales'].rolling(window=3).mean()
plt.figure(figsize=(12, 6))
plt.plot(df['Sales'], label='Original')
plt.plot(df['Moving_Avg'], label='3-Month Moving Average', color='orange')
plt.title('Sales Trend with Moving Average')
plt.legend()
plt.show()

train = df[:-6]
test = df[-6:]

model = ARIMA(train['Sales'], order=(2, 1, 1))
model_fit = model.fit()

# Forecasting
forecast = model_fit.forecast(steps=len(test))
forecast = pd.Series(forecast, index=test.index)

# Plot Forecast vs Actual
plt.figure(figsize=(12, 6))
plt.plot(train['Sales'], label='Training Data')
plt.plot(test['Sales'], label='Actual Sales')
plt.plot(forecast, label='Forecasted Sales', linestyle='--')
plt.title('ARIMA Forecast vs Actual')
plt.legend()
plt.show()

# Evaluation
# Use squared difference manually since 'squared' argument might not be supported
rmse = (mean_squared_error(test['Sales'], forecast)) ** 0.5
mape = mean_absolute_percentage_error(test['Sales'], forecast)

print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2%}")

forecast_df = pd.DataFrame({'Date': test.index, 'Actual': test['Sales'], 'Forecasted': forecast})
forecast_df.to_csv('forecast_output.csv', index=False)
