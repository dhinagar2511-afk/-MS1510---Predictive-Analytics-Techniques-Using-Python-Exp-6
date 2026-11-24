# Experiment 6: ARIMA Model for Forecasting

## Aim

To evaluate predictive models using performance metrics.

## Algorithm

1.  **START**
2.  Prepare stationary time series.
3.  Identify ARIMA -- (p, d, q) parameters using ACF/PACF.
4.  Fit the ARIMA model on the data.
5.  Forecast future periods and plot the results.
6.  Evaluate forecast accuracy.
7.  **STOP**

## Program

```python
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Step 2: Prepare the time series (synthetic monthly data with an upward trend)
np.random.seed(42)
n = 120
dates = pd.date_range("2015-01-01", periods=n, freq="MS")
series = pd.Series(100 + 0.8 * np.arange(n) + np.cumsum(np.random.normal(0, 2, n)),
                   index=dates)
print(series.head())

# Train-test split (last 12 months for testing)
train, test = series[:-12], series[-12:]

# Stationarity check (ADF test)
def adf(x, label):
    stat, p = adfuller(x)[:2]
    print(f"ADF test on {label}: statistic = {stat:.4f}, p-value = {p:.4f}")

adf(train, "original series")
diff1 = train.diff().dropna()
adf(diff1, "first difference")

# Step 3: ACF and PACF of the differenced series
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(diff1, lags=20, ax=axes[0])
plot_pacf(diff1, lags=20, ax=axes[1])
plt.tight_layout()
plt.savefig("acf_pacf.png", dpi=120, bbox_inches="tight")
plt.show()

# Step 4: Fit ARIMA(p, d, q) - no significant ACF/PACF lags, so p = 0, q = 0, d = 1 (with drift)
model = ARIMA(train, order=(0, 1, 0), trend="t")
fit = model.fit()
print("\n", fit.summary().tables[1])
print("AIC:", round(fit.aic, 2))

# Step 5: Forecast and plot
forecast = fit.get_forecast(steps=12)
pred = forecast.predicted_mean
ci = forecast.conf_int()

plt.figure(figsize=(10, 4))
plt.plot(train, label="Train")
plt.plot(test, label="Actual (Test)", color="green")
plt.plot(pred, label="Forecast", color="red", linestyle="--")
plt.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color="pink", alpha=0.3)
plt.title("ARIMA(0,1,0) with Drift Forecast")
plt.legend()
plt.grid(True)
plt.savefig("arima_forecast.png", dpi=120, bbox_inches="tight")
plt.show()

# Step 6: Evaluate forecast accuracy
mae = mean_absolute_error(test, pred)
rmse = np.sqrt(mean_squared_error(test, pred))
mape = np.mean(np.abs((test - pred) / test)) * 100
print(f"\nMAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.2f}%")
```

## Output

```
2015-01-01    100.993428
2015-02-01    101.516900
2015-03-01    103.612277
2015-04-01    107.458336
2015-05-01    107.790030
Freq: MS, dtype: float64
ADF test on original series: statistic = 0.0633, p-value = 0.9635
ADF test on first difference: statistic = -10.1163, p-value = 0.0000

 ==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
x1             0.5839      0.176      3.325      0.001       0.240       0.928
sigma2         3.2881      0.464      7.082      0.000       2.378       4.198
==============================================================================
AIC: 435.01

MAE  : 2.7582
RMSE : 3.0630
MAPE : 1.62%
```

**ACF and PACF of the differenced series**

<img width="1428" height="468" alt="image" src="https://github.com/user-attachments/assets/a680b697-3b83-481e-83c0-353eff7c5c50" />


**ARIMA forecast vs actual values**

<img width="993" height="454" alt="image" src="https://github.com/user-attachments/assets/3f1693a8-9cfc-4460-8564-281a0efeaaca" />


## Result

The program was executed successfully. An ARIMA model was fitted on 108
monthly observations and used to forecast the next 12 months.

-   The original series is non-stationary (ADF p-value = 0.96). After
    first differencing it is stationary (p-value < 0.001), so **d = 1**.
-   The ACF and PACF of the differenced series show no significant lags,
    so **p = 0** and **q = 0**. The selected model is **ARIMA(0,1,0) with
    drift** (estimated drift ≈ 0.58 per month).
-   Forecast accuracy on the 12 test months: **MAE = 2.76**,
    **RMSE = 3.06**, **MAPE = 1.62%**.
-   The actual values stay inside the 95% confidence band of the forecast.

Thus, the ARIMA model forecast the time series with low error, and the
performance was evaluated using MAE, RMSE and MAPE.
