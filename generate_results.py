# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import json
import os
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# -----------------------------
# CREATE OUTPUT FOLDER (IMPORTANT)
# -----------------------------
os.makedirs("outputs", exist_ok=True)

# -----------------------------
# LOAD DATA (FIXED FOR GITHUB)
# -----------------------------
df = pd.read_csv('data/cleandata.csv')

print(df.head(8))

"""### Time Series Data Preparation for 'LOS'"""

df['Adm_Date'] = pd.to_datetime(df['Adm. Date/Time']).dt.date

daily_los = df.groupby('Adm_Date')['LOS'].mean().reset_index()

min_date = daily_los['Adm_Date'].min()
max_date = daily_los['Adm_Date'].max()
date_range = pd.date_range(start=min_date, end=max_date, freq='D')

daily_los = daily_los.set_index('Adm_Date').reindex(date_range).ffill().reset_index()
daily_los = daily_los.rename(columns={'index': 'Adm_Date'})

"""#### Visualize LOS"""

plt.figure(figsize=(14, 7))
sns.lineplot(x='Adm_Date', y='LOS', data=daily_los)
plt.title('Daily Average LOS')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/los_chart.png")
plt.close()

"""### Model Preparation"""

def create_lagged_features(df, column, num_lags):
    for i in range(1, num_lags + 1):
        df[f'{column}_lag_{i}'] = df[column].shift(i)
    return df

num_lags = 7
daily_los_features = create_lagged_features(daily_los.copy(), 'LOS', num_lags)
daily_los_features.dropna(inplace=True)

X = daily_los_features[[f'LOS_lag_{i}' for i in range(1, num_lags + 1)]]
y = daily_los_features['LOS']

forecast_horizon = 21

X_train = X.iloc[:-forecast_horizon]
X_test = X.iloc[-forecast_horizon:]
y_train = y.iloc[:-forecast_horizon]
y_test = y.iloc[-forecast_horizon:]

model_ts = RandomForestRegressor(n_estimators=100, random_state=42)
model_ts.fit(X_train, y_train)

y_pred_test = model_ts.predict(X_test)

"""### Forecast Future LOS"""

last_known_data = daily_los['LOS'].tail(num_lags).tolist()
future_predictions = []

future_dates = pd.date_range(start=daily_los['Adm_Date'].max() + pd.Timedelta(days=1),
                             periods=forecast_horizon, freq='D')

for _ in range(forecast_horizon):
    input_features = np.array(last_known_data[-num_lags:]).reshape(1, -1)
    next_pred = model_ts.predict(input_features)[0]
    future_predictions.append(next_pred)
    last_known_data.append(next_pred)

future_forecast_df = pd.DataFrame({
    'Adm_Date': future_dates,
    'Forecasted_LOS': future_predictions
})

"""### SAVE LOS FORECAST JSON (FIXED)"""

future_forecast_df.to_json("outputs/los_forecast.json", orient="records")

"""### BED OCCUPANCY SIMULATION"""

daily_admissions_count = df.groupby('Adm_Date').size().reset_index(name='Admissions')
daily_admissions_count['Adm_Date'] = pd.to_datetime(daily_admissions_count['Adm_Date'])

daily_occ = daily_los.copy()
daily_occ['Adm_Date'] = pd.to_datetime(daily_occ['Adm_Date'])
daily_occ = daily_occ.rename(columns={'LOS': 'Avg_LOS'})

daily_occ = daily_occ.merge(daily_admissions_count, on='Adm_Date', how='left').fillna(0)

daily_occ['Occupancy'] = (daily_occ['Admissions'] * daily_occ['Avg_LOS']).clip(upper=80)

"""### XGBOOST MODEL"""

num_lags_occ = 7
for i in range(1, num_lags_occ + 1):
    daily_occ[f'occ_lag_{i}'] = daily_occ['Occupancy'].shift(i)

daily_occ.dropna(inplace=True)

X_occ = daily_occ[[f'occ_lag_{i}' for i in range(1, num_lags_occ + 1)]]
y_occ = daily_occ['Occupancy']

X_train_occ = X_occ.iloc[:-7]
y_train_occ = y_occ.iloc[:-7]

xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train_occ, y_train_occ)

"""### FINAL FORECAST"""

last_occ_values = y_occ.tail(num_lags_occ).tolist()
occ_forecast = []

for _ in range(7):
    inp = np.array(last_occ_values[-num_lags_occ:]).reshape(1, -1)
    pred = xgb_model.predict(inp)[0]
    pred = min(80, max(0, pred))
    occ_forecast.append(pred)
    last_occ_values.append(pred)

forecast_dates = pd.date_range(start=daily_occ['Adm_Date'].max() + pd.Timedelta(days=1), periods=7)

final_tuned_df = pd.DataFrame({
    'Date': forecast_dates,
    'Tuned_Predicted_Occupancy': occ_forecast
})

"""### SAVE FINAL JSON (IMPORTANT FIX)"""

final_tuned_df.to_json("outputs/finaloccupancy.json", orient="records")

"""### SAVE CHARTS (FIXED)"""

plt.figure(figsize=(12,6))
sns.lineplot(x='Date', y='Tuned_Predicted_Occupancy', data=final_tuned_df)
plt.axhline(y=80, color='red')
plt.tight_layout()
plt.savefig("outputs/occupancychart.png")
plt.close()

plt.figure(figsize=(12,6))
sns.lineplot(x='Adm_Date', y='Occupancy', data=daily_occ)
plt.tight_layout()
plt.savefig("outputs/demandchart.png")
plt.close()
