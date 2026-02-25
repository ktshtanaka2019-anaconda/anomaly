# -*- coding: utf-8 -*-
"""
AutoEncoder Anomaly Detection Script
-----------------------------------------------
Environment:
  PyOD    == 2.0.6
  PyTorch == 2.10.0-cpu
  Python  >= 3.9 (Anaconda)
  OS      : Linux Fedora 42

Features (6 dimensions):
  1. PowerUsage         : Power consumption (10,000 kW)
  2. SolarPower         : Solar power generation (10,000 kW)
  3. Temperature        : Air temperature (degC)
  4. Humidity           : Relative humidity (%)
  5. WindSpeed          : Average wind speed (m/s)
  6. Precip_D_composite : Precipitation proxy index
                          (Humidity x0.7 + Pressure-change-rate x0.3)

Note on Precipitation "--":
  "--" in JMA data means "no precipitation (< 0.5 mm)".
  Since all rows are "--", a composite proxy index is used
  instead of zero-filling (which would yield zero variance).
-----------------------------------------------
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler

# -- Seed fixation for reproducibility ------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

import torch
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
print(f"PyTorch {torch.__version__}  (CPU build)")

# PyOD 2.x: explicitly use auto_encoder_torch (PyTorch backend)
from pyod.models.auto_encoder import AutoEncoder
import pyod
print(f"PyOD    {pyod.__version__}\n")


# -- Configuration --------------------------------------------------------
CONTAMINATION  = 0.05          # Fraction of data treated as anomalies (top 5%)
TARGET_DATE    = '2026-02-14'
POWER_CSV      = '20260214_power_usage.csv'
WEATHER_CSV    = '20260214tokyo-atomosphere.csv'

# Power CSV: row 54 is the header for 5-min interval data
POWER_SKIPROWS = 54

# Feature columns (update HIDDEN_NEURONS if this list changes)
FEATURE_COLS = [
    'PowerUsage',
    'SolarPower',
    'Temperature',
    'Humidity',
    'WindSpeed',
    'Precip_D_composite',   # Precipitation proxy (composite)
]


# =========================================================================
# 1. Load power data  (5-min interval / cp932)
# =========================================================================
print("=== 1. Loading data ===")

df_power = pd.read_csv(POWER_CSV, encoding='cp932', skiprows=POWER_SKIPROWS)
df_power.columns = ['DATE', 'TIME', 'PowerUsage', 'SolarPower', 'SolarRatio']

df_power['Datetime'] = pd.to_datetime(
    df_power['DATE'].astype(str) + ' ' + df_power['TIME'].astype(str),
    errors='coerce'
)
df_power.dropna(subset=['Datetime'], inplace=True)
df_power.set_index('Datetime', inplace=True)
df_power = df_power[['PowerUsage', 'SolarPower']].apply(pd.to_numeric, errors='coerce')

print(f"Power data   : {len(df_power):>4} rows  "
      f"({df_power.index[0].strftime('%H:%M')} to {df_power.index[-1].strftime('%H:%M')})")


# =========================================================================
# 2. Load weather data + build precipitation proxy  (10-min / UTF-8)
# =========================================================================
df_weather = pd.read_csv(WEATHER_CSV, encoding='utf-8', skiprows=2)
df_weather.columns = [
    'Time', 'Pressure_local', 'Pressure_sea', 'Precipitation',
    'Temperature', 'Humidity', 'WindSpeed_avg',
    'WindDir', 'WindSpeed_max', 'WindDir_max', 'Sunshine'
]

df_weather['Datetime'] = pd.to_datetime(
    TARGET_DATE + ' ' + df_weather['Time'].astype(str),
    errors='coerce'
)
df_weather.dropna(subset=['Datetime'], inplace=True)
df_weather.set_index('Datetime', inplace=True)

for col in ['Pressure_local', 'Temperature', 'Humidity', 'WindSpeed_avg']:
    df_weather[col] = pd.to_numeric(df_weather[col], errors='coerce')

# -- Build Precip_D_composite (Approach 4) --------------------------------
#
# Proxy B: humidity-based  (Humidity - 40) / 60  -> clipped to [0, 1]
#   Dry (40%) = 0,  Saturated (100%) = 1
hum     = df_weather['Humidity'].fillna(df_weather['Humidity'].mean())
proxy_b = ((hum - 40) / 60).clip(0, 1)

# Proxy C: pressure-change-rate  (10-min diff, drop direction only)
#   Pressure drop -> precursor of deteriorating weather
p        = df_weather['Pressure_local'].ffill(limit=None)
dp       = p.diff().fillna(0)
dp_proxy = (-dp).clip(lower=0)
dp_max   = dp_proxy.max()
proxy_c  = (dp_proxy / dp_max) if dp_max > 0 else dp_proxy

# Composite index (weights adjustable)
df_weather['Precip_D_composite'] = 0.7 * proxy_b + 0.3 * proxy_c

# Keep required columns and resample to 5-min (forward-fill)
weather_use = ['Temperature', 'Humidity', 'WindSpeed_avg', 'Precip_D_composite']
df_w5 = df_weather[weather_use].resample('5min').ffill()

print(f"Weather data : {len(df_weather):>4} rows  -> resampled to 5-min: {len(df_w5)} rows")


# =========================================================================
# 3. Merge and standardize
# =========================================================================
df = df_power.join(df_w5, how='inner').ffill().bfill()
df.columns = ['PowerUsage', 'SolarPower', 'Temperature',
              'Humidity', 'WindSpeed', 'Precip_D_composite']
df.dropna(inplace=True)

print(f"Merged data  : {len(df):>4} rows\n")
print("Feature statistics:")
print(df[FEATURE_COLS].describe().round(2))

X        = df[FEATURE_COLS].values
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)


# =========================================================================
# 4. Build and train AutoEncoder
# =========================================================================
print("\n=== 2. AutoEncoder training ===")
print(f"Input dimensions: {len(FEATURE_COLS)}  ({', '.join(FEATURE_COLS)})")

# Network structure (encoder side only; decoder is built symmetrically)
#   Input(6) -> 32 -> 16 -> 8 -> [bottleneck] -> 8 -> 16 -> 32 -> Output(6)
HIDDEN_NEURONS = [32, 16, 8]

clf = AutoEncoder(
    hidden_neuron_list = HIDDEN_NEURONS,
    epoch_num          = 100,
    batch_size         = 16,
    dropout_rate       = 0.1,
    contamination      = CONTAMINATION,
    random_state       = SEED,
    verbose            = 1,
)

clf.fit(X_scaled)

y_pred   = clf.labels_           # 0 = normal / 1 = anomaly
y_scores = clf.decision_scores_  # Reconstruction error (higher = more anomalous)

df['Anomaly'] = y_pred
df['Score']   = y_scores

anomalies = df[df['Anomaly'] == 1].sort_values('Score', ascending=False)


# =========================================================================
# 5. Print result summary
# =========================================================================
print("\n=== 3. Detection results ===")
print(f"Total records    : {len(df)}")
print(f"Anomalies found  : {len(anomalies)}  (contamination={CONTAMINATION:.0%})")
print(f"Decision threshold: {clf.threshold_:.6f}")
print(f"\nTop 10 anomalies by score:")
print(
    anomalies[FEATURE_COLS + ['Score']].head(10).to_string(
        float_format=lambda x: f'{x:.2f}'
    )
)


# =========================================================================
# 6. Visualization (4 panels)
# =========================================================================
print("\n=== 4. Generating plots ===")

fig, axes = plt.subplots(4, 1, figsize=(15, 18), sharex=True)
fig.suptitle(
    f'AutoEncoder Anomaly Detection Results\n'
    f'2026-02-14  |  Features: {len(FEATURE_COLS)}D  |  '
    f'Anomalies: {len(anomalies)} / {len(df)} records',
    fontsize=13, fontweight='bold'
)

# -- Panel (a): Power consumption + Temperature ---------------------------
ax = axes[0]
ax.plot(df.index, df['PowerUsage'],
        color='royalblue', lw=1.5, alpha=0.8, label='Power Usage (10k kW)')
ax.scatter(anomalies.index, anomalies['PowerUsage'],
           color='red', s=70, zorder=5, label=f'Anomaly ({len(anomalies)} pts)')
ax.set_ylabel('Power Usage (10k kW)', color='royalblue')
ax.tick_params(axis='y', labelcolor='royalblue')
ax.grid(True, ls='--', alpha=0.4)

ax_r = ax.twinx()
ax_r.plot(df.index, df['Temperature'],
          color='orange', lw=1.2, alpha=0.6, label='Temperature (degC)')
ax_r.scatter(anomalies.index, anomalies['Temperature'],
             color='darkred', marker='x', s=60, zorder=5)
ax_r.set_ylabel('Temperature (degC)', color='orange')
ax_r.tick_params(axis='y', labelcolor='orange')

lines_l, labels_l = ax.get_legend_handles_labels()
lines_r, labels_r = ax_r.get_legend_handles_labels()
ax.legend(lines_l + lines_r, labels_l + labels_r, loc='upper left', fontsize=8)
ax.set_title('(a) Power Usage and Temperature', fontsize=10)

# -- Panel (b): Solar power + Humidity ------------------------------------
ax = axes[1]
ax.fill_between(df.index, df['SolarPower'],
                color='gold', alpha=0.5, label='Solar Power (10k kW)')
ax.plot(df.index, df['SolarPower'], color='goldenrod', lw=1.0, alpha=0.7)
ax.scatter(anomalies.index, anomalies['SolarPower'],
           color='red', s=70, zorder=5, label='Anomaly')
ax.set_ylabel('Solar Power (10k kW)', color='goldenrod')
ax.tick_params(axis='y', labelcolor='goldenrod')
ax.grid(True, ls='--', alpha=0.4)

ax_r = ax.twinx()
ax_r.plot(df.index, df['Humidity'],
          color='teal', lw=1.2, alpha=0.6, label='Humidity (%)')
ax_r.set_ylabel('Humidity (%)', color='teal')
ax_r.tick_params(axis='y', labelcolor='teal')

lines_l, labels_l = ax.get_legend_handles_labels()
lines_r, labels_r = ax_r.get_legend_handles_labels()
ax.legend(lines_l + lines_r, labels_l + labels_r, loc='upper left', fontsize=8)
ax.set_title('(b) Solar Power and Humidity', fontsize=10)

# -- Panel (c): Precipitation proxy + Wind speed --------------------------
ax = axes[2]
ax.fill_between(df.index, df['Precip_D_composite'],
                color='steelblue', alpha=0.35, label='Precip Proxy (composite)')
ax.plot(df.index, df['Precip_D_composite'],
        color='steelblue', lw=1.2, alpha=0.8)
ax.scatter(anomalies.index, anomalies['Precip_D_composite'],
           color='red', s=70, zorder=5, label='Anomaly')
ax.set_ylabel('Precip Proxy [0-1]', color='steelblue')
ax.tick_params(axis='y', labelcolor='steelblue')
ax.set_ylim(-0.05, 1.05)
ax.grid(True, ls='--', alpha=0.4)

ax_r = ax.twinx()
ax_r.plot(df.index, df['WindSpeed'],
          color='gray', lw=1.2, alpha=0.6, label='Wind Speed (m/s)')
ax_r.set_ylabel('Wind Speed (m/s)', color='gray')
ax_r.tick_params(axis='y', labelcolor='gray')

lines_l, labels_l = ax.get_legend_handles_labels()
lines_r, labels_r = ax_r.get_legend_handles_labels()
ax.legend(lines_l + lines_r, labels_l + labels_r, loc='upper left', fontsize=8)
ax.set_title('(c) Precipitation Proxy (Humidity x0.7 + Pressure-change x0.3) and Wind Speed',
             fontsize=10)

# -- Panel (d): Anomaly score + threshold ---------------------------------
ax = axes[3]
ax.plot(df.index, df['Score'],
        color='purple', lw=1.2, alpha=0.8, label='Anomaly Score')
ax.fill_between(df.index, df['Score'], alpha=0.15, color='purple')
ax.axhline(clf.threshold_, color='red', ls='--', lw=1.8,
           label=f'Threshold = {clf.threshold_:.4f}')
ax.scatter(anomalies.index, anomalies['Score'],
           color='red', s=70, zorder=5, label='Anomaly')
ax.set_ylabel('Reconstruction Error Score')
ax.set_title('(d) Anomaly Score over Time and Decision Threshold', fontsize=10)
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, ls='--', alpha=0.4)

# x-axis formatting
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=2))
plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
axes[-1].set_xlabel('Time (2026-02-14)')

fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=1.5)

output_fig = 'ae_anomaly_result_v3.png'
plt.savefig(output_fig, dpi=150, bbox_inches='tight')
print(f"Plot saved as '{output_fig}'")


# =========================================================================
# 7. Export results to CSV
# =========================================================================
output_csv = 'ae_anomaly_result_v3.csv'
df.to_csv(output_csv, encoding='utf-8-sig')
print(f"Result data saved as '{output_csv}'")
print("\nDone.")
