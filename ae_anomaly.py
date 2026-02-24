# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# PyODのPyTorch版 AutoEncoder をインポート
from pyod.models.auto_encoder import AutoEncoder

# 設定
contamination = 0.05  # 外れ値の混入率 (上位5%を特異点とみなす)
power_csv = '20260214_power_usage.csv'
weather_csv = '20260214tokyo-atomosphere.csv'
target_date = '2026-02-14' # データの日付

print("データの読み込みと前処理を開始します...")

# ---------------------------------------------------------
# 1. 電力データの読み込み (5分間隔 / Shift-JIS)
# ---------------------------------------------------------
power_skip = 54
# 東電のデータは cp932 (Shift-JIS) で読み込む
df_power = pd.read_csv(power_csv, encoding='cp932', skiprows=power_skip)
df_power['Datetime'] = pd.to_datetime(df_power['DATE'] + ' ' + df_power['TIME'])
df_power.set_index('Datetime', inplace=True)

cols_power = ['当日実績(５分間隔値)(万kW)', '太陽光発電実績(５分間隔値)(万kW)']
df_power = df_power[cols_power].copy()
df_power.columns = ['PowerUsage', 'SolarPower']

for col in df_power.columns:
    df_power[col] = pd.to_numeric(df_power[col], errors='coerce')


# ---------------------------------------------------------
# 2. 気象データの読み込み (10分間隔 / UTF-8)
# ---------------------------------------------------------
# BeautifulSoup経由等で取得したデータは utf-8
df_weather = pd.read_csv(weather_csv, encoding='utf-8', skiprows=2)

df_weather['Datetime'] = pd.to_datetime(target_date + ' ' + df_weather.iloc[:, 0].astype(str), errors='coerce')
df_weather.dropna(subset=['Datetime'], inplace=True)
df_weather.set_index('Datetime', inplace=True)

df_weather = df_weather.iloc[:, [4, 5, 6]].copy()
df_weather.columns = ['Temperature', 'Humidity', 'WindSpeed']

for col in df_weather.columns:
    df_weather[col] = pd.to_numeric(df_weather[col], errors='coerce')

# 10分間隔の気象データを、電力データに合わせて5分間隔にリサンプリング（前方補完）
df_weather_resampled = df_weather.resample('5min').ffill()


# ---------------------------------------------------------
# 3. データの結合と標準化
# ---------------------------------------------------------
df_merged = df_power.join(df_weather_resampled, how='inner')
df_merged = df_merged.ffill().bfill()

print(f"データ結合完了: 全 {len(df_merged)} 件の統合データを作成しました。")

X = df_merged.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ---------------------------------------------------------
# 4. AutoEncoderモデルの構築と学習 (PyTorchベース)
# ---------------------------------------------------------
print("AutoEncoderによる特異点検知を実行中...")
# PyOD(PyTorch版)の正しいパラメータに修正
clf = AutoEncoder(
    hidden_neuron_list=[5, 3, 3, 5], # 修正: hidden_neurons -> hidden_neuron_list
    epoch_num=50,                    # 修正: epochs -> epoch_num
    batch_size=32,
    contamination=contamination,
    dropout_rate=0.2,                # オプション: 過学習防止
    random_state=42
)

clf.fit(X_scaled)

y_pred = clf.labels_
y_scores = clf.decision_scores_

df_merged['Anomaly'] = y_pred
df_merged['Score'] = y_scores


# ---------------------------------------------------------
# 5. 結果の出力と可視化
# ---------------------------------------------------------
anomalies = df_merged[df_merged['Anomaly'] == 1]
print(f"\n--- 特異点検知結果 ---")
print(f"検出された特異点（外れ値）の数: {len(anomalies)}件\n")

print("▼ 特異点データ（異常度スコア降順、上位10件）")
print(anomalies.sort_values('Score', ascending=False).head(10))

# グラフ描画
fig, ax1 = plt.subplots(figsize=(14, 7))

ax1.plot(df_merged.index, df_merged['PowerUsage'], label='Power Usage', color='blue', alpha=0.6)
ax1.scatter(anomalies.index, anomalies['PowerUsage'], color='red', s=50, label='Anomaly (Power)', zorder=5)
ax1.set_xlabel('Datetime')
ax1.set_ylabel('Power Usage (10k kW)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True, linestyle='--', alpha=0.5)

ax2 = ax1.twinx()
ax2.plot(df_merged.index, df_merged['Temperature'], label='Temperature', color='orange', alpha=0.6)
ax2.scatter(anomalies.index, anomalies['Temperature'], color='darkred', marker='x', s=50, label='Anomaly (Temp)', zorder=5)
ax2.set_ylabel('Temperature (℃)', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.title('AutoEncoder Anomaly Detection: Power Usage & Weather')
fig.tight_layout()

plt.savefig('ae_anomaly_result.png')
print("\nグラフを 'ae_anomaly_result.png' として保存しました。")
# plt.show() # 必要に応じてコメントアウトを外す
