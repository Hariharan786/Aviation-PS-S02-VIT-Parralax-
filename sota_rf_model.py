import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = r"c:\Users\Dwarakesh\Downloads\parallax2026\CMaps"
DATASETS = ["FD001", "FD002", "FD003", "FD004"]
SEQUENCE_LENGTH = 30
MAX_RUL = 125

cols = ['unit', 'cycles', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]
# Sensors that typically have zero or near-zero variance across CMAPSS datasets
drop_sensors = ['s1', 's5', 's10', 's16', 's18', 's19']
valid_sensors = [f's{i}' for i in range(1, 22) if f's{i}' not in drop_sensors]
settings = ['setting1', 'setting2', 'setting3']

def load_data(dataset_name):
    train = pd.read_csv(os.path.join(DATA_DIR, f"train_{dataset_name}.txt"), sep=r'\s+', header=None, names=cols)
    test = pd.read_csv(os.path.join(DATA_DIR, f"test_{dataset_name}.txt"), sep=r'\s+', header=None, names=cols)
    rul = pd.read_csv(os.path.join(DATA_DIR, f"RUL_{dataset_name}.txt"), sep=r'\s+', header=None, names=['RUL'])
    return train, test, rul

def add_rul(train_df):
    rul = pd.DataFrame(train_df.groupby('unit')['cycles'].max()).reset_index()
    rul.columns = ['unit', 'max_cycles']
    train_df = train_df.merge(rul, on=['unit'], how='left')
    train_df['RUL'] = train_df['max_cycles'] - train_df['cycles']
    train_df.drop('max_cycles', axis=1, inplace=True)
    train_df['RUL'] = train_df['RUL'].clip(upper=MAX_RUL)
    return train_df

def cmapss_score(y_true, y_pred):
    d = y_pred - y_true
    score = 0
    for i in range(len(d)):
        if d[i] < 0:
            score += np.exp(-d[i]/13.0) - 1
        else:
            score += np.exp(d[i]/10.0) - 1
    return score

print("Loading and preparing datasets...")
train_list, test_list = [], []
unit_offset = 0

for ds in DATASETS:
    train, test, rul = load_data(ds)
    train = add_rul(train)
    
    train['unit'] += unit_offset
    test['unit'] += unit_offset
    rul['unit'] = rul.index + 1 + unit_offset
    
    train_list.append(train)
    test_list.append((test, rul, ds))
    unit_offset += max(train['unit'].max(), test['unit'].max())

full_train = pd.concat(train_list, ignore_index=True)
full_train[valid_sensors] = full_train[valid_sensors].astype(np.float64)

print("Step 1: Operating Condition Normalization (K-Means)")
# 6 distinct operating conditions in CMAPSS
kmeans = KMeans(n_clusters=6, random_state=42)
full_train['cluster'] = kmeans.fit_predict(full_train[settings])

# Create scalers for each cluster
cluster_scalers = {}
for cluster_id in range(6):
    scaler = StandardScaler()
    cluster_data = full_train[full_train['cluster'] == cluster_id][valid_sensors]
    if len(cluster_data) > 0:
        scaler.fit(cluster_data)
        cluster_scalers[cluster_id] = scaler

def normalize_by_cluster(df, kmeans_model, scalers, settings_cols, sensor_cols):
    df_copy = df.copy()
    df_copy['cluster'] = kmeans_model.predict(df_copy[settings_cols])
    for cluster_id in range(6):
        idx = df_copy['cluster'] == cluster_id
        if idx.sum() > 0 and cluster_id in scalers:
            df_copy.loc[idx, sensor_cols] = scalers[cluster_id].transform(df_copy.loc[idx, sensor_cols])
    return df_copy

full_train = normalize_by_cluster(full_train, kmeans, cluster_scalers, settings, valid_sensors)

print("Step 2: Exponential Moving Average (EMA) Smoothing")
def apply_ema(df, sensor_cols, span=10):
    # Apply EMA grouping by unit so we don't bleed data across different engines
    df_smoothed = df.copy()
    for col in sensor_cols:
        df_smoothed[col] = df_smoothed.groupby('unit')[col].transform(lambda x: x.ewm(span=span, adjust=False).mean())
    return df_smoothed

full_train = apply_ema(full_train, valid_sensors)

print("Step 3: Creating Rolling Window Sequences")
def create_sequences(df, features, target='RUL', seq_length=SEQUENCE_LENGTH):
    X, y = [], []
    for unit in df['unit'].unique():
        unit_data = df[df['unit'] == unit]
        data = unit_data[features].values
        labels = unit_data[target].values
        for i in range(len(unit_data) - seq_length + 1):
            X.append(data[i:i+seq_length].flatten())
            y.append(labels[i+seq_length-1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_train, y_train = create_sequences(full_train, valid_sensors)

print(f"Training Dataset Shape: {X_train.shape}")
print("Step 4: Training Optimized Random Forest Regressor")
rf_model = RandomForestRegressor(
    n_estimators=100, 
    max_depth=15, 
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1, 
    random_state=42
)
rf_model.fit(X_train, y_train)

print("Step 5: Exporting Pipeline artifacts to disk...")
artifacts = {
    'kmeans_model': kmeans,
    'cluster_scalers': cluster_scalers,
    'rf_model': rf_model,
    'valid_sensors': valid_sensors,
    'settings': settings,
    'sequence_length': SEQUENCE_LENGTH
}
joblib.dump(artifacts, 'cmapss_sota_rf_pipeline.joblib')
print("-> Saved to 'cmapss_sota_rf_pipeline.joblib'")

print("\nStep 6: Evaluating on Test Datasets")
y_true_all, y_pred_all = [], []

def create_test_sequences(test_df, rul_df, features, seq_length=SEQUENCE_LENGTH):
    X, y = [], []
    units = test_df['unit'].unique()
    for unit in units:
        unit_data = test_df[test_df['unit'] == unit]
        data = unit_data[features].values
        if len(data) >= seq_length:
            X.append(data[-seq_length:].flatten())
        else:
            pad = np.zeros((seq_length - len(data), data.shape[1]))
            X.append(np.vstack((pad, data)).flatten())
        true_rul = rul_df[rul_df['unit'] == unit]['RUL'].values[0]
        y.append(true_rul)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

results = []
for test, rul, ds_name in test_list:
    test[valid_sensors] = test[valid_sensors].astype(np.float64)
    # 1. Normalize
    test = normalize_by_cluster(test, kmeans, cluster_scalers, settings, valid_sensors)
    # 2. Smooth
    test = apply_ema(test, valid_sensors)
    # 3. Sequence
    X_test, y_test = create_test_sequences(test, rul, valid_sensors)
    # 4. Predict
    y_pred = rf_model.predict(X_test)
    
    y_true_all.extend(y_test)
    y_pred_all.extend(y_pred)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    c_score = cmapss_score(y_test, y_pred)
    
    results.append({'Dataset': ds_name, 'RMSE': rmse, 'MAE': mae, 'CMAPSS_Score': c_score})

df_results = pd.DataFrame(results)
print("\n--- RESULTS BY DATASET ---")
print(df_results.to_string(index=False))

overall_rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
overall_mae = mean_absolute_error(y_true_all, y_pred_all)
overall_score = cmapss_score(np.array(y_true_all), np.array(y_pred_all))

print("\n--- OVERALL SOTA RESULTS ---")
print(f"Total RMSE:  {overall_rmse:.2f}")
print(f"Total MAE:   {overall_mae:.2f}")
print(f"Total Score: {overall_score:.2f}")
