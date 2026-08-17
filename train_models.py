import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys

# Attempt to import tensorflow, install if missing
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:
    print("TensorFlow not found. Please install it using: pip install tensorflow")
    sys.exit(1)

# Dataset configuration
DATA_DIR = r"c:\Users\Dwarakesh\Downloads\parallax2026\CMaps"
DATASETS = ["FD001", "FD002", "FD003", "FD004"]
SEQUENCE_LENGTH = 50

# Column names
cols = ['unit', 'cycles', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]

def load_data(dataset_name):
    train_file = os.path.join(DATA_DIR, f"train_{dataset_name}.txt")
    test_file = os.path.join(DATA_DIR, f"test_{dataset_name}.txt")
    rul_file = os.path.join(DATA_DIR, f"RUL_{dataset_name}.txt")
    
    train = pd.read_csv(train_file, sep='\s+', header=None, names=cols)
    test = pd.read_csv(test_file, sep='\s+', header=None, names=cols)
    rul = pd.read_csv(rul_file, sep='\s+', header=None, names=['RUL'])
    
    return train, test, rul

def add_rul(train_df):
    # Calculate true RUL based on max cycle per unit
    rul = pd.DataFrame(train_df.groupby('unit')['cycles'].max()).reset_index()
    rul.columns = ['unit', 'max_cycles']
    train_df = train_df.merge(rul, on=['unit'], how='left')
    train_df['RUL'] = train_df['max_cycles'] - train_df['cycles']
    train_df.drop('max_cycles', axis=1, inplace=True)
    # Piecewise linear degradation capping
    train_df['RUL'] = train_df['RUL'].clip(upper=125)
    return train_df

def process_datasets():
    train_list = []
    test_list = []
    
    # We append a unique dataset ID to units so they don't overlap across FD001-004
    unit_offset = 0
    
    for ds in DATASETS:
        train, test, rul = load_data(ds)
        
        train = add_rul(train)
        
        train['unit'] += unit_offset
        test['unit'] += unit_offset
        
        # For test, we append the actual RUL values to the LAST cycle of each unit
        # to evaluate our model
        rul['unit'] = rul.index + 1 + unit_offset
        
        train_list.append(train)
        test_list.append((test, rul))
        
        unit_offset += max(train['unit'].max(), test['unit'].max())
        
    full_train = pd.concat(train_list, ignore_index=True)
    
    # Standardize
    features = cols[2:]
    scaler = StandardScaler()
    full_train[features] = scaler.fit_transform(full_train[features])
    
    # Standardize test sets
    test_frames = []
    for test, rul in test_list:
        test[features] = scaler.transform(test[features])
        test_frames.append((test, rul))
        
    return full_train, test_frames, features

def create_sequences(df, features, target='RUL', seq_length=SEQUENCE_LENGTH):
    X, y = [], []
    for unit in df['unit'].unique():
        unit_data = df[df['unit'] == unit]
        data = unit_data[features].values
        labels = unit_data[target].values
        
        for i in range(len(unit_data) - seq_length + 1):
            X.append(data[i:i+seq_length])
            y.append(labels[i+seq_length-1])
            
    return np.array(X), np.array(y)

def create_test_sequences(test_df, rul_df, features, seq_length=SEQUENCE_LENGTH):
    # Only use the last sequence for each unit to match the single RUL provided
    X, y = [], []
    units = test_df['unit'].unique()
    
    # Check if there are units that have fewer cycles than seq_length
    # For simplicity, we pad them with zeros at the beginning
    for unit in units:
        unit_data = test_df[test_df['unit'] == unit]
        data = unit_data[features].values
        
        if len(data) >= seq_length:
            X.append(data[-seq_length:])
        else:
            pad = np.zeros((seq_length - len(data), data.shape[1]))
            X.append(np.vstack((pad, data)))
            
        true_rul = rul_df[rul_df['unit'] == unit]['RUL'].values[0]
        y.append(true_rul)
        
    return np.array(X), np.array(y)

def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

print("Processing datasets...")
full_train, test_list, features = process_datasets()

print("Creating sequences for LSTM (this might take a minute)...")
X_train_seq, y_train_seq = create_sequences(full_train, features)

print(f"Train sequence shape: {X_train_seq.shape}")

print("Building and training LSTM...")
lstm_model = build_lstm((X_train_seq.shape[1], X_train_seq.shape[2]))
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lstm_history = lstm_model.fit(
    X_train_seq, y_train_seq,
    epochs=10, 
    batch_size=256,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

print("\nTraining Random Forest...")
# For Random Forest, we will flatten the time sequence to create 2D features
X_train_rf = X_train_seq.reshape(X_train_seq.shape[0], -1)
rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
rf_model.fit(X_train_rf, y_train_seq)

print("\nEvaluating on Test Sets...")
y_pred_lstm_all, y_pred_rf_all, y_true_all = [], [], []

for test, rul in test_list:
    X_test_seq, y_test_seq = create_test_sequences(test, rul, features)
    
    y_pred_lstm = lstm_model.predict(X_test_seq, verbose=0).flatten()
    
    X_test_rf = X_test_seq.reshape(X_test_seq.shape[0], -1)
    y_pred_rf = rf_model.predict(X_test_rf)
    
    y_pred_lstm_all.extend(y_pred_lstm)
    y_pred_rf_all.extend(y_pred_rf)
    y_true_all.extend(y_test_seq)

lstm_rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_lstm_all))
lstm_mae = mean_absolute_error(y_true_all, y_pred_lstm_all)

rf_rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_rf_all))
rf_mae = mean_absolute_error(y_true_all, y_pred_rf_all)

print("\n" + "="*40)
print("FINAL RESULTS ACROSS ALL 4 DATASETS:")
print(f"LSTM Model         -> RMSE: {lstm_rmse:.2f}, MAE: {lstm_mae:.2f}")
print(f"Random Forest Model -> RMSE: {rf_rmse:.2f}, MAE: {rf_mae:.2f}")
print("="*40)

# Save metrics to a file for the agent to read
with open("metrics.txt", "w") as f:
    f.write(f"LSTM RMSE: {lstm_rmse:.2f}\n")
    f.write(f"LSTM MAE: {lstm_mae:.2f}\n")
    f.write(f"RF RMSE: {rf_rmse:.2f}\n")
    f.write(f"RF MAE: {rf_mae:.2f}\n")

# Visualization
plt.figure(figsize=(10, 5))
plt.scatter(y_true_all, y_pred_lstm_all, alpha=0.3, label='LSTM', color='blue')
plt.scatter(y_true_all, y_pred_rf_all, alpha=0.3, label='Random Forest', color='green')
plt.plot([0, 150], [0, 150], 'r--')
plt.xlabel('True RUL')
plt.ylabel('Predicted RUL')
plt.title('True vs Predicted RUL (All Datasets)')
plt.legend()
plt.savefig('rul_comparison.png')
print("Saved comparison plot to rul_comparison.png")
