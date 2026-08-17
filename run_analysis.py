import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn

DATA_DIR = r"c:\Users\Dwarakesh\Downloads\parallax2026\CMaps"
DATASETS = ["FD001", "FD002", "FD003", "FD004"]
SEQUENCE_LENGTH = 50

cols = ['unit', 'cycles', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]

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
    train_df['RUL'] = train_df['RUL'].clip(upper=125)
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

train_list, test_list = [], []
unit_offset = 0

for ds in DATASETS:
    train, test, rul = load_data(ds)
    train = add_rul(train)
    train['unit'] += unit_offset
    test['unit'] += unit_offset
    rul['unit'] = rul.index + 1 + unit_offset
    
    train['dataset'] = ds
    test['dataset'] = ds
    
    train_list.append(train)
    test_list.append((test, rul, ds))
    unit_offset += max(train['unit'].max(), test['unit'].max())

full_train = pd.concat(train_list, ignore_index=True)
features = cols[2:]
scaler = StandardScaler()
full_train[features] = scaler.fit_transform(full_train[features])

def create_sequences(df, features, target='RUL', seq_length=SEQUENCE_LENGTH):
    X, y = [], []
    for unit in df['unit'].unique():
        unit_data = df[df['unit'] == unit]
        data = unit_data[features].values
        labels = unit_data[target].values
        for i in range(len(unit_data) - seq_length + 1):
            X.append(data[i:i+seq_length])
            y.append(labels[i+seq_length-1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_train_seq, y_train_seq = create_sequences(full_train, features)

class RUL_LSTM(nn.Module):
    def __init__(self, input_size):
        super(RUL_LSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.fc1 = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- TRAIN LSTM ---
print("Training LSTM...")
lstm_model = RUL_LSTM(X_train_seq.shape[2]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.005)
from torch.utils.data import DataLoader, TensorDataset
dataset = TensorDataset(torch.tensor(X_train_seq), torch.tensor(y_train_seq).unsqueeze(1))
dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

lstm_model.train()
for epoch in range(3):
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = lstm_model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

# --- TRAIN RF ---
print("Training RF...")
X_train_rf = X_train_seq.reshape(X_train_seq.shape[0], -1)
rf_model = RandomForestRegressor(n_estimators=30, max_depth=10, n_jobs=-1, random_state=42)
rf_model.fit(X_train_rf, y_train_seq)

# --- EVALUATION ---
lstm_model.eval()
results = []
y_true_all, y_pred_lstm_all, y_pred_rf_all = [], [], []

def create_test_sequences(test_df, rul_df, features, seq_length=SEQUENCE_LENGTH):
    X, y = [], []
    units = test_df['unit'].unique()
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
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


for test, rul, ds_name in test_list:
    test[features] = scaler.transform(test[features])
    X_test_seq, y_test_seq = create_test_sequences(test, rul, features)
    
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test_seq).to(device)
        y_pred_lstm = lstm_model(X_test_tensor).cpu().numpy().flatten()
    
    X_test_rf = X_test_seq.reshape(X_test_seq.shape[0], -1)
    y_pred_rf = rf_model.predict(X_test_rf)
    
    y_true_all.extend(y_test_seq)
    y_pred_lstm_all.extend(y_pred_lstm)
    y_pred_rf_all.extend(y_pred_rf)
    
    for model_name, preds in [('LSTM', y_pred_lstm), ('RF', y_pred_rf)]:
        rmse = np.sqrt(mean_squared_error(y_test_seq, preds))
        mae = mean_absolute_error(y_test_seq, preds)
        c_score = cmapss_score(y_test_seq, preds)
        
        results.append({
            'Dataset': ds_name,
            'Model': model_name,
            'RMSE': rmse,
            'MAE': mae,
            'CMAPSS_Score': c_score,
        })

df_results = pd.DataFrame(results)

df_summary = df_results.groupby('Model').agg({
    'RMSE': 'mean',
    'MAE': 'mean',
    'CMAPSS_Score': 'sum'
}).reset_index()

print("\n--- OVERALL SUMMARY ---")
print(df_summary.to_string(index=False))

# Calculate errors (residuals)
# Error = Predicted - True
# Positive error means predicted > true (Late prediction - VERY DANGEROUS)
# Negative error means predicted < true (Early prediction - SAFE)
errors_lstm = np.array(y_pred_lstm_all) - np.array(y_true_all)
errors_rf = np.array(y_pred_rf_all) - np.array(y_true_all)

# Plot 1: True vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_true_all, y_pred_lstm_all, alpha=0.4, label='LSTM', color='blue', s=15)
plt.scatter(y_true_all, y_pred_rf_all, alpha=0.4, label='Random Forest', color='green', s=15)
plt.plot([0, 150], [0, 150], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('True Remaining Useful Life (RUL)')
plt.ylabel('Predicted RUL')
plt.title('True vs Predicted RUL')
plt.legend()
plt.tight_layout()
plt.savefig('true_vs_pred.png')
plt.close()

# Plot 2: Error Distribution (Residuals)
plt.figure(figsize=(10, 6))
sns.histplot(errors_lstm, kde=True, color='blue', label='LSTM Errors', stat='density', alpha=0.5, bins=50)
sns.histplot(errors_rf, kde=True, color='green', label='Random Forest Errors', stat='density', alpha=0.5, bins=50)
plt.axvline(x=0, color='red', linestyle='--', lw=2, label='Zero Error')
plt.xlabel('Prediction Error (Predicted - True RUL)')
plt.ylabel('Density')
plt.title('Error Distribution (Residuals)\nPositive Errors = Late Prediction (Dangerous!)')
plt.legend()
plt.tight_layout()
plt.savefig('error_distribution.png')
plt.close()

print("Plots saved: true_vs_pred.png, error_distribution.png")
