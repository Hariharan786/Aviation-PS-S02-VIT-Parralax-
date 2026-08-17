import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_on_training_sets(model_path='cmapss_sota_rf_pipeline_compressed.joblib', data_dir='CMaps'):
    print(f"Loading SOTA Pipeline from {model_path}...")
    artifacts = joblib.load(model_path)
    
    kmeans = artifacts['kmeans_model']
    scalers = artifacts['cluster_scalers']
    rf_model = artifacts['rf_model']
    valid_sensors = artifacts['valid_sensors']
    settings = artifacts['settings']
    seq_length = artifacts['sequence_length']
    MAX_RUL = 125
    
    cols = ['unit', 'cycles', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]
    datasets = ["FD001", "FD002", "FD003", "FD004"]
    
    overall_y_true = []
    overall_y_pred = []
    
    results = []
    
    for ds in datasets:
        print(f"\nProcessing Training Dataset: {ds}...")
        filepath = os.path.join(data_dir, f"train_{ds}.txt")
        df = pd.read_csv(filepath, sep=r'\s+', header=None, names=cols)
        
        # 1. Calculate True RUL (clipped)
        rul = pd.DataFrame(df.groupby('unit')['cycles'].max()).reset_index()
        rul.columns = ['unit', 'max_cycles']
        df = df.merge(rul, on=['unit'], how='left')
        df['RUL'] = df['max_cycles'] - df['cycles']
        df['RUL'] = df['RUL'].clip(upper=MAX_RUL)
        
        # 2. Normalize by operating condition
        df[valid_sensors] = df[valid_sensors].astype(np.float64)
        df['cluster'] = kmeans.predict(df[settings])
        for cluster_id in range(6):
            idx = df['cluster'] == cluster_id
            if idx.sum() > 0 and cluster_id in scalers:
                df.loc[idx, valid_sensors] = scalers[cluster_id].transform(df.loc[idx, valid_sensors])
                
        # 3. Apply EMA Smoothing
        for col in valid_sensors:
            df[col] = df.groupby('unit')[col].transform(lambda x: x.ewm(span=10, adjust=False).mean())
            
        # 4. Create sequences
        X, y = [], []
        for unit in df['unit'].unique():
            unit_data = df[df['unit'] == unit]
            data = unit_data[valid_sensors].values
            labels = unit_data['RUL'].values
            
            for i in range(len(unit_data) - seq_length + 1):
                X.append(data[i:i+seq_length].flatten())
                y.append(labels[i+seq_length-1])
                
        X = np.array(X, dtype=np.float32)
        y_true = np.array(y, dtype=np.float32)
        
        # 5. Predict in bulk
        y_pred = rf_model.predict(X)
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        results.append({'Dataset': ds, 'RMSE': rmse, 'MAE': mae})
        
        overall_y_true.extend(y_true)
        overall_y_pred.extend(y_pred)
        
    df_results = pd.DataFrame(results)
    print("\n--- RESULTS ON TRAINING DATASETS ---")
    print(df_results.to_string(index=False))
    
    overall_rmse = np.sqrt(mean_squared_error(overall_y_true, overall_y_pred))
    overall_mae = mean_absolute_error(overall_y_true, overall_y_pred)
    
    print("\n--- TOTAL TRAINING SET PERFORMANCE ---")
    print(f"Total RMSE: {overall_rmse:.2f}")
    print(f"Total MAE:  {overall_mae:.2f}")

if __name__ == "__main__":
    evaluate_on_training_sets()
