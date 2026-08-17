import pandas as pd
import numpy as np
import joblib

def normalize_by_cluster(df, kmeans_model, scalers, settings_cols, sensor_cols):
    """Normalize sensor data based on K-Means operating condition clusters."""
    df_copy = df.copy()
    df_copy['cluster'] = kmeans_model.predict(df_copy[settings_cols])
    for cluster_id in range(6):
        idx = df_copy['cluster'] == cluster_id
        if idx.sum() > 0 and cluster_id in scalers:
            df_copy.loc[idx, sensor_cols] = scalers[cluster_id].transform(df_copy.loc[idx, sensor_cols])
    return df_copy

def apply_ema(df, sensor_cols, span=10):
    """Apply Exponential Moving Average smoothing."""
    df_smoothed = df.copy()
    if 'unit' in df_smoothed.columns:
        for col in sensor_cols:
            df_smoothed[col] = df_smoothed.groupby('unit')[col].transform(lambda x: x.ewm(span=span, adjust=False).mean())
    else:
        for col in sensor_cols:
            df_smoothed[col] = df_smoothed[col].ewm(span=span, adjust=False).mean()
    return df_smoothed

def predict_rul_for_engine(engine_history_df, model_path='cmapss_sota_rf_pipeline.joblib'):
    """
    Predict the Remaining Useful Life (RUL) for a single engine given its history.
    
    Args:
        engine_history_df (pd.DataFrame): The time-series data for a specific engine up to the current cycle.
                                          Must contain the setting columns (setting1, setting2, setting3)
                                          and all valid sensor columns (s2, s3, s4, etc.).
        model_path (str): Path to the saved joblib model artifact.
        
    Returns:
        float: The predicted RUL for this engine.
    """
    # 1. Load the saved pipeline artifacts
    try:
        pipeline = joblib.load(model_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at {model_path}. Make sure you run sota_rf_model.py first to train and save the model.")
        
    kmeans = pipeline['kmeans_model']
    cluster_scalers = pipeline['cluster_scalers']
    rf_model = pipeline['rf_model']
    valid_sensors = pipeline['valid_sensors']
    settings = pipeline['settings']
    seq_length = pipeline['sequence_length']
    
    # Ensure correct data types
    engine_history_df[valid_sensors] = engine_history_df[valid_sensors].astype(np.float64)
    
    # 2. Normalize by operating condition clusters
    df_processed = normalize_by_cluster(engine_history_df, kmeans, cluster_scalers, settings, valid_sensors)
    
    # 3. Apply EMA smoothing
    df_processed = apply_ema(df_processed, valid_sensors)
    
    # 4. Prepare sequence (last N cycles)
    data = df_processed[valid_sensors].values
    if len(data) >= seq_length:
        X = data[-seq_length:].flatten()
    else:
        # Zero-pad if the engine hasn't reached the sequence length yet
        pad = np.zeros((seq_length - len(data), data.shape[1]))
        X = np.vstack((pad, data)).flatten()
        
    # Reshape for sklearn (1 sample, n_features)
    X = X.reshape(1, -1)
    
    # 5. Predict and return
    prediction = rf_model.predict(X)[0]
    return prediction

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Predict RUL for an engine using a saved Random Forest pipeline.")
    parser.add_argument("--input", "-i", type=str, help="Path to a CSV file containing the engine's sensor data.")
    parser.add_argument("--model", "-m", type=str, default="cmapss_sota_rf_pipeline.joblib", help="Path to the saved model (.joblib).")
    
    args = parser.parse_args()
    
    if args.input:
        if not os.path.exists(args.input):
            print(f"Error: Input file '{args.input}' not found.")
        else:
            print(f"Loading data from {args.input}...")
            # Load the user's CSV file
            input_df = pd.read_csv(args.input)
            
            try:
                predicted_rul = predict_rul_for_engine(input_df, args.model)
                print("\n" + "="*50)
                print(f" SUCCESS: Predicted Remaining Useful Life (RUL): {predicted_rul:.2f} cycles")
                print("="*50 + "\n")
            except Exception as e:
                print(f"Error during prediction: {e}")
    else:
        # Fallback to the original example if no arguments are provided
        print("No input file provided. Running default example on CMaps/test_FD001.txt...\n")
        DATA_DIR = "CMaps"
        test_file = os.path.join(DATA_DIR, "test_FD001.txt")
        
        if os.path.exists(test_file):
            cols = ['unit', 'cycles', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]
            test_df = pd.read_csv(test_file, sep=r'\s+', header=None, names=cols)
            
            # Extract data for a single engine (e.g., unit 1)
            engine_1_data = test_df[test_df['unit'] == 1].copy()
            
            print(f"Data shape for Engine #1: {engine_1_data.shape}")
            predicted_rul = predict_rul_for_engine(engine_1_data, args.model)
            print(f"Predicted Remaining Useful Life (RUL) for Engine #1: {predicted_rul:.2f} cycles")
            
            # Optionally save this single engine to a sample CSV for the user
            sample_path = "sample_engine_data.csv"
            engine_1_data.to_csv(sample_path, index=False)
            print(f"\nTip: I saved this engine's data to '{sample_path}'.")
            print(f"You can now test the script from the terminal by running:")
            print(f"python predict.py --input {sample_path}")
        else:
            print(f"Could not find {test_file} for example run.")
            print("To use this script, run: python predict.py --input your_data.csv")
