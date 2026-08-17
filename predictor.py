import joblib
import pandas as pd
import numpy as np

class CMAPSS_Predictor:
    def __init__(self, model_path='cmapss_sota_rf_pipeline_compressed.joblib'):
        """
        Loads the compressed SOTA Random Forest pipeline, including 
        all necessary K-Means clusters and scalers.
        """
        print(f"Loading SOTA Pipeline from {model_path}...")
        self.artifacts = joblib.load(model_path)
        
        self.kmeans = self.artifacts['kmeans_model']
        self.scalers = self.artifacts['cluster_scalers']
        self.rf_model = self.artifacts['rf_model']
        self.valid_sensors = self.artifacts['valid_sensors']
        self.settings = self.artifacts['settings']
        self.seq_length = self.artifacts['sequence_length']
        print("Pipeline loaded successfully!")

    def predict_engine_rul(self, engine_dataframe):
        """
        Predicts the Remaining Useful Life (RUL) of a single engine.
        
        Parameters:
        engine_dataframe (pd.DataFrame): Historical cycle data for one engine.
                                         Must contain 'setting1', 'setting2', 'setting3' 
                                         and the 21 sensor columns ('s1' to 's21').
                                         
        Returns:
        float: The predicted Remaining Useful Life (RUL) in cycles.
        """
        # 1. Create a copy to avoid mutating the original data
        df = engine_dataframe.copy()
        
        # Cast sensors to float64 to prevent Pandas LossySetitemError during normalization
        df[self.valid_sensors] = df[self.valid_sensors].astype(np.float64)
        
        # 2. Operating Condition Normalization (Identify flight regime via K-Means)
        df['cluster'] = self.kmeans.predict(df[self.settings])
        
        for cluster_id in range(6):
            idx = df['cluster'] == cluster_id
            if idx.sum() > 0 and cluster_id in self.scalers:
                df.loc[idx, self.valid_sensors] = self.scalers[cluster_id].transform(df.loc[idx, self.valid_sensors])
                
        # 3. Exponential Moving Average (EMA) Smoothing (Filter mechanical noise)
        for col in self.valid_sensors:
            df[col] = df[col].astype(np.float64).ewm(span=10, adjust=False).mean()
            
        # 4. Extract and flatten the last sequence window for the model
        data = df[self.valid_sensors].values
        if len(data) >= self.seq_length:
            X = data[-self.seq_length:].flatten()
        else:
            # Zero-pad if the engine hasn't recorded enough cycles yet
            pad = np.zeros((self.seq_length - len(data), data.shape[1]))
            X = np.vstack((pad, data)).flatten()
            
        # 5. Predict the RUL
        predicted_rul = self.rf_model.predict([X])[0]
        return predicted_rul

    def predict_batch(self, filepath, critical_threshold=30):
        """
        Reads a CSV or TXT file containing multiple engines, predicts the RUL for each,
        classifies them as Normal or Abnormal, and generates a plot.
        
        Parameters:
        filepath (str): Path to the .csv or .txt file.
        critical_threshold (int): RUL below this value is classified as 'Abnormal/Critical'.
        """
        import matplotlib.pyplot as plt
        import os
        
        print(f"Loading data from {filepath}...")
        if filepath.endswith('.txt'):
            # CMAPSS raw format
            cols = ['unit', 'cycles', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]
            df = pd.read_csv(filepath, sep=r'\s+', header=None, names=cols)
        else:
            # Standard CSV format (assumes it has the proper headers)
            df = pd.read_csv(filepath)
            
        units = df['unit'].unique()
        print(f"Found {len(units)} engines in the dataset. Predicting RUL...")
        
        results = []
        for unit_id in units:
            unit_data = df[df['unit'] == unit_id]
            rul = self.predict_engine_rul(unit_data)
            status = "Abnormal (Critical)" if rul <= critical_threshold else "Normal"
            results.append({'Engine': unit_id, 'Predicted RUL': rul, 'Status': status})
            
        results_df = pd.DataFrame(results)
        
        # --- Plotting ---
        plt.figure(figsize=(15, 6))
        colors = ['red' if s == 'Abnormal (Critical)' else 'green' for s in results_df['Status']]
        bars = plt.bar(results_df['Engine'].astype(str), results_df['Predicted RUL'], color=colors)
        
        plt.axhline(y=critical_threshold, color='black', linestyle='--', label=f'Critical Threshold ({critical_threshold} cycles)')
        plt.title('Predicted Remaining Useful Life (RUL) per Engine')
        plt.xlabel('Engine Unit ID')
        plt.ylabel('Predicted RUL (Cycles)')
        plt.xticks(rotation=90, fontsize=8)
        plt.legend()
        plt.tight_layout()
        
        plot_path = "rul_predictions_plot.png"
        plt.savefig(plot_path)
        plt.close()
        
        print(f"\nPredictions complete! Plot saved as '{plot_path}'")
        print("\n--- Prediction Summary ---")
        print(f"Total Engines: {len(results_df)}")
        print(f"Normal Engines: {sum(results_df['Status'] == 'Normal')}")
        print(f"Abnormal Engines: {sum(results_df['Status'] != 'Normal')}")
        
        return results_df

# ==========================================
# Usage Example
# ==========================================
if __name__ == "__main__":
    # Initialize the predictor
    predictor = CMAPSS_Predictor('cmapss_sota_rf_pipeline_compressed.joblib')
    
    # Process an entire file (TXT or CSV) and generate the plot!
    # By default, any engine with RUL <= 30 is flagged as Abnormal.
    results = predictor.predict_batch("CMaps/test_FD001.txt", critical_threshold=30)
    
    # Print the first 10 predictions
    print("\nFirst 10 Engine Predictions:")
    print(results.head(10).to_string(index=False))
