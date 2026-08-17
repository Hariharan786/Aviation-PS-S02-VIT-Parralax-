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

# ==========================================
# Usage Example
# ==========================================
if __name__ == "__main__":
    # Initialize the predictor
    predictor = CMAPSS_Predictor('cmapss_sota_rf_pipeline_compressed.joblib')
    
    # Let's load the test data for FD001 as an example
    cols = ['unit', 'cycles', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]
    test_data = pd.read_csv("CMaps/test_FD001.txt", sep=r'\s+', header=None, names=cols)
    
    # Grab all historical data for Engine Unit 1
    engine_1_data = test_data[test_data['unit'] == 1]
    
    # Predict the RUL
    rul = predictor.predict_engine_rul(engine_1_data)
    print(f"\nPredicted Remaining Useful Life (RUL) for Engine Unit 1: {rul:.2f} cycles")
