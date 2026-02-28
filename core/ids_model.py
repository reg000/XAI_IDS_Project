import os
import joblib
import pandas as pd
from catboost import CatBoostClassifier

class IDSModel:
    def __init__(self, model_path='models/cat_evolution.cbm', features_path='models/feature_names.joblib'):
        """Loads the pre-trained AI and the required feature list into memory."""
        print("[*] Waking up the AI Engine...")
        
        # Handle pathing whether we run from root folder or inside core/
        if not os.path.exists(model_path):
            model_path = '../' + model_path
            features_path = '../' + features_path
            
        try:
            self.model = CatBoostClassifier()
            self.model.load_model(model_path)
            self.features = joblib.load(features_path)
            print(f"[+] AI Engine Ready. Listening for {len(self.features)} specific features.")
        except Exception as e:
            print(f"[-] Error loading model: {e}")

    def predict(self, feature_dict):
        """
        Takes a dictionary of extracted features from a packet, 
        feeds it to the AI, and returns the verdict.
        """
        # 1. Convert the raw dictionary into a Pandas DataFrame in the exact order the AI expects
        try:
            df = pd.DataFrame([feature_dict], columns=self.features)
        except Exception as e:
            return "ERROR", 0.0, f"Feature mismatch: {e}"

        # 2. Ask the CatBoost model for a prediction
        prediction = self.model.predict(df)[0]
        
        # 3. Get the confidence percentage (e.g., 99.8% sure it's an attack)
        probabilities = self.model.predict_proba(df)[0]
        confidence = probabilities.max() * 100
        
        # 4. Format the output
        if prediction == 1:
            return "🚨 ATTACK (PortScan)", confidence
        else:
            return "✅ NORMAL", confidence

# Quick test to ensure it loads without crashing if you run this file directly
if __name__ == "__main__":
    brain = IDSModel()