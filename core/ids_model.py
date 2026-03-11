import os
import joblib
import pandas as pd
from catboost import CatBoostClassifier

class IDSModel:
    def __init__(self, model_path='models/cat_hybrid_v3_1.cbm', features_path='models/feature_names_v3_1.joblib'):
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
        probabilities = self.model.predict_proba(df)[0]
        confidence = probabilities.max() * 100
        
        # Extract features for heuristic checks
        duration = float(feature_dict.get(' Flow Duration', 0))
        port = float(feature_dict.get(' Destination Port', 0))

        
        # 3. Apply SOC Guardrails
        if prediction == 1:
            # Rule 1: The "First Packet Bias" Filter
            # Ignore attacks if the AI is less than 85% confident
            if confidence < 85.0:
                return "✅ NORMAL (Override - Low Conf)", confidence
                
            # Rule 2: The "Human / Legitimate App" Filter
            # Ephemeral ports (>= 49152) and Web ports (80, 443) are standard for background traffic.
            # If the flow duration is longer than a microsecond stealth-burst (>50,000us), it's legitimate.
            if (port in [80.0, 443.0] or port >= 49152) and duration > 50000:
                return "✅ NORMAL (Override - Duration)", confidence
            
            # For all other ports, or ultra-fast stealth scans, trust the AI
            return "🚨 ATTACK", confidence
            
        return "✅ NORMAL", confidence

# Quick test to ensure it loads without crashing if you run this file directly
if __name__ == "__main__":
    brain = IDSModel()