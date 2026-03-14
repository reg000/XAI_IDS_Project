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
        
        # Extract time and context features
        duration = float(feature_dict.get(' Flow Duration', 0))
        port = float(feature_dict.get(' Destination Port', 0))
        
        # Extract volume features safely (handling potential leading spaces from CIC-IDS format)
        fwd_pkts = float(feature_dict.get(' Total Fwd Packets', feature_dict.get('Total Fwd Packets', 0)))
        bwd_pkts = float(feature_dict.get(' Total Backward Packets', feature_dict.get('Total Backward Packets', 0)))
        total_pkts = fwd_pkts + bwd_pkts

        # 3. Apply SOC Guardrails
        if prediction == 1:
            # Rule 1: The "First Packet Bias" Filter
            if confidence < 85.0:
                return "✅ NORMAL (Override - Low Conf)", confidence
                
            # Define Enterprise & Web Ports (Added 88 Kerberos and 135 RPC)
            enterprise_ports = [21.0, 22.0, 80.0, 88.0, 135.0, 139.0, 389.0, 443.0, 445.0, 465.0, 3268.0]
            
            # Rule 2: The "Volume & Time" Filter
            if port in enterprise_ports or port >= 49152:
                # Lowered from > 4 to >= 3 to allow standard TCP 3-way handshakes to pass
                if total_pkts >= 3:
                    return "✅ NORMAL (Override - Volume)", confidence
                # If it lasts longer than 50ms, it's a human/normal app
                elif duration > 50000:
                    return "✅ NORMAL (Override - Duration)", confidence
            
            # For all other ports, or ultra-fast, low-volume stealth scans, trust the AI
            return "🚨 ATTACK", confidence
            
        return "✅ NORMAL", confidence

if __name__ == "__main__":
    brain = IDSModel()