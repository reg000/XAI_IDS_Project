import os
import joblib
import pandas as pd
from catboost import CatBoostClassifier
from core.explainer import XAIEngine

class IDSModel:
    def __init__(self, model_path='models/cat_hybrid_v3_1.cbm', features_path='models/feature_names_v3_1.joblib'):
        """Loads the pre-trained AI and the required feature list into memory."""
        print("[*] Waking up the AI Engine...")
        
        if not os.path.exists(model_path):
            model_path = '../' + model_path
            features_path = '../' + features_path
            
        try:
            self.model = CatBoostClassifier()
            self.model.load_model(model_path)
            self.features = joblib.load(features_path)
            print(f"[+] AI Engine Ready. Listening for {len(self.features)} specific features.")
            
            # Pass the loaded model and features to your dedicated explainer module
            self.xai = XAIEngine(self.model, self.features)
            
        except Exception as e:
            print(f"[-] Error loading model: {e}")

    def predict(self, feature_dict, src_ip="Unknown", dst_ip="Unknown", pkt_time="Unknown"):
        """Feeds packet to AI, applies Guardrails, and routes to XAI if malicious."""
        try:
            df = pd.DataFrame([feature_dict], columns=self.features)
        except Exception as e:
            return "ERROR", 0.0, None

        prediction = self.model.predict(df)[0]
        probabilities = self.model.predict_proba(df)[0]
        confidence = probabilities.max() * 100
        
        duration = float(feature_dict.get(' Flow Duration', 0))
        port = float(feature_dict.get(' Destination Port', 0))
        
        fwd_pkts = float(feature_dict.get(' Total Fwd Packets', feature_dict.get('Total Fwd Packets', 0)))
        bwd_pkts = float(feature_dict.get(' Total Backward Packets', feature_dict.get('Total Backward Packets', 0)))
        total_pkts = fwd_pkts + bwd_pkts

        # 3. Apply SOC Guardrails
        if prediction == 1:
            if confidence < 85.0:
                return "✅ NORMAL (Override - Low Conf)", confidence, None
                
            enterprise_ports = [21.0, 22.0, 80.0, 88.0, 135.0, 139.0, 389.0, 443.0, 445.0, 465.0, 3268.0]
            
            if port in enterprise_ports or port >= 49152:
                if total_pkts >= 3:
                    return "✅ NORMAL (Override - Volume)", confidence, None
                elif duration > 50000:
                    return "✅ NORMAL (Override - Duration)", confidence, None
            
            # The attack is verified. Call the explainer to log it!
            xai_data = self.xai.log_attack(df, confidence, port, src_ip, dst_ip, pkt_time)
            return "🚨 ATTACK", confidence, xai_data
            
        return "✅ NORMAL", confidence, None

if __name__ == "__main__":
    brain = IDSModel()