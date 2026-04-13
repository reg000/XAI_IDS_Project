"""
Enhanced IDS Model (V4) - Production Inference Module
Uses discriminative flow-rate features instead of synthetic TCP flags.
"""

import os
import sys
import joblib
import pandas as pd
from catboost import CatBoostClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class IDSModelV4:
    def __init__(self, log_filename="forensics_log.jsonl", 
                 model_filename='cat_enhanced_v4.cbm', 
                 features_filename='feature_names_v4.joblib'):
        """Loads the enhanced V4 CatBoost model with discriminative features."""
        print("[*] Initializing Enhanced IDS Engine (V4)...")
        
        model_path = os.path.join(BASE_DIR, 'models', model_filename)
        features_path = os.path.join(BASE_DIR, 'models', features_filename)
            
        try:
            self.model = CatBoostClassifier()
            self.model.load_model(model_path)
            self.features = joblib.load(features_path)
            print(f"[+] Model Ready. Monitoring {len(self.features)} flow metrics.")
            print(f"    Key discriminators: Flow Duration, Packet Rates, Packet Lengths")
            
        except Exception as e:
            print(f"[-] Error loading model: {e}")
            sys.exit(1)

    def predict(self, feature_dict, src_ip="Unknown", dst_ip="Unknown", pkt_time="Unknown"):
        """
        Enhanced prediction with optimized thresholds for port scan detection.
        
        Args:
            feature_dict: Dictionary with 12 flow features
            src_ip, dst_ip, pkt_time: Metadata for logging
            
        Returns:
            (verdict: bool, confidence: float, metadata: dict)
        """
        try:
            # Build feature vector in correct order
            df = pd.DataFrame([feature_dict], columns=self.features)
        except Exception as e:
            print(f"[-] Feature extraction error: {e}")
            return False, 0.0, None 

        # Get prediction and probability
        prediction = self.model.predict(df)[0]
        probabilities = self.model.predict_proba(df)[0]
        confidence = probabilities.max() * 100
        
        # Extract metadata
        duration = float(feature_dict.get(' Flow Duration', 0))
        port = float(feature_dict.get(' Destination Port', 0))
        fwd_pkts = float(feature_dict.get(' Total Fwd Packets', 0))
        bwd_pkts = float(feature_dict.get(' Total Bwd Packets', 0))
        total_pkts = fwd_pkts + bwd_pkts

        # Port scan specific guardrails
        if prediction == 1:  # Model says attack
            # Require minimum confidence
            if confidence < 80.0:
                return False, confidence, None
            
            # Filter out benign high-port traffic
            if port >= 49152 and total_pkts < 2:
                return False, confidence, None
            
            # Attack verified - return alert
            return True, confidence, {
                'port': port,
                'duration_us': duration,
                'fwd_packets': fwd_pkts,
                'bwd_packets': bwd_pkts,
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'timestamp': pkt_time
            }
            
        return False, confidence, None

if __name__ == "__main__":
    print("[*] Enhanced IDS V4 module loaded successfully")
