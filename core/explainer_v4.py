"""
Enhanced XAI Engine (V4) - SHAP-based Forensic Explainer
Provides interpretable explanations for port scan detections.
"""

import os
import json
from datetime import datetime
import joblib
import pandas as pd
import numpy as np

try:
    import shap
except ImportError:
    print("[!] Warning: shap not installed. Explainer will provide feature values only.")
    shap = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, 'logs')

class XAIEngineV4:
    """Explainable AI engine using SHAP for CatBoost model interpretability."""
    
    def __init__(self, model, feature_names, log_filename="forensics_log.jsonl"):
        """Initialize SHAP explainer for the enhanced V4 model."""
        print("[*] Initializing XAI Engine (V4) with SHAP...")
        
        self.model = model
        self.feature_names = feature_names
        
        # Initialize SHAP explainer
        if shap is not None:
            self.explainer = shap.TreeExplainer(model)
            print("[+] SHAP TreeExplainer initialized")
        else:
            self.explainer = None
            print("[!] SHAP unavailable - using feature values for explanation")
        
        # Setup logging
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        
        self.log_path = os.path.join(LOG_DIR, log_filename)
        if os.path.exists(self.log_path):
            print(f"[+] Appending to existing log: {log_filename}")
        else:
            print(f"[+] Creating new forensic log: {log_filename}")

    def explain_prediction(self, df, confidence, port, src_ip, dst_ip, pkt_time):
        """
        Generate SHAP-based explanation for a detected attack.
        """
        try:
            if self.explainer is not None:
                # Calculate SHAP values
                shap_values = self.explainer.shap_values(df)
                
                # 1. FIX: Grab the 1D array for the single packet (index 0)
                if isinstance(shap_values, list):
                    shap_vals = shap_values[1][0]  # Attack class, first row
                else:
                    shap_vals = shap_values[0]     # First row
                
                # 2. FIX: Safely extract the base value BEFORE converting to float
                expected_val = self.explainer.expected_value
                if isinstance(expected_val, (list, tuple, np.ndarray)):
                    base_value = float(expected_val[1] if len(expected_val) > 1 else expected_val[0])
                else:
                    base_value = float(expected_val)
            else:
                # Fallback: use zeros if SHAP unavailable
                shap_vals = np.zeros(len(self.feature_names))
                base_value = 0.5
            
            # Build explanation
            impacts = []
            for i, feature_name in enumerate(self.feature_names):
                
                # ── FORMAT FIX: Round raw value and SHAP impact to 3 decimals ──
                value = round(float(df.iloc[0, i]), 3)
                impact = round(float(shap_vals[i]), 3)
                
                impacts.append({
                    "feature": feature_name,
                    "value": value,
                    "shap_impact": impact
                })
            
            # Sort all 16 features by magnitude (largest impact first)
            impacts.sort(key=lambda x: abs(x["shap_impact"]), reverse=True)
            top_3 = impacts[:3] 
            
            # FORMAT FIX: Convert the list into a Dictionary specifically for the UI charts
            full_profile = {
                item["feature"]: {
                    "value": item["value"], 
                    "shap_impact": item["shap_impact"]
                }
                for item in impacts
            }
            
            return {
                "base_probability": round(base_value, 4),
                "top_3_drivers": top_3,
                "full_feature_profile": full_profile  # <--- Matches your HTML exactly!
            }
            
        except Exception as e:
            print(f"[-] Explanation generation failed: {e}")
            return {
                "error": str(e),
                "features": [{"feature": f, "value": float(df.iloc[0, i])} 
                           for i, f in enumerate(self.feature_names)]
            }

    def log_detection(self, df, confidence, port, src_ip, dst_ip, pkt_time):
        """Write forensic entry to JSONL audit trail."""
        try:
            explanation = self.explain_prediction(df, confidence, port, src_ip, dst_ip, pkt_time)
            
            log_entry = {
                "timestamp": pkt_time,
                "source_ip": src_ip,
                "dest_ip": dst_ip,
                "dest_port": int(port),
                "confidence_score": round(float(confidence), 2),
                "attack_signature": {
                    "flow_duration_us": float(df.iloc[0][' Flow Duration']) if ' Flow Duration' in df.columns else 0,
                    "fwd_packets": float(df.iloc[0][' Total Fwd Packets']) if ' Total Fwd Packets' in df.columns else 0,
                    "bwd_packets": float(df.iloc[0][' Total Bwd Packets']) if ' Total Bwd Packets' in df.columns else 0,
                },
                "xai_explanation": explanation
            }
            
            # Append to log file
            with open(self.log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            
            return log_entry
            
        except Exception as e:
            print(f"[-] Forensic logging failed: {e}")
            return None
