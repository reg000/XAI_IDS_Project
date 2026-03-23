import os
import json
from datetime import datetime
import shap

class XAIEngine:
    def __init__(self, model, feature_names):
        """Initializes the SHAP Explainer using the pre-loaded CatBoost model."""
        print("[*] Initializing SHAP TreeExplainer in dedicated module...")
        self.explainer = shap.TreeExplainer(model)
        self.features = feature_names
        self.log_dir = "logs"
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        print("[+] XAI Forensic Module Online.")

    def log_attack(self, df, confidence, port, src_ip, dst_ip, pkt_time):
        """Calculates SHAP values and writes the JSON Audit Trail."""
        try:
            # 1. Calculate SHAP values
            shap_vals = self.explainer.shap_values(df)[0]
            
            # Handle standard formatting for expected base value
            base_val = self.explainer.expected_value
            if isinstance(base_val, (list, tuple)):
                base_val = float(base_val[0])
            else:
                base_val = float(base_val)

            full_profile = {}
            impacts_list = []

            # 2. Map features to their mathematical impact
            for i, feature_name in enumerate(self.features):
                val = float(df.iloc[0, i])
                impact = float(shap_vals[i])
                
                impact_str = f"+{impact:.4f}" if impact > 0 else f"{impact:.4f}"
                
                impacts_list.append({
                    "feature": feature_name.strip(),
                    "value": val,
                    "shap_impact": impact_str,
                    "raw_impact": impact
                })
                
                full_profile[feature_name.strip()] = {"value": val, "shap_impact": impact_str}

            # 3. Extract Top 3 Drivers
            impacts_list.sort(key=lambda x: abs(x["raw_impact"]), reverse=True)
            top_3 = [{"feature": f["feature"], "value": f["value"], "shap_impact": f["shap_impact"]} for f in impacts_list[:3]]

            # 4. Build JSON Schema
            log_entry = {
                "timestamp": pkt_time,
                "source_ip": src_ip,
                "dest_ip": dst_ip,
                "dest_port": port,
                "confidence_score": round(float(confidence), 2),
                "xai_explanation": {
                    "base_model_probability": round(base_val, 4),
                    "top_3_drivers": top_3,
                    "full_feature_profile": full_profile
                }
            }

            # 5. Write to File Asynchronously
            with open(f"{self.log_dir}/forensics_log.jsonl", "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            return log_entry

        except Exception as e:
            print(f"[-] SHAP Forensics Failed: {e}")
            return None