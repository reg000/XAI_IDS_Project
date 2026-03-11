import os
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- Configuration ---
TEST_PATH = 'data/UNSW_NB15_testing-set.csv'
MODEL_PATH = 'models/cat_hybrid_v3_1.cbm'
FEATURE_NAMES_PATH = 'models/feature_names_v3_1.joblib'

def map_unsw_to_cic(filepath, features):
    """Replicates the mapping logic from train_hybrid.py for consistency."""
    print(f"[*] Loading and mapping UNSW test data from {filepath}...")
    df = pd.read_csv(filepath)
    new_df = pd.DataFrame()
    
    # 1. Port Reconstruction
    if 'service' in df.columns:
        port_map = {
            'http': 80.0, 'dns': 53.0, 'smtp': 25.0, 'ftp': 21.0, 
            'ftp-data': 20.0, 'pop3': 110.0, 'ssh': 22.0, 'ssl': 443.0, 
            'snmp': 161.0, 'dhcp': 67.0, 'radius': 1812.0, 'irc': 194.0
        }
        new_df[' Destination Port'] = df['service'].map(port_map).fillna(0.0)
    else:
        new_df[' Destination Port'] = 0.0

    # 2. Basic Math Mapping
    new_df[' Flow Duration'] = df['dur'] * 1e6
    new_df[' Total Fwd Packets'] = df['spkts']
    new_df[' Total Bwd Packets'] = df['dpkts']
    new_df['Total Length of Fwd Packets'] = df['sbytes']
    new_df[' Total Length of Bwd Packets'] = df['dbytes']
    new_df[' Fwd Packet Length Max'] = 0.0 # Placeholder as it's missing in raw UNSW CSV
    new_df[' Bwd Packet Length Max'] = 0.0
    new_df[' Flow IAT Mean'] = pd.to_numeric(df['sinpkt'], errors='coerce').fillna(0) * 1000
    
    safe_dur = df['dur'].apply(lambda x: max(x, 0.000001))
    new_df[' Flow Packets/s'] = (df['spkts'] + df['dpkts']) / safe_dur

    # 3. Flag Mapping
    state_col = df['state'].astype(str).str.upper()
    new_df[' SYN Flag Count'] = state_col.apply(lambda x: 1.0 if x in ['CON', 'SYN'] else 0.0)
    new_df[' RST Flag Count'] = state_col.apply(lambda x: 1.0 if x in ['INT', 'RST', 'REQ'] else 0.0)
    new_df[' ACK Flag Count'] = state_col.apply(lambda x: 1.0 if x in ['CON', 'FIN'] else 0.0)
    new_df[' FIN Flag Count'] = state_col.apply(lambda x: 1.0 if x in ['FIN'] else 0.0)

    # 4. Cleanup
    new_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    new_df.fillna(0, inplace=True)
    
    return new_df[features], df['label']

def main():
    # 1. Load Model and Features
    print("[*] Loading Hybrid V2 Model...")
    if not os.path.exists(MODEL_PATH):
        print(f"[-] Error: Model file {MODEL_PATH} not found!")
        return
        
    cat_model = CatBoostClassifier()
    cat_model.load_model(MODEL_PATH)
    features = joblib.load(FEATURE_NAMES_PATH)

    # 2. Prepare Test Data
    X_test, y_test = map_unsw_to_cic(TEST_PATH, features)

    # 3. Run Predictions
    print("[*] Running inference on UNSW test set...")
    preds = cat_model.predict(X_test)

    # 4. Detailed Evaluation
    print("\n" + "="*45)
    print("🔬 FINAL HYBRID VALIDATION RESULTS")
    print("="*45)
    print(f"Testing Accuracy: {accuracy_score(y_test, preds) * 100:.2f}%")
    print("\nClassification Report (UNSW Unseen Data):")
    print(classification_report(y_test, preds, target_names=['Normal', 'Attack']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("="*45)

if __name__ == "__main__":
    main()