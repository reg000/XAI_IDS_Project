import os
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# --- Configuration ---
CIC_PATH = 'data/PortScan.csv'
UNSW_PATH = 'data/UNSW_NB15_training-set.csv'
STEALTH_AUGMENT_PATH = 'data/stealth_augment.csv'  # <-- Added Stealth path
MODEL_DIR = 'models/'

# The exactly spaced "Stealth-Aware 14" features for Scapy compatibility
FEATURES = [
    ' Destination Port', ' Flow Duration', ' Total Fwd Packets', 
    ' Total Bwd Packets', 'Total Length of Fwd Packets', ' Total Length of Bwd Packets', 
    ' Fwd Packet Length Max', ' Bwd Packet Length Max', ' Flow IAT Mean', ' Flow Packets/s',
    ' SYN Flag Count', ' RST Flag Count', ' ACK Flag Count', ' FIN Flag Count'
]
TARGET = ' Label'

def load_cic_data(filepath):
    """Loads and cleans the original CIC-IDS2017 dataset."""
    print(f"[*] Loading CIC-IDS2017 dataset from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing {filepath}")
        
    df = pd.read_csv(filepath)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # 'BENIGN' is 0, 'PortScan' (or others) is 1
    df['Binary_Label'] = df[TARGET].apply(lambda x: 0 if x == 'BENIGN' else 1)
    
    # Filter to only the 14 features we need + Label
    actual_features = [f for f in FEATURES if f in df.columns]
    return df[actual_features + ['Binary_Label']]

def map_unsw_to_cic(filepath):
    """Loads UNSW-NB15 and maps its columns to the CIC-IDS2017 14-feature format."""
    print(f"[*] Loading and mapping UNSW-NB15 dataset from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing {filepath}. Please download the CSV.")
        
    df = pd.read_csv(filepath)
    new_df = pd.DataFrame()
    
    # 1. Port Reconstruction: Map text 'services' to numeric ports
    if 'dsport' in df.columns:
        new_df[' Destination Port'] = pd.to_numeric(df['dsport'], errors='coerce').fillna(0)
    elif 'service' in df.columns:
        port_map = {
            'http': 80.0, 'dns': 53.0, 'smtp': 25.0, 'ftp': 21.0, 
            'ftp-data': 20.0, 'pop3': 110.0, 'ssh': 22.0, 'ssl': 443.0, 
            'snmp': 161.0, 'dhcp': 67.0, 'radius': 1812.0, 'irc': 194.0
        }
        # Anything marked as '-' or unknown becomes 0.0 (Background noise port)
        new_df[' Destination Port'] = df['service'].map(port_map).fillna(0.0)
    else:
        new_df[' Destination Port'] = 0.0

    # 2. Mathematical Mapping
    new_df[' Flow Duration'] = df['dur'] * 1e6  # Convert Seconds to Microseconds
    new_df[' Total Fwd Packets'] = df['spkts']
    new_df[' Total Bwd Packets'] = df['dpkts']
    new_df['Total Length of Fwd Packets'] = df['sbytes']
    new_df[' Total Length of Bwd Packets'] = df['dbytes']
    
    # Missing direct mappings are safely zeroed out
    new_df[' Fwd Packet Length Max'] = 0.0 
    new_df[' Bwd Packet Length Max'] = 0.0
    new_df[' Flow IAT Mean'] = pd.to_numeric(df['sinpkt'], errors='coerce').fillna(0) * 1000 # ms to us
    
    # Calculate Packets/s safely
    safe_dur = df['dur'].apply(lambda x: max(x, 0.000001))
    new_df[' Flow Packets/s'] = (df['spkts'] + df['dpkts']) / safe_dur

    # 3. Flag Extraction (Derived from UNSW's 'state' column)
    state_col = df['state'].astype(str).str.upper()
    new_df[' SYN Flag Count'] = state_col.apply(lambda x: 1.0 if x in ['CON', 'SYN'] else 0.0)
    new_df[' RST Flag Count'] = state_col.apply(lambda x: 1.0 if x in ['INT', 'RST', 'REQ'] else 0.0)
    new_df[' ACK Flag Count'] = state_col.apply(lambda x: 1.0 if x in ['CON', 'FIN'] else 0.0)
    new_df[' FIN Flag Count'] = state_col.apply(lambda x: 1.0 if x in ['FIN'] else 0.0)

    # 4. Label Mapping
    new_df['Binary_Label'] = df['label']
    
    # Clean up any residual math errors
    new_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    new_df.fillna(0, inplace=True)
    
    return new_df[FEATURES + ['Binary_Label']]

def main():
    try:
        df_cic = load_cic_data(CIC_PATH)
        df_unsw = map_unsw_to_cic(UNSW_PATH)
    except Exception as e:
        print(f"[-] Error loading data: {e}")
        return

    # --- THE V3 STEALTH INJECTION PHASE ---
    print(f"\n[*] Injecting Targeted Stealth Vectors from {STEALTH_AUGMENT_PATH}...")
    try:
        df_stealth = pd.read_csv(STEALTH_AUGMENT_PATH)
        df_stealth['Binary_Label'] = 1 # Force attack label
        # Oversampling: multiply the 46 rows by 10 to increase their mathematical weight
        df_stealth_boosted = pd.concat([df_stealth] * 10, ignore_index=True)
    except Exception as e:
        print(f"[-] Error loading stealth data: {e}. Make sure the harvester succeeded.")
        return

    # 2. Combine Datasets
    print("[*] Merging all datasets into the Final V3 matrix...")
    df_combined = pd.concat([df_cic, df_unsw, df_stealth_boosted], ignore_index=True)
    
    X = df_combined[FEATURES]
    y = df_combined['Binary_Label']

    # 3. Split Data
    print(f"[*] Total V3 Samples: {len(df_combined)}")
    print("[*] Splitting data (80% Train / 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train Hybrid Model (V3)
    print("\n[+] Training CatBoost Stealth-Aware Model (V3)...")
    cat_model = CatBoostClassifier(iterations=200, learning_rate=0.1, depth=6, verbose=0, random_seed=42)
    cat_model.fit(X_train, y_train)
    cat_preds = cat_model.predict(X_test)

    # 5. Evaluate
    print("\n" + "="*40)
    print("📊 STEALTH-AWARE MODEL (V3) RESULTS")
    print("="*40)
    print(f"V3 Accuracy: {accuracy_score(y_test, cat_preds) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, cat_preds, target_names=['Normal', 'Attack']))

    # 6. Save as V3
    print("\n[*] Saving V3 models to disk safely...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save as v3! 
    cat_model.save_model(os.path.join(MODEL_DIR, 'cat_hybrid_v3_1.cbm')) 
    joblib.dump(FEATURES, os.path.join(MODEL_DIR, 'feature_names_v3_1.joblib'))
    
    print("[+] Training Complete! V3 Model is locked and loaded.")

if __name__ == "__main__":
    main()