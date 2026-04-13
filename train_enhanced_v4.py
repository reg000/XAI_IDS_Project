"""
XAI-IDS Enhanced Training Script (V4)
Fixes port scan detection by using discriminative flow-rate features
instead of extracted TCP flags.

Key improvements:
- Flow Bytes/s (vs /s metric)
- Flow Packets/s with variance
- IAT Max (Inter-Arrival Time Max)
- Packet Length statistics
- Proper class weighting for imbalanced attacks
"""

import os
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt

# --- Configuration ---
CIC_PATH = '/mnt/user-data/uploads/PortScan.csv'
UNSW_PATH = '/mnt/user-data/uploads/UNSW_NB15_training-set.csv'
UNSW_TEST_PATH = '/mnt/user-data/uploads/UNSW_NB15_testing-set.csv'
STEALTH_PATH = '/mnt/user-data/uploads/stealth_augment.csv'
MODEL_DIR = '/home/claude/models/'
OUTPUT_DIR = '/mnt/user-data/outputs/'

# --- ENHANCED 16-FEATURE SET (Better Port Scan Discrimination) ---
# Removed the synthetic TCP flags; using actual flow metrics instead
FEATURES_ENHANCED = [
    ' Destination Port',           # 1. Port number (port scans hit many ports)
    ' Flow Duration',              # 2. Duration in microseconds (scans are SHORT)
    ' Total Fwd Packets',          # 3. Forward packets
    ' Total Bwd Packets',          # 4. Backward packets  
    'Total Length of Fwd Packets', # 5. Forward bytes
    ' Total Length of Bwd Packets',# 6. Backward bytes
    ' Fwd Packet Length Max',      # 7. Max forward packet
    ' Bwd Packet Length Max',      # 8. Max backward packet
    ' Flow IAT Mean',              # 9. Mean inter-arrival time
    ' Flow IAT Max',               # 10. MAX inter-arrival time (DISCRIMINATIVE!)
    ' Flow Packets/s',             # 11. Packet rate (scans are FAST)
    ' Flow Bytes/s',               # 12. Byte rate (scans are FAST) 
    ' Fwd Packets/s',              # 13. Forward packet rate
    ' Bwd Packets/s',              # 14. Backward packet rate
    ' Packet Length Variance',     # 15. Variance (scans have LOW variance)
    'Init_Win_bytes_forward'       # 16. Initial window (TCP behavior)
]

TARGET = ' Label'

def load_cic_data(filepath):
    """Loads CIC-IDS2017 PortScan dataset with all features."""
    print(f"[*] Loading CIC-IDS2017 dataset from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing {filepath}")
        
    df = pd.read_csv(filepath)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Binary label: BENIGN=0, Everything else (PortScan, etc)=1
    df['Binary_Label'] = df[TARGET].apply(lambda x: 0 if x == 'BENIGN' else 1)
    
    # Filter to only available features
    available_features = [f for f in FEATURES_ENHANCED if f in df.columns]
    return df[available_features + ['Binary_Label']], available_features

def map_unsw_to_cic(filepath, available_features):
    """
    Maps UNSW-NB15 to CIC feature space.
    Many features are missing, so we'll fill with 0 where needed.
    """
    print(f"[*] Loading UNSW-NB15 from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing {filepath}")
        
    df = pd.read_csv(filepath)
    new_df = pd.DataFrame()
    
    # Map each feature carefully - use only what's available in both datasets
    for feat in available_features:
        if feat == ' Destination Port':
            new_df[feat] = 0  # Not in UNSW
        elif feat == ' Flow Duration':
            new_df[feat] = df['dur'] * 1e6
        elif feat == ' Total Fwd Packets':
            new_df[feat] = df['spkts']
        elif feat == ' Total Bwd Packets':
            new_df[feat] = pd.to_numeric(df['dpkts'], errors='coerce').fillna(0)
        elif feat == 'Total Length of Fwd Packets':
            new_df[feat] = df['sbytes']
        elif feat == ' Total Length of Bwd Packets':
            new_df[feat] = df['dbytes']
        elif feat == ' Fwd Packet Length Max':
            new_df[feat] = (df['sbytes'] / (df['spkts'] + 0.001)).clip(upper=1500)
        elif feat == ' Bwd Packet Length Max':
            new_df[feat] = (df['dbytes'] / (df['dpkts'] + 0.001)).clip(upper=1500)
        elif feat == ' Flow IAT Mean':
            new_df[feat] = pd.to_numeric(df['sinpkt'], errors='coerce').fillna(0) * 1000
        elif feat == ' Flow IAT Max':
            new_df[feat] = pd.to_numeric(df['sinpkt'], errors='coerce').fillna(0) * 1000 * 5
        elif feat == ' Flow Packets/s':
            safe_dur = df['dur'].apply(lambda x: max(x, 0.000001))
            new_df[feat] = (df['spkts'] + pd.to_numeric(df['dpkts'], errors='coerce').fillna(0)) / safe_dur
        elif feat == ' Bwd Packets/s':
            safe_dur = df['dur'].apply(lambda x: max(x, 0.000001))
            new_df[feat] = pd.to_numeric(df['dpkts'], errors='coerce').fillna(0) / safe_dur
        elif feat == ' Packet Length Variance':
            new_df[feat] = np.random.rand(len(df)) * 500
        elif feat == 'Init_Win_bytes_forward':
            new_df[feat] = 0
        else:
            new_df[feat] = 0
    
    
    # Label: UNSW uses 1 for attack, 0 for normal
    new_df['Binary_Label'] = df['label']
    
    # Cleanup
    new_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    new_df.fillna(0, inplace=True)
    
    return new_df[available_features + ['Binary_Label']]

def main():
    try:
        # Load CIC data first to get available features
        df_cic, available_features = load_cic_data(CIC_PATH)
        print(f"[+] CIC loaded. Using {len(available_features)} features:")
        for f in available_features:
            print(f"    - {f}")
        
        # Load UNSW
        df_unsw = map_unsw_to_cic(UNSW_PATH, available_features)
        
    except Exception as e:
        print(f"[-] Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return

    # Load stealth augmentation (if available)
    try:
        df_stealth = pd.read_csv(STEALTH_PATH)
        # Map stealth data to same feature space
        df_stealth_mapped = pd.DataFrame()
        for feat in available_features:
            if feat in df_stealth.columns:
                df_stealth_mapped[feat] = df_stealth[feat]
            else:
                df_stealth_mapped[feat] = 0
        df_stealth_mapped['Binary_Label'] = 1  # All attacks
        df_stealth_boosted = pd.concat([df_stealth_mapped] * 15, ignore_index=True)  # Boost to 690 rows
        print(f"[+] Stealth augmentation loaded and boosted to {len(df_stealth_boosted)} rows")
    except Exception as e:
        print(f"[!] Stealth augmentation not found: {e}")
        df_stealth_boosted = pd.DataFrame()

    # Combine all datasets
    print("\n[*] Combining datasets...")
    if len(df_stealth_boosted) > 0:
        df_combined = pd.concat([df_cic, df_unsw, df_stealth_boosted], ignore_index=True)
    else:
        df_combined = pd.concat([df_cic, df_unsw], ignore_index=True)
    
    # Check class distribution
    print(f"\n[*] Class distribution BEFORE balancing:")
    print(df_combined['Binary_Label'].value_counts())
    
    X = df_combined[available_features]
    y = df_combined['Binary_Label']
    
    # Split data
    print("\n[*] Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train enhanced model
    print(f"\n[+] Training CatBoost Enhanced Model (V4)...")
    print(f"    Features: {len(available_features)}")
    print(f"    Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Calculate class weights to handle imbalance
    n_benign = (y_train == 0).sum()
    n_attack = (y_train == 1).sum()
    class_weight = {0: n_attack / n_benign, 1: 1.0}  # Boost minority class
    
    cat_model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=8,
        class_weights=class_weight,
        verbose=50,
        random_seed=42,
        border_count=128
    )
    cat_model.fit(X_train, y_train, eval_set=(X_test, y_test))
    
    # Predict and evaluate
    print("\n" + "="*60)
    print("📊 V4 ENHANCED MODEL RESULTS")
    print("="*60)
    
    y_pred = cat_model.predict(X_test)
    y_pred_proba = cat_model.predict_proba(X_test)[:, 1]
    
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"Recall (True Positive Rate): {recall_score(y_test, y_pred)*100:.2f}%")
    print(f"Precision: {precision_score(y_test, y_pred)*100:.2f}%")
    print(f"F1-Score: {f1_score(y_test, y_pred)*100:.2f}%")
    
    print("\n" + "-"*60)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['BENIGN', 'PortScan/Attack']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"TN: {cm[0,0]} | FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]} | TP: {cm[1,1]}")
    
    # Feature importance
    print("\n" + "-"*60)
    print("Top 10 Feature Importance:")
    importances = cat_model.feature_importances_
    for feat, imp in sorted(zip(available_features, importances), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {feat:40s} {imp:6.2f}%")
    
    # Save models
    print("\n[*] Saving V4 models...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    cat_model.save_model(os.path.join(MODEL_DIR, 'cat_enhanced_v4.cbm'))
    joblib.dump(available_features, os.path.join(MODEL_DIR, 'feature_names_v4.joblib'))
    
    # Copy to outputs for download
    import shutil
    shutil.copy(os.path.join(MODEL_DIR, 'cat_enhanced_v4.cbm'), 
                os.path.join(OUTPUT_DIR, 'cat_enhanced_v4.cbm'))
    shutil.copy(os.path.join(MODEL_DIR, 'feature_names_v4.joblib'),
                os.path.join(OUTPUT_DIR, 'feature_names_v4.joblib'))
    
    print(f"[+] Model saved to {MODEL_DIR}")
    print(f"[+] Files copied to {OUTPUT_DIR}")
    print("\n[+] Training Complete! ✅")

if __name__ == "__main__":
    main()
