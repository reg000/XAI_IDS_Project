"""
XAI-IDS Enhanced Training Script (V4) – Multi-Dataset
========================================================
Combines CIC-IDS2017 (PortScan) with UNSW-NB15 data locally.
Includes dynamic column mapping to prevent feature dropping.
"""

import os
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, recall_score,
                             precision_score, f1_score)

# ── paths ──────────────────────────────────────────────────────────────────
CIC_PATH = 'data/PortScan.csv'
UNSW_PATHS = [
    'data/UNSW_NB15_training-set.csv',
    'data/UNSW_NB15_testing-set.csv'
]
MODEL_DIR = 'models'

# ── canonical 16-feature set ───────────────────────────────────────────────
FEATURES_ENHANCED = [
    ' Destination Port',
    ' Flow Duration',
    ' Total Fwd Packets',
    ' Total Bwd Packets',
    'Total Length of Fwd Packets',
    ' Total Length of Bwd Packets',
    ' Fwd Packet Length Max',
    ' Bwd Packet Length Max',
    ' Flow IAT Mean',
    ' Flow IAT Max',
    ' Flow Packets/s',
    ' Flow Bytes/s',
    ' Fwd Packets/s',
    ' Bwd Packets/s',
    ' Packet Length Variance',
    'Init_Win_bytes_forward',
]

TARGET = ' Label'

# ── CIC column name normalisation map ─────────────────────────────────────
CIC_COLUMN_MAP = {
    ' Total Backward Packets': ' Total Bwd Packets',
    'Bwd Packet Length Max': ' Bwd Packet Length Max',
    'Flow Bytes/s': ' Flow Bytes/s',
    'Fwd Packets/s': ' Fwd Packets/s'
}

def normalise_cic_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename any known CIC variant column names to their canonical form."""
    rename = {old: new for old, new in CIC_COLUMN_MAP.items() if old in df.columns}
    if rename:
        print(f"  [!] Normalised {len(rename)} column name(s):")
        for old, new in rename.items():
            print(f"      {repr(old)}  ->  {repr(new)}")
    return df.rename(columns=rename)


def load_cic_data(filepath):
    print(f"[*] Loading CIC-IDS2017 dataset from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing {filepath}")

    df = pd.read_csv(filepath)
    df = normalise_cic_columns(df)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Binary label
    df['Binary_Label'] = df[TARGET].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)

    available_features = [f for f in FEATURES_ENHANCED if f in df.columns]
    print(f"[+] CIC loaded. Using {len(available_features)}/{len(FEATURES_ENHANCED)} features.")

    return df[available_features + ['Binary_Label']], available_features


def map_unsw_to_cic(filepath, available_features):
    """Mathematically maps UNSW-NB15 flow features into the CIC-IDS2017 format."""
    print(f"[*] Loading UNSW-NB15 from {filepath}...")
    
    df = pd.read_csv(filepath)
    new_df = pd.DataFrame()

    for feat in available_features:
        if feat == ' Destination Port':
            new_df[feat] = 0
        elif feat == ' Flow Duration':
            new_df[feat] = pd.to_numeric(df['dur'], errors='coerce').fillna(0) * 1e6
        elif feat == ' Total Fwd Packets':
            new_df[feat] = pd.to_numeric(df['spkts'], errors='coerce').fillna(0)
        elif feat == ' Total Bwd Packets':
            new_df[feat] = pd.to_numeric(df['dpkts'], errors='coerce').fillna(0)
        elif feat == 'Total Length of Fwd Packets':
            new_df[feat] = pd.to_numeric(df['sbytes'], errors='coerce').fillna(0)
        elif feat == ' Total Length of Bwd Packets':
            new_df[feat] = pd.to_numeric(df['dbytes'], errors='coerce').fillna(0)
        elif feat == ' Fwd Packet Length Max':
            spkts = pd.to_numeric(df['spkts'], errors='coerce').replace(0, np.nan)
            new_df[feat] = (pd.to_numeric(df['sbytes'], errors='coerce') / spkts).clip(upper=1500).fillna(0)
        elif feat == ' Bwd Packet Length Max':
            dpkts = pd.to_numeric(df['dpkts'], errors='coerce').replace(0, np.nan)
            new_df[feat] = (pd.to_numeric(df['dbytes'], errors='coerce') / dpkts).clip(upper=1500).fillna(0)
        elif feat == ' Flow IAT Mean':
            new_df[feat] = pd.to_numeric(df['sinpkt'], errors='coerce').fillna(0) * 1000
        elif feat == ' Flow IAT Max':
            new_df[feat] = pd.to_numeric(df['sinpkt'], errors='coerce').fillna(0) * 1000 * 5
        elif feat == ' Flow Packets/s':
            dur = pd.to_numeric(df['dur'], errors='coerce').fillna(0)
            spkts = pd.to_numeric(df['spkts'], errors='coerce').fillna(0)
            dpkts = pd.to_numeric(df['dpkts'], errors='coerce').fillna(0)
            rate = (spkts + dpkts) / dur.where(dur > 0)
            new_df[feat] = rate.fillna(0)
        elif feat == ' Flow Bytes/s':
            dur = pd.to_numeric(df['dur'], errors='coerce').fillna(0)
            sbytes = pd.to_numeric(df['sbytes'], errors='coerce').fillna(0)
            dbytes = pd.to_numeric(df['dbytes'], errors='coerce').fillna(0)
            rate = (sbytes + dbytes) / dur.where(dur > 0)
            new_df[feat] = rate.fillna(0)
        elif feat == ' Fwd Packets/s':
            dur = pd.to_numeric(df['dur'], errors='coerce').fillna(0)
            spkts = pd.to_numeric(df['spkts'], errors='coerce').fillna(0)
            rate = spkts / dur.where(dur > 0)
            new_df[feat] = rate.fillna(0)
        elif feat == ' Bwd Packets/s':
            dur = pd.to_numeric(df['dur'], errors='coerce').fillna(0)
            dpkts = pd.to_numeric(df['dpkts'], errors='coerce').fillna(0)
            rate = dpkts / dur.where(dur > 0)
            new_df[feat] = rate.fillna(0)
        elif feat == ' Packet Length Variance':
            new_df[feat] = 0.0
        elif feat == 'Init_Win_bytes_forward':
            new_df[feat] = 0
        else:
            new_df[feat] = 0

    new_df['Binary_Label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
    new_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    new_df.fillna(0, inplace=True)

    print(f"  [+] Successfully mapped {len(new_df)} UNSW flows.")
    return new_df[available_features + ['Binary_Label']]


def main():
    frames = []

    # 1. Load CIC Data
    try:
        df_cic, available_features = load_cic_data(CIC_PATH)
        frames.append(df_cic)
    except Exception as e:
        print(f"[-] Failed to load CIC data: {e}")
        return

    # 2. Load UNSW Data (if available)
    for unsw_path in UNSW_PATHS:
        if os.path.exists(unsw_path):
            try:
                df_unsw = map_unsw_to_cic(unsw_path, available_features)
                frames.append(df_unsw)
            except Exception as e:
                print(f"[-] Failed to process {unsw_path}: {e}")
        else:
            print(f"[*] Skipping {unsw_path} (File not found)")

    # 3. Combine Datasets
    df_combined = pd.concat(frames, ignore_index=True)

    print(f"\n[*] Final Class distribution:")
    print(df_combined['Binary_Label'].value_counts())

    X = df_combined[available_features]
    y = df_combined['Binary_Label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── train ──────────────────────────────────────────────────────────────
    n_benign = int((y_train == 0).sum())
    n_attack = int((y_train == 1).sum())
    
    # Safely handle weights if dataset is heavily skewed
    if n_attack == 0 or n_benign == 0:
        print("[-] Error: Dataset is missing either Benign or Attack labels!")
        return
        
    class_weight = {0: n_attack / n_benign, 1: 1.0}

    print(f"\n[+] Training CatBoost V4 on {len(X_train)} samples ({len(available_features)} features)...")
    cat_model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=8,
        class_weights=class_weight,
        verbose=50,
        random_seed=42,
        border_count=128,
    )
    cat_model.fit(X_train, y_train, eval_set=(X_test, y_test))

    # ── evaluate ───────────────────────────────────────────────────────────
    y_pred = cat_model.predict(X_test)

    print("\n" + "=" * 60)
    print("V4 RESULTS")
    print("=" * 60)
    print(f"Accuracy  : {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"Recall    : {recall_score(y_test, y_pred)*100:.2f}%")
    print(f"Precision : {precision_score(y_test, y_pred)*100:.2f}%")

    # ── save ───────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_out    = os.path.join(MODEL_DIR, 'cat_enhanced_v4.cbm')
    features_out = os.path.join(MODEL_DIR, 'feature_names_v4.joblib')

    cat_model.save_model(model_out)
    joblib.dump(available_features, features_out)

    print(f"\n[+] Saved: {model_out}")
    print(f"[+] Saved: {features_out}")
    print("[+] Training complete. You are ready to test!")

if __name__ == "__main__":
    main()