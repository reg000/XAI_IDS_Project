import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# --- Configuration ---
DATA_PATH = 'data/PortScan.csv'
MODEL_DIR = 'models/'

# The "Lite 10" features we selected for live Scapy sniffing compatibility
FEATURES = [
    ' Destination Port', ' Flow Duration', ' Total Fwd Packets', 
    ' Total Bwd Packets', 'Total Length of Fwd Packets', ' Total Length of Bwd Packets', 
    ' Fwd Packet Length Max', ' Bwd Packet Length Max', ' Flow IAT Mean', ' Flow Packets/s'
]
TARGET = ' Label'

def load_and_clean_data(filepath):
    """Loads CSV and handles AI-crashing errors like Infinity/NaN values."""
    print(f"[*] Loading dataset from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. Please download it.")
        
    df = pd.read_csv(filepath)
    
    # Clean data: Replace infinite values and drop missing rows
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Simplify labels: 'BENIGN' stays 0, anything else (PortScan) becomes 1
    df['Binary_Label'] = df[TARGET].apply(lambda x: 0 if x == 'BENIGN' else 1)
    
    return df

def main():
    # 1. Prepare Data
    df = load_and_clean_data(DATA_PATH)
    
    # Ensure all features exist in the dataset
    actual_features = [f for f in FEATURES if f in df.columns]
    
    X = df[actual_features]
    y = df['Binary_Label']

    # 2. Split Data (80% Training, 20% Testing)
    print("[*] Splitting data into Train and Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train Baseline: Random Forest
    print("\n[1/2] Training Baseline Model (Random Forest)...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    
    # 4. Train Evolution: CatBoost
    print("\n[2/2] Training Evolution Model (CatBoost)...")
    cat_model = CatBoostClassifier(iterations=200, learning_rate=0.1, depth=6, verbose=0, random_seed=42)
    cat_model.fit(X_train, y_train)
    cat_preds = cat_model.predict(X_test)

    # 5. Evaluate and Compare
    print("\n" + "="*40)
    print("📊 MODEL COMPARISON RESULTS")
    print("="*40)
    print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_preds) * 100:.2f}%")
    print(f"CatBoost Accuracy:      {accuracy_score(y_test, cat_preds) * 100:.2f}%")
    print("\nCatBoost Classification Report:")
    print(classification_report(y_test, cat_preds, target_names=['Normal', 'Attack']))

    # 6. Save Models for Live Sniffing
    print("\n[*] Saving models to disk...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(rf_model, os.path.join(MODEL_DIR, 'rf_baseline.joblib'))
    cat_model.save_model(os.path.join(MODEL_DIR, 'cat_evolution.cbm')) # CatBoost native format
    
    # Save the feature list so our sniffer knows exactly what to look for
    joblib.dump(actual_features, os.path.join(MODEL_DIR, 'feature_names.joblib'))
    print("[+] Training complete! Models ready for XAI and Live Sniffing.")

if __name__ == "__main__":
    main()