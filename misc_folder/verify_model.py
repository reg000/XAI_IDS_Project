from sklearn.model_selection import cross_val_score
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

# 1. Load data and features
features = joblib.load('models/feature_names.joblib')
df = pd.read_csv('data/PortScan.csv').replace([np.inf, -np.inf], np.nan).dropna()
X = df[features]
y = df[' Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

print("[*] Starting 5-Fold Cross-Validation (This will take a minute)...\n")

# --- RANDOM FOREST CV ---
print("[1/2] Evaluating Random Forest...")
# n_jobs=-1 uses all your CPU cores to make it faster
rf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rf_scores = cross_val_score(rf_model, X, y, cv=5)

print(f"-> RF Scores: {rf_scores}")
print(f"-> RF Mean Accuracy: {rf_scores.mean() * 100:.2f}%")
print(f"-> RF Standard Deviation: {rf_scores.std():.5f}\n")

# --- CATBOOST CV ---
print("[2/2] Evaluating CatBoost...")
# verbose=0 keeps the output clean so it doesn't print 500 lines of training logs
cat_model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, verbose=0, random_seed=42)
cat_scores = cross_val_score(cat_model, X, y, cv=5)

print(f"-> CatBoost Scores: {cat_scores}")
print(f"-> CatBoost Mean Accuracy: {cat_scores.mean() * 100:.2f}%")
print(f"-> CatBoost Standard Deviation: {cat_scores.std():.5f}")

print("\n[+] Cross-Validation Complete!")