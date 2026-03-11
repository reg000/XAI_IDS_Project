import joblib
from catboost import CatBoostClassifier
import pandas as pd

def main():
    print("[*] Loading CatBoost STEALTH-AWARE Model (V3)...")
    
    # Load the V3 model
    cat_model = CatBoostClassifier()
    cat_model.load_model('models/cat_hybrid_v3_1.cbm')
    
    # Load the V3 feature names
    features = joblib.load('models/feature_names_v3_1.joblib')

    # Extract the mathematical importance
    importances = cat_model.get_feature_importance()

    # Format into a clean table
    importance_df = pd.DataFrame({'Feature': features, 'Importance (%)': importances})
    importance_df = importance_df.sort_values(by='Importance (%)', ascending=False)

    print("\n[+] V3 Feature Importance Ranking:")
    print("========================================")
    print(importance_df.to_string(index=False, float_format=lambda x: f"{x:.2f}%"))

if __name__ == "__main__":
    main()