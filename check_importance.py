import joblib
from catboost import CatBoostClassifier
import pandas as pd

def main():
    print("[*] Loading CatBoost HYBRID Model (V2) and Feature Names...")
    
    # Load the new V2 model
    cat_model = CatBoostClassifier()
    cat_model.load_model('models/cat_hybrid_v2.cbm')
    
    # Load the new V2 feature names
    features = joblib.load('models/feature_names_v2.joblib')

    # Extract the mathematical importance
    importances = cat_model.get_feature_importance()

    # Format into a clean table and sort from highest to lowest
    importance_df = pd.DataFrame({'Feature': features, 'Importance (%)': importances})
    importance_df = importance_df.sort_values(by='Importance (%)', ascending=False)

    print("\n[+] Hybrid V2 Feature Importance Ranking:")
    print("========================================")
    print(importance_df.to_string(index=False, float_format=lambda x: f"{x:.2f}%"))

if __name__ == "__main__":
    main()