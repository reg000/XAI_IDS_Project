"""
debug_v4.py – V4 Pipeline Diagnostic Tool
==========================================
Run this BEFORE test_pcap_v4.py to find exactly why 0 attacks are detected.

It checks every layer in order:
  1. Model feature list vs extractor output keys
  2. NaN / column ordering issues in the DataFrame fed to the model
  3. Raw model probabilities (no threshold applied)
  4. Feature value distributions for a sample of real flows
  5. Whether attack-like flows exist in the time window

Usage:
    python debug_v4.py
"""

import os, sys, json
from datetime import datetime
import numpy as np
import pandas as pd
from scapy.utils import PcapReader
from scapy.layers.inet import IP

# ── Adjust these paths to match your project layout ──────────────────────────
PCAP_PATH    = "data/PortScanPcapFridayWorkingHrs.pcapng"
MODEL_DIR    = "models"
MODEL_FILE   = "cat_enhanced_v4.cbm"
FEATURES_FILE= "feature_names_v4.joblib"
# ─────────────────────────────────────────────────────────────────────────────

# Time-sync offset used in test_pcap_v4.py
TIME_OFFSET   = 30600          # seconds (8.5 hours)
START_TS      = datetime(2017, 7, 7, 13, 45, 0).timestamp()

MAX_FLOWS     = 2000           # how many flows to sample for analysis
PRINT_EVERY   = 200            # show progress every N flows

SEP = "=" * 70


def load_model_and_features():
    import joblib
    from catboost import CatBoostClassifier

    model_path    = os.path.join(MODEL_DIR, MODEL_FILE)
    features_path = os.path.join(MODEL_DIR, FEATURES_FILE)

    print(f"\n{'─'*70}")
    print("STEP 1 – Model & Feature File")
    print(f"{'─'*70}")
    print(f"  Model    : {model_path}  exists={os.path.exists(model_path)}")
    print(f"  Features : {features_path}  exists={os.path.exists(features_path)}")

    if not os.path.exists(model_path) or not os.path.exists(features_path):
        print("\n  [FATAL] One or both files are missing.")
        print("  → The V4 model has NOT been trained yet.")
        print("  → Run train_enhanced_v4.py first, then re-run this script.")
        sys.exit(1)

    model = CatBoostClassifier()
    model.load_model(model_path)
    features = joblib.load(features_path)

    print(f"\n  Model loaded OK. Expects {len(features)} features:")
    for i, f in enumerate(features, 1):
        print(f"    [{i:>2}] {repr(f)}")   # repr() reveals hidden spaces!

    return model, features


def check_feature_alignment(model_features):
    """
    Compare model's expected feature names with what the extractor produces.
    Catches leading-space mismatches, missing features, or extras.
    """
    from core.extractor_v4 import FeatureExtractorV4
    from scapy.layers.inet import TCP

    print(f"\n{'─'*70}")
    print("STEP 2 – Feature Name Alignment (model vs extractor)")
    print(f"{'─'*70}")

    # Build a dummy packet to get one feature dict
    from scapy.all import Ether, IP as SIP, TCP as STCP
    dummy = Ether() / SIP(src="1.2.3.4", dst="5.6.7.8") / STCP(sport=12345, dport=80, flags="S")

    ext = FeatureExtractorV4()
    f = ext.extract_features(dummy)

    if f is None:
        print("  [FATAL] Extractor returned None for a basic TCP/SYN packet.")
        sys.exit(1)

    extractor_keys  = set(f.keys())
    model_keys      = set(model_features)
    missing_in_ext  = model_keys - extractor_keys    # model wants, extractor doesn't produce
    extra_in_ext    = extractor_keys - model_keys    # extractor produces, model doesn't want

    print(f"\n  Model features   : {len(model_keys)}")
    print(f"  Extractor keys   : {len(extractor_keys)}")

    if missing_in_ext:
        print(f"\n  [MISMATCH] Extractor is MISSING {len(missing_in_ext)} features the model expects:")
        for k in sorted(missing_in_ext):
            print(f"    MISSING  {repr(k)}")
    else:
        print("\n  All model-expected features are present in extractor output. ✓")

    if extra_in_ext:
        print(f"\n  [INFO] Extractor produces {len(extra_in_ext)} features the model ignores:")
        for k in sorted(extra_in_ext):
            print(f"    EXTRA    {repr(k)}")

    # Check column ordering when creating the DataFrame
    row = {k: f.get(k, float("nan")) for k in model_features}
    df_test = pd.DataFrame([row], columns=model_features)
    nan_cols = df_test.columns[df_test.isnull().any()].tolist()
    if nan_cols:
        print(f"\n  [BUG] DataFrame has NaN in these columns (will break model):")
        for c in nan_cols:
            print(f"    NaN → {repr(c)}")
    else:
        print("  DataFrame columns all filled (no NaN). ✓")

    return len(missing_in_ext) == 0 and len(nan_cols) == 0


def sample_pcap_flows(model, model_features):
    """
    Stream the PCAP, extract up to MAX_FLOWS post-skip flows,
    run raw model predictions, and print diagnostics.
    """
    from core.extractor_v4 import FeatureExtractorV4

    print(f"\n{'─'*70}")
    print(f"STEP 3 – Live Feature Extraction & Raw Model Probabilities")
    print(f"  (sampling up to {MAX_FLOWS} flows after the time-skip)")
    print(f"{'─'*70}")

    if not os.path.exists(PCAP_PATH):
        print(f"  [SKIP] PCAP not found: {PCAP_PATH}")
        return

    ext = FeatureExtractorV4()
    records = []         # list of (feature_dict, proba_attack, proba_normal, has_nan)
    flow_count = 0
    pkt_count  = 0
    skipped    = 0

    try:
        with PcapReader(PCAP_PATH) as reader:
            for pkt in reader:
                pkt_count += 1
                pkt_time = float(pkt.time) - TIME_OFFSET

                if pkt_time < START_TS:
                    skipped += 1
                    if skipped % 200_000 == 0:
                        t = datetime.fromtimestamp(pkt_time).strftime('%H:%M:%S')
                        print(f"  Skipping... {t}", end="\r")
                    continue

                feats = ext.extract_features(pkt)
                if feats is None:
                    continue

                # Build DataFrame exactly as ids_model_v4 does
                row = {k: feats.get(k, float("nan")) for k in model_features}
                df  = pd.DataFrame([row], columns=model_features)

                has_nan = df.isnull().values.any()
                probas  = model.predict_proba(df)[0]   # [P(normal), P(attack)]
                p_normal = float(probas[0])
                p_attack = float(probas[1])

                records.append({
                    "features": feats,
                    "p_attack": p_attack,
                    "p_normal": p_normal,
                    "has_nan":  has_nan,
                })
                flow_count += 1

                if flow_count % PRINT_EVERY == 0:
                    print(f"  Flows sampled: {flow_count}", end="\r")

                if flow_count >= MAX_FLOWS:
                    break

    except KeyboardInterrupt:
        print("\n  [Stopped by user]")

    print(f"\n  Total packets read : {pkt_count}")
    print(f"  Packets skipped    : {skipped}")
    print(f"  Flows sampled      : {flow_count}")

    if not records:
        print("  [WARNING] No flows were extracted – check time offset or PCAP path.")
        return

    # ── Raw probability distribution ─────────────────────────────────────────
    p_attacks = np.array([r["p_attack"] for r in records])
    nan_rows  = sum(r["has_nan"] for r in records)

    print(f"\n{'─'*70}")
    print("STEP 4 – Raw Attack Probability Distribution (no threshold applied)")
    print(f"{'─'*70}")
    print(f"  Rows with NaN in model input : {nan_rows}")
    print(f"  P(attack) min   : {p_attacks.min():.6f}")
    print(f"  P(attack) max   : {p_attacks.max():.6f}")
    print(f"  P(attack) mean  : {p_attacks.mean():.6f}")
    print(f"  P(attack) median: {np.median(p_attacks):.6f}")

    buckets = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    print(f"\n  P(attack) distribution across {flow_count} sampled flows:")
    for lo, hi in buckets:
        count = int(((p_attacks >= lo) & (p_attacks < hi)).sum())
        bar = "#" * (count * 30 // max(flow_count, 1))
        print(f"    [{lo:.1f}–{hi:.1f}) : {count:>5}  {bar}")

    above_50  = int((p_attacks >= 0.50).sum())
    above_70  = int((p_attacks >= 0.70).sum())
    above_90  = int((p_attacks >= 0.90).sum())
    print(f"\n  Flows where P(attack) ≥ 0.50  (model threshold) : {above_50}")
    print(f"  Flows where P(attack) ≥ 0.70                    : {above_70}")
    print(f"  Flows where P(attack) ≥ 0.90                    : {above_90}")

    if above_50 == 0:
        print("\n  [DIAGNOSIS] The model NEVER exceeds 0.50 attack probability.")
        print("  → Most likely cause: model was trained on different features than")
        print("    what the extractor now produces. Re-train with train_enhanced_v4.py.")
    else:
        print(f"\n  [OK] Model does flag {above_50} flows as attacks but threshold/guardrails")
        print("  may be filtering them. Check ids_model_v4.py confidence threshold.")

    # ── Feature value sanity check ────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("STEP 5 – Feature Value Ranges (sample)")
    print(f"{'─'*70}")
    df_all = pd.DataFrame([r["features"] for r in records])
    print(df_all.describe().T[["min", "mean", "max"]].to_string())

    # ── Top suspicious flows ──────────────────────────────────────────────────
    top_n = sorted(records, key=lambda x: x["p_attack"], reverse=True)[:5]
    print(f"\n{'─'*70}")
    print("STEP 6 – Top 5 Highest Attack-Probability Flows")
    print(f"{'─'*70}")
    for rank, r in enumerate(top_n, 1):
        print(f"\n  Rank #{rank}  P(attack)={r['p_attack']:.4f}  P(normal)={r['p_normal']:.4f}")
        for k, v in r["features"].items():
            print(f"    {k:40s} : {v:.4f}")


def main():
    print(SEP)
    print("  XAI-IDS V4 Diagnostic Tool")
    print(SEP)

    # Step 1: load model
    model, model_features = load_model_and_features()

    # Step 2: check alignment
    aligned = check_feature_alignment(model_features)

    # Step 3-6: live sampling
    sample_pcap_flows(model, model_features)

    print(f"\n{SEP}")
    print("  Diagnostic complete.")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
