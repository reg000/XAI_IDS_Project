"""
inspect_csv.py – Print the exact column names in your CIC-IDS2017 CSV.
Run this once to find the correct names for train_enhanced_v4.py.

Usage:
    python inspect_csv.py
"""
import sys
import pandas as pd

CSV_PATH = "data/PortScan.csv"   # adjust if needed

try:
    df = pd.read_csv(CSV_PATH, nrows=0)  # headers only, no data loaded
except FileNotFoundError:
    print(f"[!] File not found: {CSV_PATH}")
    sys.exit(1)

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
    ' Label',
]

print(f"\nAll {len(df.columns)} columns in {CSV_PATH}:\n")
for i, col in enumerate(df.columns, 1):
    print(f"  [{i:>3}] {repr(col)}")

print(f"\n{'─'*60}")
print("Checking FEATURES_ENHANCED against actual CSV columns:")
print(f"{'─'*60}")

missing, found = [], []
for feat in FEATURES_ENHANCED:
    if feat in df.columns:
        found.append(feat)
    else:
        # Try to find close matches (strip spaces, lowercase)
        norm = feat.strip().lower()
        candidates = [c for c in df.columns if c.strip().lower() == norm]
        missing.append((feat, candidates))

for feat in found:
    print(f"  ✓  {repr(feat)}")

for feat, candidates in missing:
    if candidates:
        print(f"  ✗  {repr(feat)}")
        print(f"       → Closest match in CSV: {[repr(c) for c in candidates]}")
    else:
        print(f"  ✗  {repr(feat)}  (no close match found)")
