import boto3
import io
import pickle
import pandas as pd
import numpy as np

s3 = boto3.client('s3')
BUCKET = 'f1-pitstop-bucket'

def load_pkl(key):
    buf = io.BytesIO()
    s3.download_fileobj(BUCKET, key, buf)
    buf.seek(0)
    return pickle.load(buf)

print("Loading models...")
lgbm_model = load_pkl('models/lgbm_model_latest.pkl')
lr_model   = load_pkl('models/lr_model_latest.pkl')
scaler     = load_pkl('models/lr_scaler_latest.pkl')
print("All loaded OK")

FEATURE_COLS = [
    "TyreLife", "Cumulative_Degradation", "LapNumber", "RaceProgress",
    "LapTime_Delta", "Stint", "Position", "TyreLife_LapNumber_ratio",
    "Compound_te", "Race_te",
]

# Check LightGBM feature names
if hasattr(lgbm_model, 'feature_name_'):
    print(f"\nLightGBM feature names: {lgbm_model.feature_name_}")

# Test samples: early lap vs late lap
samples = [
    {"TyreLife": 2,  "Cumulative_Degradation": 0.05, "LapNumber": 5,  "RaceProgress": 0.1,
     "LapTime_Delta": 0.0, "Stint": 1, "Position": 5, "TyreLife_LapNumber_ratio": 0.4,
     "Compound_te": 0.21, "Race_te": 0.19},
    {"TyreLife": 30, "Cumulative_Degradation": 0.80, "LapNumber": 40, "RaceProgress": 0.8,
     "LapTime_Delta": 1.5, "Stint": 2, "Position": 3, "TyreLife_LapNumber_ratio": 0.75,
     "Compound_te": 0.35, "Race_te": 0.30},
]

for i, s in enumerate(samples):
    df = pd.DataFrame([s])[FEATURE_COLS]
    X_raw = df.to_numpy()
    X_scaled = scaler.transform(X_raw)

    lgbm_prob = lgbm_model.predict_proba(X_raw)[:, 1][0]
    lr_prob   = lr_model.predict_proba(X_scaled)[:, 1][0]

    print(f"\nSample {i+1} (TyreLife={s['TyreLife']}, LapNumber={s['LapNumber']}):")
    print(f"  LightGBM  prob = {lgbm_prob:.4f}  → {'PIT' if lgbm_prob >= 0.5 else 'STAY OUT'}")
    print(f"  LogisticR prob = {lr_prob:.4f}  → {'PIT' if lr_prob >= 0.5 else 'STAY OUT'}")
