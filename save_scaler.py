import boto3
import io
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

s3 = boto3.client('s3')
BUCKET = 'f1-pitstop-bucket'

print("Downloading training data...")
buf = io.BytesIO()
s3.download_fileobj(BUCKET, 'processed/X_train_lgbm.parquet', buf)
X_train = pd.read_parquet(io.BytesIO(buf.getvalue()))
print(f"Shape: {X_train.shape}")

print("Fitting StandardScaler...")
scaler = StandardScaler()
scaler.fit(X_train)

print("Uploading scaler to S3...")
buf2 = io.BytesIO()
pickle.dump(scaler, buf2)
buf2.seek(0)
s3.upload_fileobj(buf2, BUCKET, 'models/lr_scaler_latest.pkl')
print("Done! Scaler saved to s3://f1-pitstop-bucket/models/lr_scaler_latest.pkl")
