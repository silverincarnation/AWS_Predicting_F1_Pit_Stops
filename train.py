"""
F1 Pit Stop Prediction — Model Training Script
===============================================
Trains a Logistic Regression and a LightGBM classifier.
All configuration is read from config.yaml; AWS credentials from .env or
IAM task role (when running inside ECS — no .env file needed on Fargate).

Usage
-----
  # Local (reads .env for credentials)
  python train.py

  # Override config path
  python train.py --config /app/config.yaml

  # Override S3 bucket at runtime
  python train.py --bucket my-other-bucket

ECS / Docker
------------
  The script is self-contained. Set environment variables via ECS Task
  Definition (AWS_REGION, S3_BUCKET) or rely on the ECS task IAM role for
  credentials — never hard-code keys.
"""

import argparse
import io
import json
import logging
import os
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import boto3
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

# Non-interactive backend — no display needed inside a container
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 (must follow backend set)

warnings.filterwarnings("ignore")



# Argument parsing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train F1 Pit Stop models on AWS ECS.")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="Override S3 bucket name from config / env var",
    )
    return parser.parse_args()

# Config & logging

def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")
    with open(path) as f:
        return yaml.safe_load(f)


def setup_logging(cfg: dict) -> logging.Logger:
    """Configure root logger from config."""
    log_cfg = cfg["logging"]
    logging.basicConfig(
        level=getattr(logging, log_cfg["level"]),
        format=log_cfg["format"],
        datefmt=log_cfg["datefmt"],
        stream=sys.stdout,  # Forward to CloudWatch Logs when running on ECS
    )
    return logging.getLogger("f1_training")


# S3 helpers
def get_s3_client(region: str) -> boto3.client:
    """
    Return a boto3 S3 client.
    On ECS Fargate the task IAM role provides credentials automatically.
    Locally, boto3 picks up credentials from .env / ~/.aws/credentials.
    """
    return boto3.client("s3", region_name=region)


def read_parquet_s3(s3_client, bucket: str, key: str) -> pd.DataFrame:
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def read_csv_series_s3(s3_client, bucket: str, key: str, col: str) -> pd.Series:
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    df  = pd.read_csv(io.BytesIO(obj["Body"].read()), index_col=0)
    return df[col].squeeze()


def upload_pickle_s3(s3_client, obj, bucket: str, key: str, logger: logging.Logger) -> None:
    buf = io.BytesIO()
    pickle.dump(obj, buf)
    buf.seek(0)
    s3_client.upload_fileobj(buf, bucket, key)
    logger.info("Uploaded → s3://%s/%s", bucket, key)


def upload_json_s3(s3_client, data: dict, bucket: str, key: str, logger: logging.Logger) -> None:
    body = json.dumps(data, indent=2).encode()
    s3_client.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    logger.info("Uploaded → s3://%s/%s", bucket, key)


def upload_bytes_s3(s3_client, local_path: str, bucket: str, key: str, logger: logging.Logger) -> None:
    with open(local_path, "rb") as fh:
        s3_client.upload_fileobj(fh, bucket, key)
    logger.info("Uploaded → s3://%s/%s", bucket, key)



#Metrics
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Return a dict of common binary-classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc"  : round(float(roc_auc_score(y_true, y_prob)), 4),
        "f1"       : round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall"   : round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "accuracy" : round(float(accuracy_score(y_true, y_pred)), 4),
    }


# Training routines
def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict,
    n_splits: int,
    seed: int,
    logger: logging.Logger,
) -> tuple:
    """
    5-fold CV → final model on full train set.
    Returns (final_model, oof_probs, test_probs, cv_aucs, metrics_test).
    """
    kf       = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof      = np.zeros(len(y_train))
    cv_aucs  = []

    logger.info("--- Logistic Regression: %d-fold CV ---", n_splits)
    t0 = time.time()

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train), start=1):
        X_tr,  X_val = X_train.iloc[tr_idx],  X_train.iloc[val_idx]
        y_tr,  y_val = y_train.iloc[tr_idx],  y_train.iloc[val_idx]

        m = LogisticRegression(**params)
        m.fit(X_tr, y_tr)

        val_prob      = m.predict_proba(X_val)[:, 1]
        oof[val_idx]  = val_prob
        fold_auc      = roc_auc_score(y_val, val_prob)
        cv_aucs.append(fold_auc)
        logger.info("  Fold %d/%d  —  val AUC = %.4f", fold, n_splits, fold_auc)

    elapsed  = time.time() - t0
    cv_mean  = float(np.mean(cv_aucs))
    cv_std   = float(np.std(cv_aucs))
    logger.info("LR CV AUC = %.4f ± %.4f  [%.1fs]", cv_mean, cv_std, elapsed)

    # Final model on full training set
    logger.info("Training final LR on full training set ...")
    final_model = LogisticRegression(**params)
    final_model.fit(X_train, y_train)

    test_prob    = final_model.predict_proba(X_test)[:, 1]
    metrics_test = compute_metrics(y_test.values, test_prob)
    logger.info("LR holdout  AUC = %.4f  F1 = %.4f", metrics_test["roc_auc"], metrics_test["f1"])
    logger.info("\n%s", classification_report(y_test, (test_prob >= 0.5).astype(int)))

    return final_model, oof, test_prob, cv_aucs, metrics_test


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict,
    n_splits: int,
    seed: int,
    logger: logging.Logger,
) -> tuple:
    """
    5-fold CV with early stopping → final model using avg best iteration.
    Returns (final_model, oof_probs, test_probs, cv_aucs, metrics_test, best_iter).
    """
    lgbm_params           = params.copy()
    early_stopping_rounds = lgbm_params.pop("early_stopping_rounds", 50)

    kf            = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof           = np.zeros(len(y_train))
    cv_aucs       = []
    fold_models   = []

    logger.info("--- LightGBM: %d-fold CV ---", n_splits)
    t0 = time.time()

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train), start=1):
        X_tr,  X_val = X_train.iloc[tr_idx],  X_train.iloc[val_idx]
        y_tr,  y_val = y_train.iloc[tr_idx],  y_train.iloc[val_idx]

        m = lgb.LGBMClassifier(**lgbm_params)
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=True),
                lgb.log_evaluation(period=100),  # Print AUC every 100 iterations
            ],
        )

        val_prob     = m.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_prob
        fold_models.append(m)

        fold_auc = roc_auc_score(y_val, val_prob)
        cv_aucs.append(fold_auc)
        logger.info(
            "  Fold %d/%d  —  val AUC = %.4f  |  best iter = %d",
            fold, n_splits, fold_auc, m.best_iteration_,
        )

    elapsed  = time.time() - t0
    cv_mean  = float(np.mean(cv_aucs))
    cv_std   = float(np.std(cv_aucs))
    logger.info("LGBM CV AUC = %.4f ± %.4f  [%.1fs]", cv_mean, cv_std, elapsed)

    # Final model — use avg best iteration (no early stopping needed)
    best_iter = int(np.mean([m.best_iteration_ for m in fold_models]))
    logger.info("Training final LightGBM (n_estimators=%d) on full training set ...", best_iter)

    final_params = lgbm_params.copy()
    final_params["n_estimators"] = best_iter

    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(X_train, y_train)

    test_prob    = final_model.predict_proba(X_test)[:, 1]
    metrics_test = compute_metrics(y_test.values, test_prob)
    logger.info("LGBM holdout AUC = %.4f  F1 = %.4f", metrics_test["roc_auc"], metrics_test["f1"])
    logger.info("\n%s", classification_report(y_test, (test_prob >= 0.5).astype(int)))

    return final_model, oof, test_prob, cv_aucs, metrics_test, best_iter

# Visualisations
def plot_roc_curves(
    y_test: pd.Series,
    test_prob_lr: np.ndarray,
    test_prob_lgbm: np.ndarray,
    cv_auc_lr: float,
    cv_auc_lgbm: float,
    save_path: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (name, y_prob, auc_cv) in zip(
        axes,
        [
            ("Logistic Regression", test_prob_lr,   cv_auc_lr),
            ("LightGBM",            test_prob_lgbm, cv_auc_lgbm),
        ],
    ):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_test    = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, lw=2, label=f"Test AUC = {auc_test:.4f}")
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
        ax.set_title(f"{name}\nCV AUC = {auc_cv:.4f}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
    plt.suptitle("ROC Curves — F1 Pit Stop Prediction", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_feature_importance(model: lgb.LGBMClassifier, save_path: str, top_n: int = 20) -> None:
    importance_df = (
        pd.DataFrame(
            {"feature": model.feature_name_, "importance": model.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .head(top_n)
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(importance_df["feature"][::-1], importance_df["importance"][::-1], color="#3cb371")
    ax.set_title(f"LightGBM — Top {top_n} Feature Importances (split)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# Main

def main() -> None:
    args = parse_args()
    try:
        from dotenv import load_dotenv  # optional in container
        load_dotenv()
    except ImportError:
        pass

    # Config
    cfg    = load_config(args.config)
    logger = setup_logging(cfg)
    logger.info("Config loaded from %s", args.config)

    BUCKET   = args.bucket or os.environ.get("S3_BUCKET", cfg["s3"]["bucket"])
    REGION   = os.environ.get("AWS_REGION", cfg["aws"]["region"])
    PROC_PFX = cfg["s3"]["processed_prefix"]
    MDL_PFX  = cfg["s3"]["models_prefix"]
    SEED     = cfg["training"]["seed"]
    N_SPLITS = cfg["training"]["n_splits"]
    TARGET   = cfg["training"]["target"]
    RUN_TS   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    logger.info("Bucket    : s3://%s", BUCKET)
    logger.info("Region    : %s", REGION)
    logger.info("Run stamp : %s", RUN_TS)

    # S3 client
    s3 = get_s3_client(REGION)

    # Load data from S3
    logger.info("Loading processed datasets from S3 ...")
    X_train_lgbm = read_parquet_s3(s3, BUCKET, PROC_PFX + "X_train_lgbm.parquet")
    X_test_lgbm  = read_parquet_s3(s3, BUCKET, PROC_PFX + "X_test_lgbm.parquet")
    X_train_lr   = read_parquet_s3(s3, BUCKET, PROC_PFX + "X_train_lr.parquet")
    X_test_lr    = read_parquet_s3(s3, BUCKET, PROC_PFX + "X_test_lr.parquet")
    y_train      = read_csv_series_s3(s3, BUCKET, PROC_PFX + "y_train.csv", TARGET)
    y_test       = read_csv_series_s3(s3, BUCKET, PROC_PFX + "y_test.csv",  TARGET)

    logger.info("X_train_lgbm : %s", X_train_lgbm.shape)
    logger.info("X_train_lr   : %s", X_train_lr.shape)
    logger.info("y_train pos  : %.3f", y_train.mean())

    # Train models
    model_lr, _, test_prob_lr, cv_aucs_lr, metrics_lr = train_logistic_regression(
        X_train_lr, y_train, X_test_lr, y_test,
        params=cfg["logistic_regression"],
        n_splits=N_SPLITS, seed=SEED, logger=logger,
    )

    model_lgbm, _, test_prob_lgbm, cv_aucs_lgbm, metrics_lgbm, best_iter = train_lightgbm(
        X_train_lgbm, y_train, X_test_lgbm, y_test,
        params=cfg["lightgbm"],
        n_splits=N_SPLITS, seed=SEED, logger=logger,
    )

    winner = (
        "LightGBM"
        if metrics_lgbm["roc_auc"] >= metrics_lr["roc_auc"]
        else "Logistic Regression"
    )
    logger.info("Best model by ROC-AUC: %s", winner)

    # Plots
    roc_path  = "/tmp/roc_curves.png"
    feat_path = "/tmp/feature_importance.png"

    plot_roc_curves(
        y_test, test_prob_lr, test_prob_lgbm,
        float(np.mean(cv_aucs_lr)), float(np.mean(cv_aucs_lgbm)),
        roc_path,
    )
    plot_feature_importance(model_lgbm, feat_path)
    logger.info("Plots saved to /tmp/")

    # Upload artifacts to S3
    logger.info("Uploading artifacts to S3 ...")

    # Models (timestamped + latest)
    for key in [MDL_PFX + f"lr_model_{RUN_TS}.pkl", MDL_PFX + "lr_model_latest.pkl"]:
        upload_pickle_s3(s3, model_lr, BUCKET, key, logger)

    for key in [MDL_PFX + f"lgbm_model_{RUN_TS}.pkl", MDL_PFX + "lgbm_model_latest.pkl"]:
        upload_pickle_s3(s3, model_lgbm, BUCKET, key, logger)

    # Metrics JSON
    metrics_payload = {
        "run_timestamp": RUN_TS,
        "config": {"seed": SEED, "n_splits": N_SPLITS},
        "logistic_regression": {
            **metrics_lr,
            "cv_auc"        : round(float(np.mean(cv_aucs_lr)), 4),
            "cv_auc_std"    : round(float(np.std(cv_aucs_lr)), 4),
            "hyperparameters": cfg["logistic_regression"],
        },
        "lightgbm": {
            **metrics_lgbm,
            "cv_auc"        : round(float(np.mean(cv_aucs_lgbm)), 4),
            "cv_auc_std"    : round(float(np.std(cv_aucs_lgbm)), 4),
            "best_iteration": best_iter,
            "hyperparameters": cfg["lightgbm"],
        },
        "winner_by_roc_auc": winner,
    }

    for key in [MDL_PFX + f"metrics_{RUN_TS}.json", MDL_PFX + "metrics_latest.json"]:
        upload_json_s3(s3, metrics_payload, BUCKET, key, logger)

    # Plots
    upload_bytes_s3(s3, roc_path,  BUCKET, MDL_PFX + f"roc_curves_{RUN_TS}.png", logger)
    upload_bytes_s3(s3, feat_path, BUCKET, MDL_PFX + f"feature_importance_{RUN_TS}.png", logger)

    logger.info("All artifacts uploaded. Training complete.")


if __name__ == "__main__":
    main()
