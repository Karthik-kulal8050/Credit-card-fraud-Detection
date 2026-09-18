import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve
)

from imblearn.over_sampling import SMOTE

from src.preprocessing import prepare_features, SCALE_COLUMNS
import mlflow
import mlflow.sklearn

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "creditcard.csv"
)

ARTIFACTS_DIR = os.path.join(
    BASE_DIR,
    "artifacts"
)

os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.10
VALIDATION_SIZE_FROM_TEMP = 0.111111

SMOTE_RATIO = 0.10

N_ESTIMATORS = 200
MIN_SAMPLES_LEAF = 2


mlflow.set_experiment("fraud-detection")
mlflow.start_run()
mlflow.log_params({
    "model": "Random Forest",
    "n_estimators": N_ESTIMATORS,
    "min_samples_leaf": MIN_SAMPLES_LEAF,
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "smote_ratio": SMOTE_RATIO
})


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("Loading dataset...")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)

mlflow.log_params({
    "dataset_rows": df.shape[0],
    "dataset_columns": df.shape[1]
})


# ============================================================
# 2. REMOVE EXACT DUPLICATES
# ============================================================

print("\nRemoving exact duplicate rows...")

duplicate_count = df.duplicated().sum()
mlflow.log_param("duplicate_rows_removed", int(duplicate_count))

print("Duplicate rows found:", duplicate_count)

df = df.drop_duplicates().copy()

print("Dataset shape after removing duplicates:", df.shape)


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

X = prepare_features(df)

y = df["Class"]

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

print("\nClass distribution:")
print(y.value_counts())


# ============================================================
# 4. TRAIN / TEMP SPLIT
# ============================================================

print("\n" + "=" * 60)
print("Creating train / validation / test split...")
print("=" * 60)

X_train_temp, X_test, y_train_temp, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)


# ============================================================
# 5. TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X_train_temp,
    y_train_temp,
    test_size=VALIDATION_SIZE_FROM_TEMP,
    stratify=y_train_temp,
    random_state=RANDOM_STATE
)

print("\nTrain shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)

print("\nTrain class distribution:")
print(y_train.value_counts())

print("\nValidation class distribution:")
print(y_val.value_counts())

print("\nTest class distribution:")
print(y_test.value_counts())


# ============================================================
# 6. FEATURE SCALING
# ============================================================

print("\n" + "=" * 60)
print("Scaling Time and Amount...")
print("=" * 60)

scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()


# Fit ONLY on training data
X_train_scaled[SCALE_COLUMNS] = scaler.fit_transform(
    X_train[SCALE_COLUMNS]
)

# Transform validation and test using the training scaler
X_val_scaled[SCALE_COLUMNS] = scaler.transform(
    X_val[SCALE_COLUMNS]
)

X_test_scaled[SCALE_COLUMNS] = scaler.transform(
    X_test[SCALE_COLUMNS]
)

print("Scaling completed.")


# ============================================================
# 7. APPLY SMOTE ONLY TO TRAINING DATA
# ============================================================

print("\n" + "=" * 60)
print("Applying SMOTE...")
print("=" * 60)

smote = SMOTE(
    sampling_strategy=SMOTE_RATIO,
    random_state=RANDOM_STATE
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("Original training shape:", X_train_scaled.shape)
print("After SMOTE:", X_train_smote.shape)

mlflow.log_params({
    "train_samples_before_smote": X_train_scaled.shape[0],
    "train_samples_after_smote": X_train_smote.shape[0]
})


print("\nOriginal class distribution:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())


# ============================================================
# 8. TRAIN RANDOM FOREST
# ============================================================

print("\n" + "=" * 60)
print("Training Random Forest...")
print("=" * 60)

model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    class_weight=None,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(
    X_train_smote,
    y_train_smote
)

print("Random Forest training completed.")


# ============================================================
# 9. VALIDATION PROBABILITIES
# ============================================================

y_val_proba = model.predict_proba(
    X_val_scaled
)[:, 1]


# ============================================================
# 10. FIND BEST THRESHOLD USING VALIDATION SET
# ============================================================

print("\n" + "=" * 60)
print("Finding best classification threshold...")
print("=" * 60)

precision, recall, thresholds = precision_recall_curve(
    y_val,
    y_val_proba
)

f1_scores = (
    2 * precision[:-1] * recall[:-1]
    / (
        precision[:-1]
        + recall[:-1]
        + 1e-10
    )
)

best_index = f1_scores.argmax()

best_threshold = float(
    thresholds[best_index]
)

best_precision = float(
    precision[best_index]
)

best_recall = float(
    recall[best_index]
)

best_f1 = float(
    f1_scores[best_index]
)

print("Best threshold:", best_threshold)
print("Validation precision:", best_precision)
print("Validation recall:", best_recall)
print("Validation F1:", best_f1)


# ============================================================
# 11. VALIDATION EVALUATION
# ============================================================

y_val_pred = (
    y_val_proba >= best_threshold
).astype(int)

print("\nValidation Confusion Matrix:")
print(
    confusion_matrix(
        y_val,
        y_val_pred
    )
)

print("\nValidation Classification Report:")
print(
    classification_report(
        y_val,
        y_val_pred,
        target_names=[
            "Legitimate",
            "Fraud"
        ]
    )
)

print(
    "Validation ROC-AUC:",
    roc_auc_score(
        y_val,
        y_val_proba
    )
)

print(
    "Validation PR-AUC:",
    average_precision_score(
        y_val,
        y_val_proba
    )
)


# ============================================================
# 12. FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

y_test_proba = model.predict_proba(
    X_test_scaled
)[:, 1]

# IMPORTANT:
# Use threshold selected from validation.
# Do NOT tune the threshold using test data.

y_test_pred = (
    y_test_proba >= best_threshold
).astype(int)

print("\nFinal Test Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_test_pred
    )
)

print("\nFinal Test Classification Report:")
print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=[
            "Legitimate",
            "Fraud"
        ]
    )
)

test_roc_auc = roc_auc_score(
    y_test,
    y_test_proba
)

test_pr_auc = average_precision_score(
    y_test,
    y_test_proba
)

print("Final Test ROC-AUC:", test_roc_auc)
print("Final Test PR-AUC:", test_pr_auc)


# ============================================================
# 13. SAVE MODEL
# ============================================================

model_path = os.path.join(
    ARTIFACTS_DIR,
    "fraud_random_forest.pkl"
)

joblib.dump(
    model,
    model_path
)

print("\nModel saved to:")
print(model_path)


# ============================================================
# 14. SAVE SCALER
# ============================================================

scaler_path = os.path.join(
    ARTIFACTS_DIR,
    "scaler.pkl"
)

joblib.dump(
    scaler,
    scaler_path
)

print("\nScaler saved to:")
print(scaler_path)


# ============================================================
# 15. SAVE THRESHOLD
# ============================================================

threshold_path = os.path.join(
    ARTIFACTS_DIR,
    "threshold.json"
)

threshold_data = {
    "threshold": best_threshold
}

with open(
    threshold_path,
    "w"
) as f:
    json.dump(
        threshold_data,
        f,
        indent=4
    )

print("\nThreshold saved to:")
print(threshold_path)


mlflow.sklearn.log_model(
    model,
    name="fraud_random_forest",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)
mlflow.log_artifact(scaler_path)
mlflow.log_artifact(threshold_path)


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TRAINING PIPELINE COMPLETED")
print("=" * 60)

print(f"Test ROC-AUC : {test_roc_auc:.4f}")
print(f"Test PR-AUC  : {test_pr_auc:.4f}")
print(f"Threshold    : {best_threshold:.4f}")

mlflow.log_metrics({
    "validation_precision": best_precision,
    "validation_recall": best_recall,
    "validation_f1": best_f1,
    "test_roc_auc": test_roc_auc,
    "test_pr_auc": test_pr_auc,
    "decision_threshold": best_threshold
})


mlflow.end_run()

print("\nArtifacts:")
print("- fraud_random_forest.pkl")
print("- scaler.pkl")
print("- threshold.json")