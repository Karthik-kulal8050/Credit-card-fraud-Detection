import os
import json
import joblib
import pandas as pd

from src.preprocessing import prepare_features, SCALE_COLUMNS


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "artifacts",
    "fraud_random_forest.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "artifacts",
    "scaler.pkl"
)

THRESHOLD_PATH = os.path.join(
    BASE_DIR,
    "artifacts",
    "threshold.json"
)


# --------------------------------------------------
# Load artifacts
# --------------------------------------------------

model = joblib.load(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

with open(THRESHOLD_PATH, "r") as f:
    threshold_data = json.load(f)

THRESHOLD = threshold_data["threshold"]


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict_fraud(transaction):
    """
    Predict whether a transaction is fraudulent.

    Parameters
    ----------
    transaction : dict or pandas DataFrame
        Must contain:
        Time, V1-V28, Amount

    Returns
    -------
    result : str
        Fraud or Legitimate

    probability : float
        Probability of fraud
    """

    # Convert dictionary to DataFrame
    if isinstance(transaction, dict):
        transaction = pd.DataFrame([transaction])

    elif not isinstance(transaction, pd.DataFrame):
        raise TypeError(
            "Transaction must be a dictionary or pandas DataFrame."
        )

    # Prepare features in the correct order
    features = prepare_features(transaction)

    # Create a copy before scaling
    features_scaled = features.copy()

    # Apply the SAME scaler used during training
    features_scaled[SCALE_COLUMNS] = scaler.transform(
        features[SCALE_COLUMNS]
    )

    # Get probability of fraud
    fraud_probability = model.predict_proba(
        features_scaled
    )[:, 1][0]

    # Apply saved validation threshold
    if fraud_probability >= THRESHOLD:
        result = "Fraud"
    else:
        result = "Legitimate"

    return result, float(fraud_probability)


# --------------------------------------------------
# Test prediction
# --------------------------------------------------

if __name__ == "__main__":

    # Load one transaction from the original dataset
    DATA_PATH = os.path.join(
        BASE_DIR,
        "data",
        "creditcard.csv"
    )

    df = pd.read_csv(DATA_PATH)

    # Take one fraudulent transaction
    transaction = df[
        df["Class"] == 1
    ].iloc[0]

    # Remove target column
    transaction = transaction.drop(
        labels=["Class"]
    ).to_dict()

    result, probability = predict_fraud(
        transaction
    )

    print("\n" + "=" * 50)
    print("CREDIT CARD FRAUD PREDICTION")
    print("=" * 50)

    print("\nPrediction:", result)

    print(
        "Fraud Probability:",
        f"{probability * 100:.2f}%"
    )

    print(
        "Decision Threshold:",
        f"{THRESHOLD:.4f}"
    )

    print("=" * 50)