# 💳 Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions using **Random Forest, SMOTE, and classification threshold tuning**.

The project focuses on handling highly imbalanced transaction data and evaluating the model using fraud-sensitive metrics such as **Precision, Recall, F1-score, ROC-AUC, and PR-AUC**.

---

## 📌 Project Overview

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions represent only a very small percentage of all transactions.

A model that predicts every transaction as legitimate can achieve extremely high accuracy while completely failing to detect fraud.

Therefore, this project focuses on:

- Handling class imbalance
- Preventing data leakage
- Feature scaling
- Comparing classification approaches
- Using SMOTE for minority-class oversampling
- Optimizing the classification threshold
- Evaluating fraud detection using Precision, Recall, F1-score, ROC-AUC and PR-AUC

---

## 📊 Dataset

The project uses the **Credit Card Fraud Detection dataset** containing anonymized credit card transactions.

Original dataset:

- **284,807 transactions**
- **492 fraudulent transactions**
- **30 input features**
- **1 target variable**

### Features

| Feature | Description |
|---|---|
| `Time` | Seconds elapsed between the transaction and the first transaction |
| `V1`–`V28` | Anonymized PCA-transformed transaction features |
| `Amount` | Transaction amount |
| `Class` | Target variable: `0 = Legitimate`, `1 = Fraud` |

The dataset is highly imbalanced, with fraud representing approximately **0.17%** of the original transactions.

---

## 🔍 Data Preprocessing

The following preprocessing steps were performed:

### 1. Duplicate Removal

The original dataset contained:

```text
1,081 duplicate rows

Exact duplicates were removed before model training.

Final dataset:

283,726 transactions

After duplicate removal:

Legitimate: 283,253
Fraud:          473
2. Train / Validation / Test Split

A stratified split was used to preserve the minority-class distribution.

Training:    80%
Validation: 10%
Testing:     10%

Final sizes:

Train:      226,980
Validation:  28,373
Test:        28,373
3. Feature Scaling

StandardScaler was applied to:

Time
Amount

The scaler was fitted only on the training data and then applied to validation and test data.

The anonymized V1–V28 features were already PCA-transformed and were not separately scaled.

⚖️ Handling Class Imbalance

Because fraud transactions are extremely rare, the project uses SMOTE (Synthetic Minority Oversampling Technique).

SMOTE was applied only to the training data.

Before SMOTE

Legitimate: 226,601
Fraud:          379

After SMOTE

Legitimate: 226,601
Fraud:       22,660

Validation and test data were kept untouched.

This prevents artificially changing the real-world class distribution during evaluation.

🤖 Model

The final model is:

Random Forest Classifier
        +
SMOTE
        +
Validation-based Threshold Tuning
Random Forest Configuration
n_estimators = 200
min_samples_leaf = 2
class_weight = None
random_state = 42
🎯 Threshold Tuning

The default classification threshold of 0.50 was not used.

Instead, fraud probabilities were generated on the validation set and different thresholds were evaluated.

The threshold that maximized validation F1-score was selected.

Selected threshold
0.6849

Therefore:

Fraud Probability >= 0.6849
        ↓
      Fraud

Fraud Probability < 0.6849
        ↓
    Legitimate

The threshold was selected using the validation set and then kept fixed for final test evaluation.

📈 Model Performance

The final model was evaluated on an untouched test set.

Test Results
Metric	Score
ROC-AUC	0.9861
PR-AUC	0.8809
Fraud Precision	0.93
Fraud Recall	0.83
Fraud F1-score	0.88
Confusion Matrix
                 Predicted
                 Legit   Fraud

Actual Legit     28323      3
Actual Fraud         8     39

This means:

28,323 legitimate transactions correctly identified
39 fraudulent transactions correctly identified
3 legitimate transactions incorrectly flagged as fraud
8 fraudulent transactions missed
📊 Why Accuracy Is Not the Main Metric

The dataset is extremely imbalanced.

If a model predicted every transaction as legitimate, it could still achieve approximately 99.8% accuracy, despite detecting zero fraud.

Therefore, this project emphasizes:

Precision
Recall
F1-score
PR-AUC
ROC-AUC
Confusion Matrix

In fraud detection, Recall is important because missing a fraudulent transaction can be costly, while Precision matters because excessive false alarms can inconvenience legitimate customers.

🧪 Prediction Pipeline

The trained model can be used through the prediction pipeline:

Transaction
     ↓
Feature Validation
     ↓
Scale Time + Amount
     ↓
Random Forest
     ↓
Fraud Probability
     ↓
Threshold = 0.6849
     ↓
Fraud / Legitimate

The following artifacts are saved:

artifacts/
├── fraud_random_forest.pkl
├── scaler.pkl
└── threshold.json
🌐 Streamlit Application

The project includes a Streamlit web application.

Users can upload a CSV containing a single transaction with:

Time, V1, V2, ..., V28, Amount

The application displays:

Fraud probability
Decision threshold
Fraud / Legitimate prediction
Transaction information
Example
Fraud Probability: 87.71%

Decision Threshold: 68.49%

Prediction:
🚨 FRAUD DETECTED
📁 Project Structure
Credit-Card-Fraud-Detection/
│
├── artifacts/
│   ├── fraud_random_forest.pkl
│   ├── scaler.pkl
│   └── threshold.json
│
├── data/
│   └── creditcard.csv
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md

The dataset is excluded from GitHub through .gitignore.

⚙️ Installation

Clone the repository and install the required dependencies:

pip install -r requirements.txt
🚀 Run the Streamlit Application

From the project root:

streamlit run app.py
🔮 Command-Line Prediction

The prediction script can also be executed directly:

python src/predict.py
🧠 Key Learning Outcomes

Through this project, I learned and implemented:

Exploratory Data Analysis
Data quality checking
Duplicate detection
Stratified train/validation/test splitting
Feature scaling
Highly imbalanced classification
SMOTE
Random Forest
Classification threshold tuning
Precision / Recall trade-offs
F1-score optimization
ROC-AUC
PR-AUC
Confusion matrix analysis
Model serialization
Building a reusable prediction pipeline
Streamlit deployment
⚠️ Limitations
The dataset contains anonymized PCA features, making individual feature interpretation difficult.
The model is trained on a historical dataset and may not represent modern transaction patterns.
A fraud probability is a model score and should not be interpreted as a guaranteed real-world probability.
The project is intended for educational and portfolio purposes and is not a production banking fraud detection system.
🛠️ Technologies Used
Python
Pandas
NumPy
Scikit-learn
Imbalanced-learn
Joblib
Streamlit