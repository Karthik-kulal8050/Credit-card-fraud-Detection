import streamlit as st
import pandas as pd

from src.predict import predict_fraud, THRESHOLD
from src.preprocessing import FEATURE_COLUMNS


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("💳 Credit Card Fraud Detection")

st.write(
    "Machine learning system for detecting potentially "
    "fraudulent credit card transactions."
)


# --------------------------------------------------
# Model Information
# --------------------------------------------------

with st.expander("ℹ️ About the Model"):

    st.write(
        """
        **Model:** Random Forest

        **Imbalance Handling:** SMOTE

        **Threshold Tuning:** Validation-set F1 optimization

        **Test ROC-AUC:** 0.9861

        **Test PR-AUC:** 0.8809

        **Fraud F1-score:** 0.88

        **Fraud Recall:** 0.83

        The classification threshold was selected using the
        validation set rather than using the default 0.50 threshold.
        """
    )


# --------------------------------------------------
# Input Instructions
# --------------------------------------------------

st.subheader("📁 Upload Transaction")

st.write(
    "Upload a CSV file containing exactly one transaction."
)

st.info(
    "The CSV must contain: Time, V1–V28, and Amount."
)


uploaded_file = st.file_uploader(
    "Choose a transaction CSV",
    type=["csv"]
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if uploaded_file is not None:

    try:

        transaction_df = pd.read_csv(
            uploaded_file
        )

        st.subheader("Transaction Data")

        st.dataframe(
            transaction_df,
            use_container_width=True
        )

        # Check exactly one transaction
        if len(transaction_df) != 1:

            st.error(
                "Please upload a CSV containing exactly one transaction."
            )

        else:

            # Check required columns
            missing_columns = [
                column
                for column in FEATURE_COLUMNS
                if column not in transaction_df.columns
            ]

            if missing_columns:

                st.error(
                    f"Missing required columns: {missing_columns}"
                )

            else:

                if st.button(
                    "🔍 Analyze Transaction",
                    use_container_width=True
                ):

                    result, probability = predict_fraud(
                        transaction_df
                    )

                    st.subheader(
                        "Prediction Result"
                    )

                    # ------------------------------------------
                    # Fraud
                    # ------------------------------------------

                    if result == "Fraud":

                        st.error(
                            "🚨 FRAUD DETECTED"
                        )

                    # ------------------------------------------
                    # Legitimate
                    # ------------------------------------------

                    else:

                        st.success(
                            "✅ TRANSACTION APPEARS LEGITIMATE"
                        )

                    # ------------------------------------------
                    # Metrics
                    # ------------------------------------------

                    col1, col2 = st.columns(2)

                    with col1:

                        st.metric(
                            "Fraud Probability",
                            f"{probability * 100:.2f}%"
                        )

                    with col2:

                        st.metric(
                            "Decision Threshold",
                            f"{THRESHOLD * 100:.2f}%"
                        )

                    # ------------------------------------------
                    # Probability Bar
                    # ------------------------------------------

                    st.write(
                        "Fraud Probability"
                    )

                    st.progress(
                        min(probability, 1.0)
                    )

    except Exception as e:

        st.error(
            f"Unable to process the uploaded file: {e}"
        )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Credit Card Fraud Detection | "
    "Random Forest + SMOTE | "
    "Machine Learning Project"
)