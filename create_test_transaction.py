import pandas as pd

df = pd.read_csv("data/creditcard.csv")

transaction = df[
    df["Class"] == 0
].iloc[[0]]

transaction.drop(
    columns=["Class"]
).to_csv(
    "test_legitimate_transaction.csv",
    index=False
)

print("Legitimate transaction created.")