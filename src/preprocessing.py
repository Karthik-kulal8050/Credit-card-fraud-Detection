import pandas as pd


FEATURE_COLUMNS = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount"
]


SCALE_COLUMNS = [
    "Time",
    "Amount"
]


def prepare_features(data):
    """
    Select and order the features required by the model.

    Parameters
    ----------
    data : pandas.DataFrame
        Transaction data containing Time, V1-V28 and Amount.

    Returns
    -------
    pandas.DataFrame
        Features in the correct order.
    """

    data = data.copy()

    missing_columns = [
        column for column in FEATURE_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return data[FEATURE_COLUMNS]