# =========================================================
# VINAMILK FINANCIAL RISK PREDICTION
# LOGISTIC REGRESSION MODEL
# =========================================================

import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    roc_auc_score,

    confusion_matrix,

    classification_report

)

# =========================================================
# CONFIG
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "Simplize_VNM_FinancialIndicator_20250315.xlsx"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "scaler.pkl"
)

# =========================================================
# LOAD EXCEL
# =========================================================

print("=" * 60)
print("LOADING DATA...")
print("=" * 60)

df = pd.read_excel(
    DATA_PATH,
    header=None
)

print("Dataset Shape:", df.shape)

# =========================================================
# CLEAN DATA
# =========================================================

df.rename(
    columns={
        0: "Indicator"
    },
    inplace=True
)

# =========================================================
# HELPER FUNCTION
# =========================================================

def extract_row(keyword):

    row = df[
        df["Indicator"]
        .astype(str)
        .str.contains(
            keyword,
            case=False,
            na=False
        )
    ]

    if len(row) == 0:

        print(f"Cannot find: {keyword}")

        return None

    values = row.iloc[0, 1:]

    values = pd.to_numeric(
        values,
        errors="coerce"
    )

    return values.values

# =========================================================
# EXTRACT DATA
# =========================================================

revenue = extract_row(
    "Doanh thu thuần"
)

net_profit = extract_row(
    "Lợi nhuận sau thuế"
)

total_assets = extract_row(
    "Tổng tài sản"
)

short_debt = extract_row(
    "Nợ ngắn hạn"
)

long_debt = extract_row(
    "Nợ dài hạn"
)

equity = extract_row(
    "Vốn chủ sở hữu"
)

roa = extract_row(
    "ROA"
)

roe = extract_row(
    "ROE"
)

# =========================================================
# CREATE DATASET
# =========================================================

data = pd.DataFrame({

    "Revenue": revenue,

    "Net_Profit": net_profit,

    "Total_Assets": total_assets,

    "Short_Debt": short_debt,

    "Long_Debt": long_debt,

    "Equity": equity,

    "ROA": roa,

    "ROE": roe

})

# =========================================================
# DROP NULL
# =========================================================

data = data.dropna()

print("\nDATASET:")
print(data.head())

print("\nShape:", data.shape)

# =========================================================
# FEATURE ENGINEERING
# =========================================================

data["Total_Debt"] = (

    data["Short_Debt"]

    +

    data["Long_Debt"]

)

data["Debt_Ratio"] = (

    data["Total_Debt"]

    /

    data["Total_Assets"]

)

data["Profit_Margin"] = (

    data["Net_Profit"]

    /

    data["Revenue"]

)

data["Equity_Ratio"] = (

    data["Equity"]

    /

    data["Total_Assets"]

)

# =========================================================
# CREATE RISK SCORE
# =========================================================

data["Risk_Score"] = (

    data["Debt_Ratio"] * 0.45

    -

    data["ROA"] * 0.03

    -

    data["ROE"] * 0.02

    -

    data["Profit_Margin"] * 0.20

    -

    data["Equity_Ratio"] * 0.10

)

# =========================================================
# CREATE TARGET
# =========================================================

threshold = data["Risk_Score"].median()

data["Risk"] = (
    data["Risk_Score"] > threshold
).astype(int)

# =========================================================
# FEATURES
# =========================================================

features = [

    "Revenue",

    "Net_Profit",

    "Total_Assets",

    "Total_Debt",

    "Equity",

    "ROA",

    "ROE",

    "Debt_Ratio",

    "Profit_Margin"

]

X = data[features]

y = data["Risk"]

print("\nFeature Shape:", X.shape)

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)

# =========================================================
# SCALING
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

# =========================================================
# TRAIN MODEL
# =========================================================

print("\nTRAINING MODEL...")

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_scaled,
    y_train
)

# =========================================================
# PREDICTION
# =========================================================

y_pred = model.predict(
    X_test_scaled
)

y_prob = model.predict_proba(
    X_test_scaled
)[:,1]

# =========================================================
# EVALUATION
# =========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nCONFUSION MATRIX")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# =========================================================
# SAVE MODEL
# =========================================================

joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    scaler,
    SCALER_PATH
)

print("\nMODEL SAVED SUCCESSFULLY")

print("\nDONE.")
