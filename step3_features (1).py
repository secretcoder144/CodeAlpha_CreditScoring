import numpy as np
import pandas as pd

# Input: credit_clean.csv from Step 2
# Month order in this dataset: 1 = most recent (Sept), 6 = oldest (April)
df = pd.read_csv("credit_clean.csv")

pay_cols = [f"PAY_{i}" for i in range(1, 7)]        # repayment status
bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]   # bill amount
amt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]     # amount paid

# --- 1. Payment delay features -------------------------------------------
# -2 / -1 / 0 mean "no delay", so clip to 0. Only 1 or more means a real delay.
delays = df[pay_cols].clip(lower=0)
df["avg_delay"] = delays.mean(axis=1)
df["max_delay"] = delays.max(axis=1)
df["late_months"] = (df[pay_cols] >= 1).sum(axis=1)        # how many months paid late
df["delay_trend"] = delays["PAY_1"] - delays["PAY_6"]      # positive = getting worse

# --- 2. Credit usage features ---------------------------------------------
util = df[bill_cols].div(df["LIMIT_BAL"], axis=0).clip(lower=0, upper=1.5)
df["avg_utilization"] = util.mean(axis=1)                  # how much of the limit is used
df["latest_utilization"] = util["BILL_AMT1"]
df["over_limit_months"] = (util > 1).sum(axis=1)

# --- 3. Repayment behaviour features --------------------------------------
total_bill = df[bill_cols].clip(lower=0).sum(axis=1)
total_paid = df[amt_cols].sum(axis=1)
df["payment_ratio"] = (total_paid / (total_bill + 1)).clip(0, 1)   # share of bills actually paid
df["zero_payment_months"] = (df[amt_cols] == 0).sum(axis=1)
df["avg_payment"] = df[amt_cols].mean(axis=1)

# --- 4. Bill trend --------------------------------------------------------
df["avg_bill"] = df[bill_cols].mean(axis=1)
df["bill_trend"] = df["BILL_AMT1"] - df["BILL_AMT6"]       # positive = debt is growing
df["debt_to_limit"] = df["avg_bill"].clip(lower=0) / df["LIMIT_BAL"]

# Safety check: no infinite or missing values
df = df.replace([np.inf, -np.inf], np.nan)
print("Missing after feature engineering:", df.isnull().sum().sum())

new_feats = ["avg_delay", "max_delay", "late_months", "delay_trend",
             "avg_utilization", "latest_utilization", "over_limit_months",
             "payment_ratio", "zero_payment_months", "avg_payment",
             "avg_bill", "bill_trend", "debt_to_limit"]

# How strongly each new feature relates to default (-1 to 1, bigger = stronger)
print(df[new_feats + ["default"]].corr()["default"].drop("default").round(3)
      .sort_values(key=abs, ascending=False))

print("Shape:", df.shape)
df.to_csv("credit_features.csv", index=False)
