import pandas as pd

# Needs: pip install pandas xlrd openpyxl
# 1. Load. The real column names are in the 2nd row, so header=1
df = pd.read_excel("default of credit card clients.xls", header=1)
print("Shape at start:", df.shape)

# 2. Rename columns so they are easy to read
df = df.rename(columns={
    "PAY_0": "PAY_1",                              # dataset skips PAY_1, fix the name
    "default payment next month": "default",       # target column
})

# 3. Drop ID. It is just a row number and carries no information
df = df.drop(columns=["ID"])

# 4. Missing values check (this dataset has none, but always prove it)
print("Missing values:", df.isnull().sum().sum())

# 5. Remove duplicate rows
print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)

# 6. Fix undocumented category codes
# EDUCATION: documented codes are 1-4. Codes 0, 5, 6 are not documented -> merge into 4 (others)
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
# MARRIAGE: documented codes are 1-3. Code 0 is not documented -> merge into 3 (others)
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

# 7. Make labels readable, then one-hot encode the category columns
df["SEX"] = df["SEX"].map({1: "male", 2: "female"})
df["EDUCATION"] = df["EDUCATION"].map({1: "graduate", 2: "university", 3: "high_school", 4: "other"})
df["MARRIAGE"] = df["MARRIAGE"].map({1: "married", 2: "single", 3: "other"})
df = pd.get_dummies(df, columns=["SEX", "EDUCATION", "MARRIAGE"], drop_first=True)

# 8. Check class balance (needed for Step 5)
print("Default rate:", round(df["default"].mean(), 3))

print("Shape at end:", df.shape)
df.to_csv("credit_clean.csv", index=False)
