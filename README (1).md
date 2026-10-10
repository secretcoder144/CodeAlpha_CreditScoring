# CodeAlpha_CreditScoring

Predict whether a credit card customer will default (miss the payment) next month, using their past billing and payment history.

## Dataset
- **Default of Credit Card Clients** (UCI Machine Learning Repository), 30,000 customers from Taiwan (2005).
- 23 input columns: credit limit, age, sex, education, marriage, 6 months of repayment status, bill amounts and payment amounts.
- Target: `default payment next month` (1 = defaulted). About 22% of customers defaulted.

## Steps
1. **Cleaning** (`step2_cleaning.py`): fixed header and column names, dropped `ID`, removed 35 duplicate rows, merged undocumented codes in `EDUCATION` and `MARRIAGE` into "other", one-hot encoded categories. No missing values.
2. **Feature engineering** (`step3_features.py`): 13 new features, such as `late_months`, `avg_delay`, `avg_utilization`, `payment_ratio`, `zero_payment_months`, `bill_trend`.
3. **Modelling** (`step4_to_8_model.py`):
   - Stratified 80/20 train/test split. Scaling is done inside a pipeline, so no test data leaks into training.
   - Class imbalance handled with `class_weight="balanced"`.
   - 3 models tuned with 3-fold cross-validation: Logistic Regression, Decision Tree, Random Forest.
   - Decision threshold chosen using cross-validated predictions on training data only.

## Results (test set, 5,993 customers)
At the default threshold of 0.5:

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.448 | 0.605 | 0.515 | 0.762 |
| Decision Tree | 0.390 | 0.670 | 0.493 | 0.762 |
| Random Forest | 0.487 | 0.599 | 0.537 | 0.777 |

Best model: **Random Forest**, threshold 0.58 (picked on training data):
- Precision 0.536, Recall 0.529, F1 0.532, ROC-AUC 0.777.
- It catches 701 of 1,326 real defaulters, and wrongly flags 606 good customers.

![ROC curves](roc_curves.png)
![Confusion matrix](confusion_matrix.png)
![Feature importance](feature_importance.png)

## What matters most
- Payment history dominates: `avg_delay`, `max_delay`, `late_months` and `PAY_1` (last month's status) are the top features.
- Credit limit, age and bill size matter much less.

## Feature engineering impact
- Logistic Regression ROC-AUC went from 0.719 to 0.761 after adding the new features.
- Random Forest barely changed (0.770 to 0.771), because trees already find these patterns themselves.

## Limitations
- Data is from one bank in Taiwan in 2005. It may not match other countries or years.
- The model misses about half of the real defaulters. It is a screening aid, not a final decision maker.
- Single train/test split and only 3 models. XGBoost and SMOTE are not tried.
- Fairness not checked: `SEX`, `MARRIAGE` and `AGE` are used as inputs. Real credit systems restrict this.

## How to run
```
pip install -r requirements.txt
python step2_cleaning.py
python step3_features.py
python step4_to_8_model.py
```
Put `default of credit card clients.xls` in the same folder first.
