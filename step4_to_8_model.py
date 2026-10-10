import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             roc_curve, confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)

SEED = 42

# ---------------------------------------------------------------------------
# STEP 4: Split first. Only then scale (scaling is inside the pipeline, so it
# learns from the training data only and no test data leaks in).
# ---------------------------------------------------------------------------
df = pd.read_csv("credit_features.csv")
X = df.drop(columns=["default"])
y = df["default"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Default rate  train: {y_train.mean():.3f}  test: {y_test.mean():.3f}")

# ---------------------------------------------------------------------------
# STEP 5: Handle imbalance with class_weight="balanced"
# (bad borrowers get more weight, so the model cannot ignore them)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# STEP 6: Train 3 models, tune with cross-validation
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

candidates = {
    "Logistic Regression": (
        make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=3000, class_weight="balanced")),
        {"logisticregression__C": [0.01, 0.1, 1, 10]},
    ),
    "Decision Tree": (
        DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
        {"max_depth": [3, 5, 8], "min_samples_leaf": [20, 50, 100]},
    ),
    "Random Forest": (
        RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
        {"n_estimators": [200], "max_depth": [8, 12], "min_samples_leaf": [10, 30]},
    ),
}

tuned = {}
for name, (model, grid) in candidates.items():
    search = GridSearchCV(model, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)
    tuned[name] = search.best_estimator_
    print(f"{name}: best CV AUC {search.best_score_:.4f}  settings {search.best_params_}")

# ---------------------------------------------------------------------------
# STEP 7: Evaluate on the untouched test set
# ---------------------------------------------------------------------------
rows = []
plt.figure(figsize=(6, 5))
for name, model in tuned.items():
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    rows.append({
        "Model": name,
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
    })
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC {rows[-1]['ROC-AUC']:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curves (test set)"); plt.legend(); plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150); plt.close()

results = pd.DataFrame(rows).set_index("Model").round(3)
print("\nResults at threshold 0.5:\n", results)
results.to_csv("model_results.csv")

best_name = results["ROC-AUC"].idxmax()
best = tuned[best_name]
print("\nBest model:", best_name)

# Threshold choice: use cross-validated predictions on TRAIN data only,
# so the test set stays clean.
train_proba = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
thresholds = np.linspace(0.2, 0.8, 61)
f1s = [f1_score(y_train, (train_proba >= t).astype(int)) for t in thresholds]
best_t = float(thresholds[int(np.argmax(f1s))])
print(f"Threshold picked on training data (best F1): {best_t:.2f}")

proba = best.predict_proba(X_test)[:, 1]
pred = (proba >= best_t).astype(int)
print(f"\nTest results for {best_name} at threshold {best_t:.2f}:")
print(classification_report(y_test, pred, target_names=["repaid", "defaulted"], digits=3))

cm = confusion_matrix(y_test, pred)
ConfusionMatrixDisplay(cm, display_labels=["repaid", "defaulted"]).plot(cmap="Blues")
plt.title(f"{best_name}: confusion matrix"); plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150); plt.close()

final = {
    "best_model": best_name,
    "threshold": round(best_t, 2),
    "precision": round(precision_score(y_test, pred), 3),
    "recall": round(recall_score(y_test, pred), 3),
    "f1": round(f1_score(y_test, pred), 3),
    "roc_auc": round(roc_auc_score(y_test, proba), 3),
    "confusion_matrix [[TN, FP],[FN, TP]]": cm.tolist(),
}
print(json.dumps(final, indent=2))
json.dump(final, open("final_metrics.json", "w"), indent=2)

# ---------------------------------------------------------------------------
# STEP 8: Explain the model (which features matter most)
# ---------------------------------------------------------------------------
rf = tuned["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values().tail(12)
plt.figure(figsize=(7, 5))
imp.plot(kind="barh", color="steelblue")
plt.title("Top 12 features (Random Forest)"); plt.xlabel("Importance")
plt.tight_layout(); plt.savefig("feature_importance.png", dpi=150); plt.close()
print("\nTop features:\n", imp.sort_values(ascending=False).round(3).head(8))

joblib.dump({"model": best, "threshold": best_t, "columns": list(X.columns)}, "credit_model.joblib")
