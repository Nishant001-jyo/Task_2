"""Classification task: Breast Cancer Wisconsin dataset (malignant vs benign).

Models: Logistic Regression, Decision Tree, Random Forest, KNN.
Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, ConfusionMatrixDisplay,
                             classification_report)

from .utils import RESULTS_DIR, save_bar_chart

RANDOM_STATE = 42


def run():
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    print(f"[Classification] Dataset: Breast Cancer | samples={X.shape[0]}, "
          f"features={X.shape[1]}, classes={[str(c) for c in data.target_names]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    }

    rows, fitted = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_prob),
            "CV Accuracy (5-fold)": cross_val_score(model, X, y, cv=5).mean(),
        })

    results = pd.DataFrame(rows).set_index("Model").round(4)
    results.to_csv(RESULTS_DIR / "classification_results.csv")
    print(results.to_string(), "\n")

    save_bar_chart(results, "F1", "Classification: F1 score on test set",
                   "classification_f1.png")
    save_bar_chart(results, "Accuracy", "Classification: Accuracy on test set",
                   "classification_accuracy.png")

    # Confusion matrices for every model
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, (name, model) in zip(axes.ravel(), fitted.items()):
        ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test, display_labels=data.target_names,
            cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "classification_confusion_matrices.png", dpi=150)
    plt.close(fig)

    best = results["F1"].idxmax()
    report = classification_report(y_test, fitted[best].predict(X_test),
                                   target_names=data.target_names)
    (RESULTS_DIR / "best_classifier_report.txt").write_text(
        f"Best model by F1: {best}\n\n{report}")
    print(f"[Classification] Best model by test F1: {best}\n{report}")
    return results


if __name__ == "__main__":
    run()
