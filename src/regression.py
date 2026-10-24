"""Regression task: predict disease progression on the scikit-learn Diabetes dataset.

Models: Linear Regression, Decision Tree, Random Forest, KNN.
Metrics: MAE, MSE, RMSE, R^2 (+ 5-fold cross-validated R^2).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .utils import RESULTS_DIR, save_bar_chart

RANDOM_STATE = 42


def run():
    data = load_diabetes(as_frame=True)
    X, y = data.data, data.target
    print(f"[Regression] Dataset: Diabetes | samples={X.shape[0]}, features={X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Decision Tree": DecisionTreeRegressor(max_depth=4, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=6, random_state=RANDOM_STATE
        ),
        "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10)),
    }

    rows, preds = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        preds[name] = y_pred
        mse = mean_squared_error(y_test, y_pred)
        cv_r2 = cross_val_score(model, X, y, cv=5, scoring="r2").mean()
        rows.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R2": r2_score(y_test, y_pred),
            "CV R2 (5-fold)": cv_r2,
        })

    results = pd.DataFrame(rows).set_index("Model").round(4)
    results.to_csv(RESULTS_DIR / "regression_results.csv")
    print(results.to_string(), "\n")

    save_bar_chart(results, "R2", "Regression: R² on test set (higher is better)",
                   "regression_r2.png")
    save_bar_chart(results, "RMSE", "Regression: RMSE on test set (lower is better)",
                   "regression_rmse.png", higher_is_better=False)

    # Predicted vs actual for the best model
    best = results["R2"].idxmax()
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_test, preds[best], alpha=0.7)
    lims = [y.min(), y.max()]
    ax.plot(lims, lims, "r--", label="Perfect prediction")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Predicted vs Actual — {best}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "regression_pred_vs_actual.png", dpi=150)
    plt.close(fig)

    print(f"[Regression] Best model by test R²: {best}\n")
    return results


if __name__ == "__main__":
    run()
