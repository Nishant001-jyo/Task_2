# Week 2 — Supervised Machine Learning Models (Scikit-learn)

Regression and classification pipelines that train several models, evaluate
them with standard metrics, and compare the results.

## Algorithms implemented
| Task | Models |
|------|--------|
| Regression (Diabetes dataset) | Linear Regression, Decision Tree, Random Forest, KNN |
| Classification (Breast Cancer dataset) | Logistic Regression, Decision Tree, Random Forest, KNN |

Both datasets ship with scikit-learn, so no download is needed.

## Project structure
```
supervised-ml-project/
├── main.py                 # runs everything
├── requirements.txt
├── src/
│   ├── regression.py       # regression pipeline
│   ├── classification.py   # classification pipeline
│   └── utils.py            # plotting helpers
└── results/                # generated CSVs, plots, report
```

## How to run
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```
Run one task only: `python -m src.regression` or `python -m src.classification`.

## Method
1. Load dataset, 80/20 train-test split (`random_state=42`; stratified for classification).
2. Scale features (StandardScaler inside a Pipeline) for Linear/Logistic Regression and KNN, which are scale-sensitive. Trees and forests need no scaling.
3. Train each model, predict on the held-out test set.
4. Evaluate:
   - Regression: MAE, MSE, RMSE, R², plus 5-fold cross-validated R²
   - Classification: Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix, plus 5-fold CV accuracy

## Results (from the run included in `results/`)

### Regression (higher R² / lower error is better)
| Model | MAE | RMSE | R² | CV R² |
|---|---|---|---|---|
| Linear Regression | 42.79 | 53.85 | 0.4526 | 0.4823 |
| Decision Tree | 47.05 | 59.74 | 0.3264 | 0.2859 |
| Random Forest | 43.64 | 53.83 | 0.4531 | 0.4336 |
| KNN | 45.11 | 56.29 | 0.4020 | 0.4292 |

### Classification
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV Acc |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9954 | 0.9807 |
| Decision Tree | 0.9386 | 0.9577 | 0.9444 | 0.9510 | 0.9342 | 0.9209 |
| Random Forest | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9931 | 0.9578 |
| KNN | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9788 | 0.9649 |

## Discussion
- **Regression:** Random Forest and Linear Regression are effectively tied on the test set (R² ≈ 0.45). Linear Regression has the better cross-validated R² and is far simpler, so it is the more reliable choice here. The single Decision Tree is clearly weakest; it overfits and has high variance. The Diabetes dataset is small and noisy, so an R² around 0.45 is typical.
- **Classification:** Logistic Regression is best on every test metric. The classes are close to linearly separable once features are scaled, so a simple linear model wins. Random Forest and KNN follow closely; the Decision Tree is weakest.
- **Recall matters most in medical screening** (a missed malignant case is costly). Treat "malignant" recall in the confusion matrices as the metric to watch, not just accuracy.
- **Caveat:** test sets are small (89 and 114 samples), so differences of 1-2 points can be noise. Cross-validation gives a steadier estimate.

## Ideas to extend
- Hyperparameter tuning with `GridSearchCV`
- Feature importance plots for Random Forest
- Try a different dataset (e.g., California Housing, Titanic)
