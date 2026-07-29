# MultiModel Analysis - API Function & Code Reference Guide

This comprehensive reference document lists all functions, classes, and methods available in the `multimodel_analysis` package, complete with signatures, parameter descriptions, default values, and working, runnable Python code examples.

---

## Table of Contents

- [Overview & Package Imports](#overview--package-imports)
- [1. Standalone Utility Functions](#1-standalone-utility-functions)
  - [`save_report()`](#save_report)
- [2. MultiModelClassifier (Classification Pipeline)](#2-multimodelclassifier-classification-pipeline)
  - [`__init__()`](#multimodelclassifier__init__)
  - [`Logistic_model()`](#multimodelclassifierlogistic_model)
  - [`Support_vector_model()`](#multimodelclassifiersupport_vector_model)
  - [`DecisionTree_model()`](#multimodelclassifierdecisiontree_model)
  - [`KNN_model()`](#multimodelclassifierknn_model)
  - [`Naive_Bayes_model()`](#multimodelclassifiernaive_bayes_model)
  - [`RandomForest_model()`](#multimodelclassifierrandomforest_model)
  - [`GradientBoosting_model()`](#multimodelclassifiergradientboosting_model)
  - [`AdaBoost_model()`](#multimodelclassifieradaboost_model)
  - [`run_all_models()`](#multimodelclassifierrun_all_models)
  - [`evaluate_model()`](#multimodelclassifierevaluate_model)
  - [`show_tabular_report()`](#multimodelclassifiershow_tabular_report)
  - [`plot_confusion_matrices()`](#multimodelclassifierplot_confusion_matrices)
  - [`plot_roc_curves()`](#multimodelclassifierplot_roc_curves)
  - [`plot_comparison()`](#multimodelclassifierplot_comparison)
  - [`get_summary()`](#multimodelclassifierget_summary)
  - [`save_report()` (Instance Method)](#multimodelclassifiersave_report)
- [3. MultiModelRegressor (Regression Pipeline)](#3-multimodelregressor-regression-pipeline)
  - [`__init__()`](#multimodelregressor__init__)
  - [`LinearRegression_model()`](#multimodelregressorlinearregression_model)
  - [`Lasso_model()`](#multimodelregressorlasso_model)
  - [`Ridge_model()`](#multimodelregressorridge_model)
  - [`SVR_model()`](#multimodelregressorsvr_model)
  - [`DecisionTree_model()`](#multimodelregressordecisiontree_model)
  - [`RandomForest_model()`](#multimodelregressorrandomforest_model)
  - [`GradientBoosting_model()`](#multimodelregressorgradientboosting_model)
  - [`AdaBoost_model()`](#multimodelregressoradaboost_model)
  - [`run_all_models()`](#multimodelregressorrun_all_models)
  - [`evaluate_model()`](#multimodelregressorevaluate_model)
  - [`show_tabular_report()`](#multimodelregressorshow_tabular_report)
  - [`plot_true_vs_predicted()`](#multimodelregressorplot_true_vs_predicted)
  - [`plot_comparison()`](#multimodelregressorplot_comparison)
  - [`get_summary()`](#multimodelregressorget_summary)
  - [`save_report()` (Instance Method)](#multimodelregressorsave_report)
- [4. Complete End-to-End Examples](#4-complete-end-to-end-examples)
  - [Full Classification Example](#full-classification-example)
  - [Full Regression Example](#full-regression-example)

---

## Overview & Package Imports

Import all classes and functions from the package:

```python
from multimodel_analysis import (
    MultiModelClassifier,
    MultiModelRegressor,
    MultiModelRegressior,  # Alias for MultiModelRegressor
    save_report
)
```

---

## 1. Standalone Utility Functions

### `save_report()`

#### Description
Saves a comparison tabular report `pandas.DataFrame` to disk. Automatically detects output format (`.csv`, `.xlsx`, `.xls`, `.html`, `.json`) from the file extension.

#### Signature
```python
save_report(df: Optional[pd.DataFrame] = None, filepath: str = "report.csv") -> None
```

#### Parameters
| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | `pandas.DataFrame` | `None` | The DataFrame returned by `show_tabular_report(return_df=True)`. |
| `filepath` | `str` | `"report.csv"` | Target file path (`.csv`, `.xlsx`, `.html`, or `.json`). |

#### Example
```python
import pandas as pd
from multimodel_analysis import save_report

# Sample metrics report DataFrame
df_report = pd.DataFrame({
    'Model': ['Random Forest', 'Logistic Regression'],
    'Accuracy': [0.95, 0.88],
    'F1 Score': [0.94, 0.87]
})

# Save to CSV
save_report(df_report, "metrics.csv")

# Save to Excel
save_report(df_report, "metrics.xlsx")

# Save to HTML
save_report(df_report, "metrics.html")

# Save to JSON
save_report(df_report, "metrics.json")
```

---

## 2. MultiModelClassifier (Classification Pipeline)

`MultiModelClassifier` automates data splitting, standard feature scaling, target label encoding, model training, metric evaluation, tabular reporting, and figure generation for classification tasks.

---

### `MultiModelClassifier.__init__()`

#### Description
Initializes the classification pipeline, performs label encoding on target `y`, and executes stratified train-test splitting and feature scaling.

#### Signature
```python
MultiModelClassifier(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, pd.DataFrame, np.ndarray],
    test_size: float = 0.3,
    scaled_data: bool = False,
    random_state: Optional[int] = 42,
    stratify: bool = True,
    n_jobs: Optional[int] = -1
)
```

#### Parameters
| Parameter | Type | Default | Description |
|---|---|---|---|
| `X` | `DataFrame` or `ndarray` | *Required* | Feature matrix. |
| `y` | `Series`, `DataFrame`, or `ndarray` | *Required* | Target vector (categorical, string, or integer). |
| `test_size` | `float` | `0.3` | Proportion of data to include in test split (0.0 to 1.0). |
| `scaled_data` | `bool` | `False` | If `True`, applies `StandardScaler` to feature matrix `X`. |
| `random_state` | `int` or `None` | `42` | Seed used by random number generator for reproducible splitting. |
| `stratify` | `bool` | `True` | If `True`, performs stratified split preserving class distributions. |
| `n_jobs` | `int` or `None` | `-1` | Number of CPU cores to use for parallel algorithms (`-1` = all cores). |

#### Example
```python
from sklearn.datasets import make_classification
from multimodel_analysis import MultiModelClassifier

X, y = make_classification(n_samples=200, n_features=6, random_state=42)

classifier = MultiModelClassifier(
    X=X,
    y=y,
    test_size=0.25,
    scaled_data=True,
    random_state=42,
    stratify=True,
    n_jobs=-1
)
```

---

### `MultiModelClassifier.Logistic_model()`

#### Description
Trains a Logistic Regression classifier and evaluates performance metrics.

#### Signature
```python
Logistic_model(random_state: Optional[int] = None, max_iter: int = 1000, **kwargs)
```

#### Example
```python
# Uses default random_state (instance default 42)
metrics = classifier.Logistic_model()

# Overrides random_state and passes custom hyperparameter max_iter
custom_metrics = classifier.Logistic_model(random_state=100, max_iter=2000, C=0.5)
```

---

### `MultiModelClassifier.Support_vector_model()`

#### Description
Trains a Support Vector Classifier (`SVC`) with probability estimation enabled.

#### Signature
```python
Support_vector_model(random_state: Optional[int] = None, kernel: str = 'linear', probability: bool = True, **kwargs)
```

#### Example
```python
# Default linear kernel
svc_res = classifier.Support_vector_model()

# Custom RBF kernel with specified random_state
svc_custom = classifier.Support_vector_model(random_state=99, kernel='rbf', C=1.5)
```

---

### `MultiModelClassifier.DecisionTree_model()`

#### Description
Trains a Decision Tree Classifier.

#### Signature
```python
DecisionTree_model(random_state: Optional[int] = None, **kwargs)
```

#### Example
```python
dt_res = classifier.DecisionTree_model(random_state=123, max_depth=5)
```

---

### `MultiModelClassifier.KNN_model()`

#### Description
Trains a K-Nearest Neighbors Classifier. Automatically calculates optimal `n_neighbors` if not provided. Strips `random_state` safely if passed.

#### Signature
```python
KNN_model(n_neighbors: Optional[int] = None, **kwargs)
```

#### Example
```python
# Auto-detects n_neighbors based on train size
knn_res = classifier.KNN_model()

# Custom n_neighbors
knn_custom = classifier.KNN_model(n_neighbors=7)
```

---

### `MultiModelClassifier.Naive_Bayes_model()`

#### Description
Trains a Gaussian Naive Bayes Classifier. Strips `random_state` safely if passed.

#### Signature
```python
Naive_Bayes_model(**kwargs)
```

#### Example
```python
nb_res = classifier.Naive_Bayes_model()
```

---

### `MultiModelClassifier.RandomForest_model()`

#### Description
Trains a Random Forest Classifier.

#### Signature
```python
RandomForest_model(n_estimators: int = 100, random_state: Optional[int] = None, **kwargs)
```

#### Example
```python
rf_res = classifier.RandomForest_model(n_estimators=150, random_state=42, max_depth=10)
```

---

### `MultiModelClassifier.GradientBoosting_model()`

#### Description
Trains a Gradient Boosting Classifier.

#### Signature
```python
GradientBoosting_model(n_estimators: int = 100, random_state: Optional[int] = None, **kwargs)
```

#### Example
```python
gb_res = classifier.GradientBoosting_model(n_estimators=120, random_state=42, learning_rate=0.05)
```

---

### `MultiModelClassifier.AdaBoost_model()`

#### Description
Trains an AdaBoost Classifier.

#### Signature
```python
AdaBoost_model(n_estimators: int = 50, random_state: Optional[int] = None, **kwargs)
```

#### Example
```python
ada_res = classifier.AdaBoost_model(n_estimators=80, random_state=42)
```

---

### `MultiModelClassifier.run_all_models()`

#### Description
Executes training and evaluation for all built-in classification baseline models, plus any optional user-supplied custom estimators. Caches evaluated model results in `classifier.models_`.

#### Signature
```python
run_all_models(custom_models: Optional[Dict[str, Any]] = None, random_state: Optional[int] = None) -> List[Tuple]
```

#### Parameters
| Parameter | Type | Default | Description |
|---|---|---|---|
| `custom_models` | `Dict[str, estimator]` or `None` | `None` | Dictionary of custom scikit-learn estimators `{'Model Name': model_instance}`. |
| `random_state` | `int` or `None` | `None` | Random seed passed to all baseline models. Defaults to instance `random_state`. |

#### Example
```python
from sklearn.ensemble import ExtraTreesClassifier

# 1. Run standard baselines
models = classifier.run_all_models()

# 2. Run with custom random_state and custom models
custom_clf = ExtraTreesClassifier(n_estimators=50, random_state=42)
models = classifier.run_all_models(
    custom_models={'Extra Trees': custom_clf},
    random_state=123
)
```

---

### `MultiModelClassifier.evaluate_model()`

#### Description
Evaluates a single fitted classifier model on test set data. Computes Classification Report, Confusion Matrix, Accuracy, Precision, Recall, F1 Score, and ROC-AUC metrics.

#### Signature
```python
evaluate_model(model: Any, X_test: Optional[Any] = None, y_true: Optional[np.ndarray] = None)
```

#### Example
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)
rf.fit(classifier.X_train_scaled, classifier.y_train)

# Evaluate using default instance test set
eval_res = classifier.evaluate_model(rf)
report, matrix, accuracy, precision, recall, f1, fpr_dict, tpr_dict, roc_auc = eval_res
print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
```

---

### `MultiModelClassifier.show_tabular_report()`

#### Description
Prints a formatted comparison table of all model evaluation metrics sorted by accuracy and identifies the best performing model. Returns a `pandas.DataFrame`.

#### Signature
```python
show_tabular_report(models: Optional[List[Tuple]] = None, return_df: bool = False) -> Optional[pd.DataFrame]
```

#### Example
```python
# Automatically runs all models if models argument is omitted
df_report = classifier.show_tabular_report(return_df=True)
```

---

### `MultiModelClassifier.plot_confusion_matrices()`

#### Description
Generates a grid heatmap of confusion matrices for all evaluated models.

#### Signature
```python
plot_confusion_matrices(models: Optional[List[Tuple]] = None, save_path: Optional[str] = None, show_plot: bool = True)
```

#### Example
```python
# Display plot on screen
classifier.plot_confusion_matrices()

# Save plot to PNG without displaying
classifier.plot_confusion_matrices(save_path="confusion_matrices.png", show_plot=False)
```

---

### `MultiModelClassifier.plot_roc_curves()`

#### Description
Plots Receiver Operating Characteristic (ROC) curves for all models in a single comparative chart. Supports binary and multiclass (macro-averaged) tasks.

#### Signature
```python
plot_roc_curves(models: Optional[List[Tuple]] = None, save_path: Optional[str] = None, show_plot: bool = True)
```

#### Example
```python
classifier.plot_roc_curves(save_path="roc_curves.png", show_plot=True)
```

---

### `MultiModelClassifier.plot_comparison()`

#### Description
Renders a grouped bar chart comparing Accuracy, Precision, Recall, and F1 Score across all models.

#### Signature
```python
plot_comparison(models: Optional[List[Tuple]] = None, save_path: Optional[str] = None, show_plot: bool = True)
```

#### Example
```python
classifier.plot_comparison(save_path="classifier_comparison.png")
```

---

### `MultiModelClassifier.get_summary()`

#### Description
Runs the full evaluation and visualization pipeline: prints the tabular report and generates confusion matrices, ROC curves, and metric comparison bar charts.

#### Signature
```python
get_summary(models: Optional[List[Tuple]] = None, save_prefix: Optional[str] = None, show_plot: bool = True)
```

#### Example
```python
# Display all reports and plots
classifier.get_summary()

# Export all reports and plots to files with a prefix
classifier.get_summary(save_prefix="iris_experiment", show_plot=False)
# Generates:
# - iris_experiment_report.csv
# - iris_experiment_confusion_matrices.png
# - iris_experiment_roc_curves.png
# - iris_experiment_comparison.png
```

---

### `MultiModelClassifier.save_report()` (Instance Method)

#### Description
Saves the classifier's latest tabular performance report to disk.

#### Signature
```python
save_report(df_or_filepath: Optional[Union[pd.DataFrame, str]] = None, filepath: Optional[str] = None) -> None
```

#### Example
```python
classifier.show_tabular_report()

# Save report using default path 'report.csv'
classifier.save_report()

# Save report to specific filename
classifier.save_report("classifier_results.xlsx")
```

---

## 3. MultiModelRegressor (Regression Pipeline)

`MultiModelRegressor` automates data splitting, feature scaling, model fitting, error metric calculation (MAE, MSE, RMSE, R2 Score), tabular reporting, and figure generation for regression tasks.

> **Note**: `MultiModelRegressior` is available as a backwards-compatibility alias for `MultiModelRegressor`.

---

### `MultiModelRegressor.__init__()`

#### Description
Initializes the regression pipeline, performs train-test splitting, and applies feature scaling.

#### Signature
```python
MultiModelRegressor(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, pd.DataFrame, np.ndarray],
    test_size: float = 0.3,
    scaled_data: bool = False,
    random_state: Optional[int] = 42,
    n_jobs: Optional[int] = -1
)
```

#### Parameters
| Parameter | Type | Default | Description |
|---|---|---|---|
| `X` | `DataFrame` or `ndarray` | *Required* | Feature matrix. |
| `y` | `Series`, `DataFrame`, or `ndarray` | *Required* | Target continuous vector/values. |
| `test_size` | `float` | `0.3` | Proportion of dataset to include in test split. |
| `scaled_data` | `bool` | `False` | If `True`, applies `StandardScaler` to feature matrix `X`. |
| `random_state` | `int` or `None` | `42` | Seed used for reproducible train-test splitting. |
| `n_jobs` | `int` or `None` | `-1` | Number of CPU cores to use for parallel algorithms. |

#### Example
```python
from sklearn.datasets import make_regression
from multimodel_analysis import MultiModelRegressor

X, y = make_regression(n_samples=200, n_features=8, noise=0.1, random_state=42)

regressor = MultiModelRegressor(
    X=X,
    y=y,
    test_size=0.2,
    scaled_data=True,
    random_state=42
)
```

---

### `MultiModelRegressor.LinearRegression_model()`

#### Description
Trains an Ordinary Least Squares Linear Regression model. Strips `random_state` safely if passed.

#### Signature
```python
LinearRegression_model(**kwargs)
```

#### Example
```python
lin_res = regressor.LinearRegression_model()
```

---

### `MultiModelRegressor.Lasso_model()`

#### Description
Trains a Lasso (L1 Regularized) Regression model.

#### Signature
```python
Lasso_model(alpha: float = 0.1, random_state: Optional[int] = None, **kwargs)
```

#### Example
```python
lasso_res = regressor.Lasso_model(alpha=0.05, random_state=42)
```

---

### `MultiModelRegressor.Ridge_model()`

#### Description
Trains a Ridge (L2 Regularized) Regression model.

#### Signature
```python
Ridge_model(alpha: float = 1.0, random_state: Optional[int] = None, **kwargs)
```

#### Example
```python
ridge_res = regressor.Ridge_model(alpha=0.5, random_state=42)
```

---

### `MultiModelRegressor.SVR_model()`

#### Description
Trains a Support Vector Regressor (`SVR`). Strips `random_state` safely if passed.

#### Signature
```python
SVR_model(kernel: str = 'rbf', **kwargs)
```

#### Example
```python
svr_res = regressor.SVR_model(kernel='rbf', C=2.0)
```

---

### `MultiModelRegressor.DecisionTree_model()`

#### Description
Trains a Decision Tree Regressor.

#### Signature
```python
DecisionTree_model(random_state: Optional[int] = None, **kwargs)
```

#### Example
```python
dt_reg = regressor.DecisionTree_model(random_state=42, max_depth=6)
```

---

### `MultiModelRegressor.RandomForest_model()`

#### Description
Trains a Random Forest Regressor.

#### Signature
```python
RandomForest_model(n_estimators: int = 100, random_state: Optional[int] = None, **kwargs)
```

#### Example
```python
rf_reg = regressor.RandomForest_model(n_estimators=120, random_state=42)
```

---

### `MultiModelRegressor.GradientBoosting_model()`

#### Description
Trains a Gradient Boosting Regressor.

#### Signature
```python
GradientBoosting_model(n_estimators: int = 100, random_state: Optional[int] = None, **kwargs)
```

#### Example
```python
gb_reg = regressor.GradientBoosting_model(n_estimators=100, random_state=42, learning_rate=0.08)
```

---

### `MultiModelRegressor.AdaBoost_model()`

#### Description
Trains an AdaBoost Regressor.

#### Signature
```python
AdaBoost_model(n_estimators: int = 50, random_state: Optional[int] = None, **kwargs)
```

#### Example
```python
ada_reg = regressor.AdaBoost_model(n_estimators=60, random_state=42)
```

---

### `MultiModelRegressor.run_all_models()`

#### Description
Executes training and evaluation for all built-in regression baseline models, plus any optional user-supplied custom estimators. Caches evaluated model results in `regressor.models_`.

#### Signature
```python
run_all_models(custom_models: Optional[Dict[str, Any]] = None, random_state: Optional[int] = None) -> List[Tuple]
```

#### Parameters
| Parameter | Type | Default | Description |
|---|---|---|---|
| `custom_models` | `Dict[str, estimator]` or `None` | `None` | Dictionary of custom scikit-learn regressor estimators. |
| `random_state` | `int` or `None` | `None` | Random seed passed to all baseline models. |

#### Example
```python
from sklearn.ensemble import ExtraTreesRegressor

models = regressor.run_all_models(
    custom_models={'Extra Trees Regressor': ExtraTreesRegressor(random_state=42)},
    random_state=99
)
```

---

### `MultiModelRegressor.evaluate_model()`

#### Description
Evaluates a single fitted regressor model on test set data. Computes MAE, MSE, RMSE, and R2 Score metrics.

#### Signature
```python
evaluate_model(model: Any, X_test: Optional[Any] = None, y_true: Optional[np.ndarray] = None)
```

#### Example
```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=42)
rf.fit(regressor.X_train_scaled, regressor.y_train)

mae, mse, rmse, r2, y_pred = regressor.evaluate_model(rf)
print(f"MAE: {mae:.4f}, R2 Score: {r2:.4f}")
```

---

### `MultiModelRegressor.show_tabular_report()`

#### Description
Prints a formatted comparison table of all regression model evaluation metrics sorted by R2 Score and identifies the best model. Returns a `pandas.DataFrame`.

#### Signature
```python
show_tabular_report(models: Optional[List[Tuple]] = None, return_df: bool = False) -> Optional[pd.DataFrame]
```

#### Example
```python
df_reg_report = regressor.show_tabular_report(return_df=True)
```

---

### `MultiModelRegressor.plot_true_vs_predicted()`

#### Description
Generates a grid scatter plot of True vs. Predicted target values with identity ($y = x$) reference lines for all models.

#### Signature
```python
plot_true_vs_predicted(models: Optional[List[Tuple]] = None, save_path: Optional[str] = None, show_plot: bool = True)
```

#### Example
```python
regressor.plot_true_vs_predicted(save_path="true_vs_pred.png", show_plot=True)
```

---

### `MultiModelRegressor.plot_comparison()`

#### Description
Renders a bar chart comparing R2 Scores across all regression models.

#### Signature
```python
plot_comparison(models: Optional[List[Tuple]] = None, save_path: Optional[str] = None, show_plot: bool = True)
```

#### Example
```python
regressor.plot_comparison(save_path="r2_comparison.png")
```

---

### `MultiModelRegressor.get_summary()`

#### Description
Runs the full evaluation and visualization pipeline: prints the tabular report and generates True vs. Predicted scatter plots and R2 Score comparison bar charts.

#### Signature
```python
get_summary(models: Optional[List[Tuple]] = None, save_prefix: Optional[str] = None, show_plot: bool = True)
```

#### Example
```python
# Display summary
regressor.get_summary()

# Export summary files
regressor.get_summary(save_prefix="housing_exp", show_plot=False)
# Generates:
# - housing_exp_report.csv
# - housing_exp_true_vs_pred.png
# - housing_exp_comparison.png
```

---

### `MultiModelRegressor.save_report()` (Instance Method)

#### Description
Saves the regressor's latest tabular performance report to disk.

#### Signature
```python
save_report(df_or_filepath: Optional[Union[pd.DataFrame, str]] = None, filepath: Optional[str] = None) -> None
```

#### Example
```python
regressor.show_tabular_report()

# Save report using default path 'report.csv'
regressor.save_report()

# Save report to specific filename
regressor.save_report("regression_metrics.xlsx")
```

---

## 4. Complete End-to-End Examples

### Full Classification Example

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from multimodel_analysis import MultiModelClassifier, save_report

# 1. Load data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# 2. Instantiate pipeline
clf = MultiModelClassifier(
    X=X,
    y=y,
    test_size=0.2,
    scaled_data=True,
    random_state=42,
    stratify=True
)

# 3. Run model benchmarking
results = clf.run_all_models(random_state=42)

# 4. Display tabular report and get DataFrame
df_report = clf.show_tabular_report(return_df=True)

# 5. Export results
clf.save_report("breast_cancer_metrics.csv")

# 6. Render summary visualizations
clf.get_summary(save_prefix="breast_cancer", show_plot=False)
```

---

### Full Regression Example

```python
import pandas as pd
from sklearn.datasets import fetch_california_housing
from multimodel_analysis import MultiModelRegressor

# 1. Load data
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names).iloc[:500]  # Subset for speed
y = data.target[:500]

# 2. Instantiate pipeline
reg = MultiModelRegressor(
    X=X,
    y=y,
    test_size=0.2,
    scaled_data=True,
    random_state=42
)

# 3. Run model benchmarking
results = reg.run_all_models(random_state=42)

# 4. Display tabular report
reg.show_tabular_report()

# 5. Export summary reports & plots in one line
reg.get_summary(save_prefix="california_housing", show_plot=False)
```
