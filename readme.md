<!-- # Multi-Model Analysis

This Python package is created by **Uditya Narayan Tiwari**. It provides a robust framework to automatically train, evaluate, and visualize multiple machine learning models for both **Classification** and **Regression** problems in a single step.

The package is designed to save time by automating the model selection process, featuring automatic data scaling, comprehensive metric reporting, and professional visualization of results.

## Installation

### Installation from PyPi
You can install this package using pip as follows:
```bash
pip install multimodel_analysis

```

### Installation from GitHub

You can install the latest version directly from GitHub:

```bash
pip install git+[https://github.com/udityamerit/multimodel_analysis.git](https://github.com/udityamerit/multimodel_analysis.git) --upgrade --force-reinstall

```

### Uninstall the Package

To uninstall the package, use the following command:

```bash
pip uninstall multimodel_analysis

```

### Requirements

This package requires the following libraries:

* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `scikit-learn`

---

## How to Use the Package

This package simplifies the workflow into three main steps: **Initialize**, **Run**, and **Visualize**.

### 1. Classification Problems

Use the `MultiModelClassifier` to automatically compare models like Logistic Regression, SVM, Decision Trees, KNN, Naive Bayes, Random Forest, Gradient Boosting, and AdaBoost.

```python
import pandas as pd
from multimodel_analysis import MultiModelClassifier

# 1. Load your dataset
# Ensure your data is numeric or pre-encoded
df = pd.read_csv('your_classification_dataset.csv')
X = df.drop('target_column', axis=1)
y = df['target_column']

# 2. Initialize the Classifier
# scaled_data=True automatically applies StandardScaler to your features
classifier = MultiModelClassifier(X, y, scaled_data=True)

# 3. Run all models
# This returns a list of results for all trained models
results = classifier.run_all_models()

# 4. Print Summary and Visualizations
# Shows Accuracy, Precision, Recall, F1, AUC, Confusion Matrices, and ROC Curves
classifier.get_summary(results)

# 5. Compare Models visually
# Plots a bar chart comparing all metrics across models
classifier.plot_comparison(results)

```

### 2. Regression Problems

Use the `MultiModelRegressior` to compare models like Linear Regression, Lasso, Ridge, SVR, Decision Trees, Random Forest, and Gradient Boosting.

```python
import pandas as pd
from multimodel_analysis import MultiModelClassifier

# 1. Load your dataset
# Ensure your data is numeric or pre-encoded
df = pd.read_csv('https://raw.githubusercontent.com/udityamerit/MultiModel-Package-for-Machine-Learning/refs/heads/main/Dataset/diabetes.csv')
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 2. Initialize the Regressor
# scaled_data=True automatically applies StandardScaler to your features
classifier = MultiModelClassifier(X, y, scaled_data=True)

# 3. Run all models
results = classifier.run_all_models()

# 4. Print Summary
# Shows MAE, MSE, RMSE, and R2 Scores for all models
classifier.get_summary(results)

# 5. Compare Models visually
# Plots the R2 Score comparison
classifier.plot_comparison(results)

```

---

## Features

* **Automatic Scaling**: Includes a `scaled_data` flag to automatically handle feature standardization (StandardScaler) to ensure optimal performance for models like SVM and KNN.
* **Comprehensive Metrics**:
* **Classification**: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
* **Regression**: MAE (Mean Absolute Error), MSE (Mean Squared Error), RMSE (Root Mean Squared Error), and R2 Score.


* **Visualizations**: Generates professional Confusion Matrices, ROC Curves, and Comparative Bar Plots using `seaborn` and `matplotlib`.
* **Model Variety**: Covers a wide range of algorithms from simple Linear/Logistic models to Ensemble methods like Random Forest and Gradient Boosting.

## Author

**Uditya Narayan Tiwari**

## License

This project is licensed under the MIT License - see the LICENSE file for details. -->


# Multi-Model Analysis

This Python package is created by **Uditya Narayan Tiwari**. It provides a robust framework to automatically train, evaluate, and visualize multiple machine learning models for both **Classification** and **Regression** problems.

It includes features for **automatic best model recommendation**, colorful confusion matrices, and comprehensive performance plots.

## Installation

### Installation from PyPi
```bash
pip install multimodel_analysis

```

### Installation from GitHub

```bash
pip install git+[https://github.com/udityamerit/multimodel_analysis.git](https://github.com/udityamerit/multimodel_analysis.git) --upgrade --force-reinstall

```

### Requirements

* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `scikit-learn`

---

## How to Use the Package

### 1. Classification Problems

Use the `MultiModelClassifier` to compare models like Logistic Regression, SVM, Decision Trees, Random Forest, etc.

```python
import pandas as pd
from multimodel_analysis import MultiModelClassifier

# 1. Load Data
df = pd.read_csv('classification_data.csv')
X = df.drop('target', axis=1)
y = df['target']

# 2. Initialize & Run
# scaled_data=True automatically handles feature scaling
classifier = MultiModelClassifier(X, y, scaled_data=True)
results = classifier.run_all_models()

# 3. Get Tabular Report & Best Model Recommendation
# Prints a clean table of Accuracy, Precision, Recall, F1, and AUC
# Also prints the "⭐ BEST MODEL RECOMMENDATION"
classifier.show_tabular_report(results)

# 4. Visualize Confusion Matrices
# Plots matrices for all models with different color palettes
classifier.plot_confusion_matrices(results)

# 5. Plot ROC Curves
# Overlays ROC curves for all models in a single plot
classifier.plot_roc_curves(results)

# 6. Compare Metrics
# Plots a bar chart comparing Accuracy, Precision, Recall, and F1 across models
classifier.plot_comparison(results)

```

### 2. Regression Problems

Use the `MultiModelRegressior` to compare models like Linear Regression, Lasso, Ridge, SVR, Random Forest, etc.

```python
import pandas as pd
from multimodel_analysis import MultiModelRegressior

# 1. Load Data
df = pd.read_csv('regression_data.csv')
X = df.drop('target', axis=1)
y = df['target']

# 2. Initialize & Run
regressor = MultiModelRegressior(X, y, scaled_data=True)
results = regressor.run_all_models()

# 3. Get Tabular Report & Best Model Recommendation
# Prints a clean table of MAE, MSE, RMSE, and R2 Score
# Also prints the "⭐ BEST MODEL RECOMMENDATION"
regressor.show_tabular_report(results)

# 4. Visualize True vs. Predicted Values
# Plots scatter plots with ideal fit lines for every model
regressor.plot_true_vs_predicted(results)

```

```python
# 5. Compare R2 Scores
# Plots a bar chart comparing R2 scores across all models
regressor.plot_comparison(results)

```

---

## Features

### Classification Features

* **`show_tabular_report()`**: Displays a sorted dataframe of metrics and prints the best model based on Accuracy.
* **`plot_confusion_matrices()`**: Generates a grid of confusion matrices, cycling through colors (Blues, Greens, Oranges, Purples, etc.) for distinction.
* **`plot_roc_curves()`**: Plots ROC curves for all models on a single graph to compare AUC performance.

### Regression Features

* **`show_tabular_report()`**: Displays a sorted dataframe of errors/scores and prints the best model based on R2 Score.
* **`plot_true_vs_predicted()`**: Generates scatter plots showing how close predicted values are to actual values.

### General Features

* **Automatic Scaling**: Handles `StandardScaler` internally if `scaled_data=True` is passed.
* **Model Variety**: Includes Linear models, SVMs, Decision Trees, Nearest Neighbors, and Ensemble methods (Random Forest, Gradient Boosting, AdaBoost).

## Author

**Uditya Narayan Tiwari**


## License

This project is licensed under the MIT License.

```


```
