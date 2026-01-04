# Multi-Model Analysis

[![PyPI version](https://img.shields.io/pypi/v/multimodel_analysis.svg)](https://pypi.org/project/multimodel_analysis/)
[![License: Apache](https://img.shields.io/badge/License-Apache-yellow.svg)](https://opensource.org/licenses/MIT)

**multimodel_analysis** is a comprehensive machine learning library created by **Uditya Narayan Tiwari**. It automates the process of training, evaluating, and visualizing multiple models for both Classification and Regression tasks.

Designed for data scientists and researchers, it streamlines model selection by providing automatic feature scaling, detailed performance metrics, and professional-grade visualizations in just a few lines of code.

---

## 📦 Installation

### Install via PyPI
The easiest way to install the package is via pip:
```bash
pip install multimodel_analysis

```

### Install from GitHub

To get the latest development version:

```bash
pip install git+[https://github.com/udityamerit/multimodel_analysis.git](https://github.com/udityamerit/Multimodel-Analysis-Pacakge.git) --upgrade --force-reinstall

```

### 📋 Requirements

* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `scikit-learn`

---

##  How to Use

### 1. Classification Analysis

Use `MultiModelClassifier` for categorical target variables. It automatically compares models like Logistic Regression, SVM, Decision Trees, Random Forest, Gradient Boosting, and more.

#### Step 1: Initialize and Train

```python
import pandas as pd
from multimodel_analysis import MultiModelClassifier

# Load your dataset
df = pd.read_csv('your_classification_data.csv')
X = df.drop('target_column', axis=1)
y = df['target_column']

# Initialize the classifier
# scaled_data=True applies StandardScaler automatically
classifier = MultiModelClassifier(X, y, test_size=0.3, scaled_data=True)

# Train all models and get results
results = classifier.run_all_models()

```

#### Step 2: View Metrics & Best Model

Generates a clean dataframe of metrics (Accuracy, Precision, Recall, F1, AUC) and prints the best recommendation.

```python
classifier.show_tabular_report(results)

```

#### Step 3: Visualize Confusion Matrices

Plots confusion matrices for every trained model with distinct color palettes.

```python
classifier.plot_confusion_matrices(results)

```

#### Step 4: Plot ROC Curves

Overlays ROC curves for all models to compare Area Under the Curve (AUC) performance.

```python
classifier.plot_roc_curves(results)

```

#### Step 5: Compare All Metrics

Plots a grouped bar chart comparing Accuracy, Precision, Recall, and F1 Score.

```python
classifier.plot_comparison(results)

```

---

### 2. Regression Analysis

Use `MultiModelRegressior` for continuous target variables. It compares Linear Regression, Lasso, Ridge, SVR, Random Forest, Gradient Boosting, etc.

#### Step 1: Initialize and Train

```python
from multimodel_analysis import MultiModelRegressior

# Load your dataset
df = pd.read_csv('your_regression_data.csv')
X = df.drop('price', axis=1)
y = df['price']

# Initialize the regressor
regressor = MultiModelRegressior(X, y, test_size=0.3, scaled_data=True)

# Train all models
results = regressor.run_all_models()

```

#### Step 2: View Metrics & Best Model

Generates a table of MAE, MSE, RMSE, and R2 Scores, and recommends the best model based on R2.

```python
regressor.show_tabular_report(results)

```

#### Step 3: True vs. Predicted Plots

Visualizes the relationship between actual and predicted values with an ideal fit line.

```python
regressor.plot_true_vs_predicted(results)

```

#### Step 4: Compare R2 Scores

Plots a bar chart to easily identify the model with the highest R2 score.

```python
regressor.plot_comparison(results)

```

---

## Key Features

### 🔹 Intelligent Automation

* **Automatic Scaling**: Simply set `scaled_data=True` to standardize your features using `StandardScaler` before training.
* **Best Model Detection**: Automatically highlights the best performing model based on Accuracy (Classification) or R2 Score (Regression).

### 🔹 Professional Visualizations

* **Colorful Confusion Matrices**: Automatically cycles through color maps (Blues, Greens, Oranges, etc.) for distinct model visualization.
* **ROC Curve Overlays**: clean comparison of True Positive vs False Positive rates.
* **Regression Fit Plots**: Scatter plots with diagonal reference lines to visually assess regression performance.

### 🔹 Extensive Model Library

* **Classifiers**: Logistic Regression, SVM, Decision Tree, KNN, Naive Bayes, Random Forest, Gradient Boosting, AdaBoost.
* **Regressors**: Linear Regression, Lasso, Ridge, SVR, Decision Tree, Random Forest, Gradient Boosting.

---

## 👨‍💻 Author

**Uditya Narayan Tiwari**


## 📄 License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
