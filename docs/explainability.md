# Explainability: which features matter?

A ranked table tells you *which model* is best. These two tools tell you *why it predicts what it
predicts*. They work for every model in `MultiModelClassifier` and `MultiModelRegressor`, including
ones with no built-in importance (SVM, KNN, neural networks), because they only use predictions.

## Feature importance (permutation)

Shuffle one column at a time and see how much the test score drops. A big drop means the model
relied on that column.

```python
from sklearn.datasets import load_breast_cancer
from multimodel_analysis import MultiModelClassifier

data = load_breast_cancer(as_frame=True)
clf = MultiModelClassifier(data.data, data.target, random_state=0)
models = clf.run_all_models(include=["logistic_regression", "random_forest", "gradient_boosting"])
clf.show_tabular_report(models)       # picks the best model, which the calls below use by default

clf.feature_importance().head(8)              # table, for the best model
clf.plot_feature_importance(top=10)           # ranked bars
clf.plot_feature_importance(model="Random Forest", top=10)
```

The table has three columns: `Feature`, `Importance` (the mean score drop: accuracy for classifiers, R2
for regressors) and `Std` (the spread over `n_repeats` shuffles, default 10). `0.10` means shuffling the
feature costs 10 points. A feature whose drop is about zero, or whose `Std` is larger than its
`Importance`, does not help the model. Without `model=`, the best model of the last report is used, so
call `show_tabular_report()` first (or pass `model="Random Forest"`). Raise
`n_repeats` for steadier numbers; `scoring="f1"` (or any scorer name) changes the yardstick;
`max_rows` (default 2000) keeps big datasets fast.

With `preprocess=True`, a text column that was one-hot encoded appears as one row per category
(`colour_red`, `colour_green`, ...), because that is what the model sees.

## Partial dependence: how does a feature act?

Sweep one feature over its range, keep everything else as it is, and plot the average prediction.

```python
clf.plot_partial_dependence(model="Random Forest")                       # the top features
clf.plot_partial_dependence(["worst radius", "worst concave points"], model="Random Forest")
clf.partial_dependence("worst radius", model="Random Forest", grid_points=6)   # the numbers
```

* Classification: the curve is the probability of one class. Choose it with `class_label=0`.
* Regression: the curve is the predicted value, for example `reg.plot_partial_dependence(["bmi", "s5"])`.
* Rising curve: larger values push the prediction up. Flat curve: the feature barely matters.

## Cautions

* Importance says what the *model* uses, not what causes the outcome.
* Two strongly related columns share credit: shuffling one is rescued by the other, so both can look
  less important than they are. Look at them together.
* Importance is measured on the **test** split, so it reflects generalisation, not memorisation.

## Notebook

[`examples/phase7_anomaly_reduction_explain.ipynb`](../examples/phase7_anomaly_reduction_explain.ipynb)
