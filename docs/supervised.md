# Classification and regression

`MultiModelClassifier` predicts a **category**; `MultiModelRegressor` predicts a **number**. They share
almost everything, so this page covers both.

## Classification

```python
from sklearn.datasets import load_breast_cancer
from multimodel_analysis import MultiModelClassifier

data = load_breast_cancer(as_frame=True)
clf = MultiModelClassifier(data.data, data.target, test_size=0.3, random_state=0)

models = clf.run_all_models()
table = clf.show_tabular_report(models, return_df=True, sort_by="F1 Score")
clf.plot_comparison(models, metric="F1 Score")
clf.plot_confusion_matrices(models)
clf.plot_roc_curves(models)
```

The report has these columns: `Accuracy`, `Precision`, `Recall`, `F1 Score`, `ROC AUC`, `Fit Time (s)`.

| Metric | Plain-English meaning |
|---|---|
| Accuracy | share of all predictions that were right |
| Precision | when it says "positive", how often it is right |
| Recall | of all real positives, how many it found |
| F1 Score | one number balancing precision and recall |
| ROC AUC | how well it ranks positives above negatives (1.0 perfect, 0.5 coin-flip) |

Targets can be numbers, strings or categories, binary or multi-class. Strings come back as strings:
`clf.predict(new_rows)` returns your original labels.

## Regression

```python
from sklearn.datasets import load_diabetes
from multimodel_analysis import MultiModelRegressor

data = load_diabetes(as_frame=True)
reg = MultiModelRegressor(data.data, data.target, random_state=0)

models = reg.run_all_models()
reg.show_tabular_report(models, sort_by="RMSE")
reg.plot_comparison(models, metric="R2 Score")
reg.plot_true_vs_predicted(models)
```

Columns: `MAE`, `MSE`, `RMSE`, `R2 Score`, `Fit Time (s)`.

| Metric | Plain-English meaning |
|---|---|
| MAE | average size of the error, in the units of your target |
| MSE / RMSE | like MAE but big mistakes count more (RMSE is back in target units) |
| R2 Score | share of the variation explained (1.0 perfect, 0 = no better than the average) |

For error metrics (`MAE`, `MSE`, `RMSE`) lower is better; the report sorts and ranks accordingly.

## Check results more fairly: cross-validation

One train/test split can be lucky. `cross_validate_models` scores every model on several folds of
the **training** split only (the test split stays untouched):

```python
cv = clf.cross_validate_models(cv=5)
```

## Improve results: hyper-parameter tuning

Every built-in model has a sensible search space (see [models.md](models.md)).

```python
tuned = clf.tune_models(include=["logistic_regression", "random_forest"], search="random", n_iter=20, cv=5)

clf.tuning_results_      # DataFrame: best CV score, candidates tried, time
clf.best_params_         # {model name: best parameters}
clf.show_tabular_report(tuned)
```

* `search="random"` (default) tries `n_iter` random combinations; `"grid"` tries all of them;
  `"halving"` quickly drops weak candidates (fast on big data).
* `scoring="f1"` (or any scikit-learn scorer name) chooses what "best" means.
* `param_grids={"knn": {"n_neighbors": [3, 5, 7]}}` replaces the built-in search space for a model.
* A model whose tuning fails is listed in `failed_models_`; the others carry on.

Each `run_all_models` / `tune_models` call replaces the fitted models, so run `tune_models` last
for the ones you want to keep, or save them first (see [reports-and-persistence.md](reports-and-persistence.md)).

## Predict on new data

```python
clf.predict(new_rows)                          # best model of the last report
clf.predict(new_rows, model="Random Forest")   # a specific model, by display name
```

`new_rows` must have the same columns as the training data. Scaling (and, with `preprocess=True`,
the whole cleaning pipeline) is applied for you. For classifiers saved with `export_model()`,
`predict_proba(new_rows)` gives class probabilities.

## Useful options

| Option | Default | Meaning |
|---|---|---|
| `test_size` | `0.3` | share of rows kept for testing (classification splits are stratified) |
| `scaled_data` | `True` | standardise features (needed by SVM, KNN, linear models) |
| `preprocess` | `False` | impute missing values and encode text columns |
| `random_state` | `42` | repeatable results |
| `n_jobs` | `-1` | CPU cores to use |
| `verbose` | `True` | print tables while running |

## Notebooks

* [`examples/phase1_results.ipynb`](../examples/phase1_results.ipynb) — first run, reports and plots
* [`examples/phase4_tuning.ipynb`](../examples/phase4_tuning.ipynb) — tuning, with before/after tables
* [`examples/phase5_more_models.ipynb`](../examples/phase5_more_models.ipynb) — the extra models and boosters
