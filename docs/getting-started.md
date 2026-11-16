# Getting started

## 1. Install

```bash
pip install multimodel-analysis
```

Needs Python 3.10 or newer. It installs numpy, pandas, matplotlib, seaborn and scikit-learn for you.

Optional extras (install only what you need):

| Extra | Command | Adds |
|---|---|---|
| `boost` | `pip install "multimodel-analysis[boost]"` | XGBoost, LightGBM, CatBoost models |
| `umap` | `pip install "multimodel-analysis[umap]"` | UMAP in `MultiModelReducer` |
| `excel` | `pip install "multimodel-analysis[excel]"` | saving reports as `.xlsx` |

`include="all"` runs every model whose package is installed, so nothing breaks if you skip the extras. Asking for one by name without its package is reported in `failed_models_`.

## 2. Pick the right class

```
Do you have a column you want to predict?
├── yes ── is it a category (yes/no, species, spam/not spam)? → MultiModelClassifier
│          is it a number (price, temperature, sales)?         → MultiModelRegressor
└── no ─── do you want to find groups?                         → MultiModelClusterer
           do you want to find unusual rows?                   → MultiModelAnomalyDetector
           do you want fewer columns / a 2-D picture?          → MultiModelReducer
```

## 3. The workflow (identical for all five)

```python
import multimodel_analysis as mm

tool = mm.MultiModelClassifier(X, y, random_state=0)   # 1. give it your data
models = tool.run_all_models()                          # 2. train the standard models
table = tool.show_tabular_report(models, return_df=True)  # 3. ranked results (a DataFrame)
tool.plot_comparison(models)                            # 4. ranked bar chart
best = tool.predict(X_new)                              # 5. use the best model
```

* `X` is a DataFrame or array of features. `y` is the target (not needed for the unsupervised classes;
  there it is optional and only used to *score* the result).
* `random_state` makes runs repeatable.
* `verbose=False` silences the printed tables when you only want the returned DataFrame.
* `run_all_models()` returns a list of results; pass that list to the report and plot methods.

### Choosing models

```python
tool.run_all_models(include=["random_forest", "svm"])       # only these
tool.run_all_models(exclude=["svm"])                         # everything standard except these
tool.run_all_models(include="all")                           # standard + extra models (those whose package is installed)
tool.run_model("random_forest", n_estimators=300)            # one model, your own settings
```

The list of keys is in [models.md](models.md). `MultiModelClassifier.default_models()` and
`.available_models()` print them in code.

### Bring your own model

Any scikit-learn-style estimator can join the comparison:

```python
from sklearn.linear_model import SGDClassifier
tool.run_all_models(custom_models={"My SGD": SGDClassifier(random_state=0)})
```

## 4. Messy data

By default the data must be all numeric with no missing values (the tool tells you what is wrong in
plain words). To let it clean up for you, pass `preprocess=True`:

```python
tool = mm.MultiModelClassifier(df, y, preprocess=True)
```

Missing numbers are filled with the median, missing categories with the most common value, numbers
are scaled and text columns are one-hot encoded. The same steps are saved with the model, so
`predict(new_df)` accepts raw, un-cleaned rows.

## 5. When a model fails

A model that cannot handle your data is *skipped*, not fatal:

```python
tool.failed_models_    # {model name: reason}
tool.warnings_         # notes such as "DBSCAN found only noise"
```

## 6. Where to go next

| I want to... | Read |
|---|---|
| predict a category or a number | [supervised.md](supervised.md) |
| find groups | [clustering.md](clustering.md) |
| find unusual rows | [anomaly-detection.md](anomaly-detection.md) |
| reduce columns / plot in 2-D | [dimensionality-reduction.md](dimensionality-reduction.md) |
| see which columns matter | [explainability.md](explainability.md) |
| save a model or share a report | [reports-and-persistence.md](reports-and-persistence.md) |
