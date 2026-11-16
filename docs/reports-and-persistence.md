# Reports, plots, and saving models

## Tables

`show_tabular_report(models, return_df=True)` prints the ranked table and returns it as a DataFrame.
Save any table with the method or the helper:

```python
df = tool.show_tabular_report(models, return_df=True)
tool.save_report(df, "results.csv")      # .csv  .xlsx  .html  .json (by extension)
```

`.xlsx` needs `pip install "multimodel-analysis[excel]"`. The old `.xls` format is not supported.

## Plots

Every plot method accepts `save_path="plot.png"` (saves a PNG) and `show_plot=False` (do not open a
window, handy on servers). Each class has the plots that make sense for it:

| Class | Plots |
|---|---|
| Classifier | `plot_comparison`, `plot_confusion_matrices`, `plot_roc_curves`, `plot_feature_importance`, `plot_partial_dependence` |
| Regressor | `plot_comparison`, `plot_true_vs_predicted`, `plot_feature_importance`, `plot_partial_dependence` |
| Clusterer | `plot_comparison`, `plot_clusters`, `plot_k_search`, `plot_silhouette` |
| Anomaly detector | `plot_comparison`, `plot_anomalies`, `plot_scores` |
| Reducer | `plot_comparison`, `plot_embeddings`, `plot_variance` |

`plot_comparison(models, metric="...")` draws one plain, ranked bar chart for any metric in the table.

## Everything at once

```python
tool.get_summary(save_prefix="out/run1", show_plot=False)
```

writes the CSV table and the PNG plots, all starting with `out/run1_` (the folder must already exist, otherwise you get an `OSError`).

## One-file HTML report

```python
tool.save_html_report("report.html", title="Breast cancer classification")
```

A single self-contained page with the results table, every plot (embedded as images) and any
warnings or failed models. Nothing else is needed to open it, so it is safe to e-mail or attach.
An example is in [`examples/phase7_report.html`](../examples/phase7_report.html).

## Save a model and use it later

```python
tool.save_model("best.joblib", model="Random Forest")     # model + its preprocessing in one file

from multimodel_analysis import load_model
loaded = load_model("best.joblib")
loaded.predict(new_rows)           # raw rows in, original labels out
loaded.predict_proba(new_rows)     # classifiers: a DataFrame of class probabilities
```

* Scaling and (with `preprocess=True`) imputing/encoding are saved with the model, so you do not
  re-create them.
* Each class saves what it can do: classifiers/regressors `predict`, clusterers `predict`, anomaly
  detectors `predict` and `score_new` (with the learned cut-off), reducers `transform`.
* `tool.export_model("Random Forest")` returns the same bundle as an object (a `TrainedModel`) without
  writing a file; `loaded.save(path)` writes it.

> **Security note:** model files are Python pickles (joblib). Only load files you created or trust,
> exactly as with any pickle.

## After a run

| Attribute | Contains |
|---|---|
| `failed_models_` | `{model: reason}` for every model that was skipped or failed |
| `warnings_` | non-fatal notes collected while fitting |
| `tuning_results_`, `best_params_` | after `tune_models()` |
| `estimators_` | the fitted estimators by display name (or use `get_model(name)`) |
