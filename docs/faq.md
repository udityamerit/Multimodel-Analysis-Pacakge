# FAQ and troubleshooting

## Which class do I use?

See the decision tree in [getting-started.md](getting-started.md#2-pick-the-right-class).

## It says my data has missing values / text columns

By default the data must be numeric and complete. Either clean it yourself, or pass `preprocess=True`
and the tool fills missing values (median for numbers, most common for categories), scales numbers
and one-hot encodes text.

## A model is missing from the report

It was skipped or failed. Look at the reason:

```python
tool.failed_models_
tool.warnings_
```

Common reasons: the optional package is not installed (`pip install "multimodel-analysis[boost]"`),
the data is too big for a slow method (use `allow_large=True`, or leave it out), or the method needs
data your set does not have (for example too few rows for cross-validation).

## "contamination must be a number between 0 and 0.5"

`MultiModelAnomalyDetector(contamination=...)` is the *share* of rows you expect to be unusual:
`0.02` means 2%, not `2`.

## DBSCAN / HDBSCAN found one big cluster, or only noise

Those methods choose their own number of clusters from the data's density. Check
`tool.auto_params_` and `tool.warnings_`; try `tune_models(include=["dbscan"])` or give your own
`param_grids`. Noise rows have label `-1`.

## `predict()` raises a TypeError for a clustering or reduction method

That method cannot place unseen rows (DBSCAN, Spectral, Agglomerative, t-SNE, ...). Use a method that
can (KMeans, Gaussian Mixture, PCA, ...); see [clustering.md](clustering.md) and
[dimensionality-reduction.md](dimensionality-reduction.md). Calling `predict` without `model=` picks the
best one that works.

## The numbers change every run

Pass `random_state=0` (or any number) when you create the object.

## My report shows no plot

Plots open in a window, or inline in a notebook (`%matplotlib inline`). On a server or in CI, use
`show_plot=False` with `save_path="x.png"`, or set `MPLBACKEND=Agg`.

## Why are my results different from scikit-learn alone?

The tool standardises features (`scaled_data=True`) and uses a fixed train/test split. Set
`scaled_data=False` to turn scaling off. Scores are on the held-out test split, so a different
`test_size` or `random_state` gives different numbers.

## Can I add my own model?

Yes: `run_all_models(custom_models={"My Model": estimator})` for any scikit-learn-style estimator.

## Is it safe to share a saved model?

Model files are pickles; load only files from sources you trust. Share the HTML report instead when
you only need to show results.

## Which Python versions work?

Python 3.10, 3.11 and 3.12 are tested on every change.

## Old code stopped working after upgrading

The original methods (`Logistic_model()`, `KNN_model()`, `get_summary()`, ...) and the old spelling
`MultiModelRegressior` still work. If you find something that does not, please
[open an issue](https://github.com/udityamerit/Multimodel-Analysis-Pacakge/issues).

## Where are the release notes?

[CHANGELOG.md](../CHANGELOG.md).
