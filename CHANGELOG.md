# Changelog

## 0.2.0.dev0 - Phase 2: package restructure

- Code split into `multimodel_analysis/core` (base, registry, utils) and `multimodel_analysis/supervised` (classification, regression). `multimodel_analysis.multimodel` remains as a compatibility shim; all old imports and method names still work.
- Model registry (`ModelSpec`, `ModelRegistry`): every built-in model is one registry entry (key, display name, factory, defaults).
- New: `run_model(name, **params)`, `available_models()`, `get_model(name)`, `run_all_models(include=, exclude=)`, `estimators_` (fitted models) and `fit_times_`.

## 0.1.4 - Phase 0/1: bug fixes and packaging

### Behaviour changes (read these)
- `scaled_data` now defaults to `True` (features are standardised, fit on the training split only).
- `Support_vector_model` now defaults to `kernel='rbf'` (was `'linear'`).
- The `ROC AUC` column is numeric (`NaN` when unavailable) instead of the string `"N/A"`.
- `run_all_models()` no longer silences all warnings: they are stored in `warnings_` and the important ones are printed.
- Models that fail are listed in `failed_models_` and shown under the report table.
- `MultiModelRegressor.run_all_models()` now includes a KNN regressor (9 models); `SVR` standardises the target internally.

### Fixed
- Confusion matrix / classification report crashed or had the wrong shape when the test split missed a class.
- Multiclass ROC AUC was `NaN` whenever one class was absent from the test split; it is now computed on the classes present.
- Continuous targets, NaN/inf or text columns in `X`, X/y length mismatch and invalid `test_size` now raise clear errors.
- Seaborn palette `FutureWarning` in `plot_comparison`.
- `LogisticRegression(n_jobs=...)` deprecation warning on scikit-learn >= 1.8.
- `.xlsx` export without `openpyxl` gives an install hint; legacy `.xls` is rejected clearly.
- Removed dead code (unused imports, broken `__main__` check).

### Added
- `verbose`, `warnings_`, `failed_models_`, `sort_by=` in `show_tabular_report`.
- `pyproject.toml`, GitHub Actions CI (tests + ruff), pytest suite (25 tests), `examples/phase1_results.ipynb`.
