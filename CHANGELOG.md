# Changelog

## 0.6.0 - Phase 7: anomaly detection, dimensionality reduction, explainability, HTML report

- New `MultiModelAnomalyDetector(X, y=None, contamination=None)`: Isolation Forest, Local Outlier Factor, One-Class SVM, Elliptic Envelope, KNN Distance and Robust Z-Score (opt-in: SGD One-Class SVM). Every model flags the same share of rows (`contamination`: from `y` when given, else 5%), so the models are comparable. Optional true labels `y` (0 / 1) add Precision, Recall, F1, ROC AUC and Average Precision; without labels the report shows `Agreement` (how much a model sees the data like the others).
  - `top_anomalies()`, `consensus(min_votes=)` (rows flagged by many models), `assign_anomalies()`, `predict()` / `score_new()` for new rows, `save_model()`, `plot_comparison / plot_anomalies / plot_scores`, `get_summary()`.
- New `MultiModelReducer(X, y=None, n_components=2)`: PCA, Kernel PCA, FastICA, Factor Analysis, Isomap, LLE, t-SNE (opt-in: Truncated SVD, Spectral Embedding, MDS, Random Projection, UMAP via `pip install multimodel_analysis[umap]`). Scored with Trustworthiness, Continuity, Global Structure, Variance Kept and (with `y`) Class Separation.
  - `find_n_components(variance=0.95)`, `plot_variance()`, `get_embedding()`, `get_loadings()`, `transform(X_new)` and `save_model()` for methods that can map new rows, `plot_embeddings()` (coloured by `y`), `get_summary()`.
- Explainability for classifiers and regressors (no extra packages, computed on the test split): `feature_importance()` (permutation importance), `plot_feature_importance()`, `partial_dependence()` and `plot_partial_dependence()`.
- `save_html_report(path, title=)` on every analyser: one self-contained HTML file with the results table, the plots, failed models and warnings; open it in any browser or e-mail it.
- Internal: clustering, anomaly detection and reduction share one `_UnsupervisedBase`; `TrainedModel.transform()` added.
- New notebook `examples/phase7_anomaly_reduction_explain.ipynb` with executed output.
- Minimum Python is now 3.10 (tested on 3.10, 3.11 and 3.12).

## 0.5.0 - Phase 6: clustering (unsupervised learning)

- New `MultiModelClusterer(X, y=None, n_clusters=3 | 'auto', scaled_data=True, preprocess=False)`: runs clustering algorithms on one dataset, scores them and recommends the best. `y` (optional true labels) is never used for fitting, only to add external scores (ARI, NMI, V-measure, Homogeneity, Completeness).
  - Standard set: KMeans, MiniBatch KMeans, Agglomerative (Ward), Gaussian Mixture, Birch, Spectral, DBSCAN. Opt-in: Bisecting KMeans, Agglomerative (average / complete), HDBSCAN, OPTICS, Mean Shift, Affinity Propagation (`include=[...]` or `include='all'`).
  - Internal scores: Silhouette, Calinski-Harabasz, Davies-Bouldin. Noise points (label -1) are reported and left out of the scores; a result with fewer than 2 clusters gets N/A and a warning instead of a crash.
  - DBSCAN `eps` and Mean Shift bandwidth are chosen from the data (`auto_params_`). Models that need O(n^2) memory are skipped on large data unless `allow_large=True`.
  - `find_optimal_k()` / `n_clusters='auto'`, `tune_models(scoring=, param_grids=)`, `assign_clusters()`, `cluster_profile()`, `get_labels()`.
  - `predict(X_new)`, `export_model()`, `save_model()` for models that support it (KMeans, Gaussian Mixture, Birch, ...), with a clear error for the ones that cannot label new rows.
  - Plots: `plot_comparison`, `plot_clusters` (PCA for more than 2 features), `plot_k_search`, `plot_silhouette`, `get_summary`.
- Internal: reporting helpers shared by all analysers moved to `core/mixins.py`; feature validation split out as `_validate_features`.
- New notebook `examples/phase6_clustering.ipynb` with executed output.

## 0.4.0 - Phase 5: more models and optional boosters

- New opt-in models. `run_all_models()` still runs the same standard set; use `include=[...]` for specific ones or `include='all'` for every model whose package is installed.
  - Classification: Extra Trees, Hist Gradient Boosting, Neural Network (MLP), Linear Discriminant Analysis, Ridge Classifier, Linear SVM.
  - Regression: Elastic Net, Bayesian Ridge, Huber Regressor, Linear SVR, Extra Trees, Hist Gradient Boosting, Neural Network (MLP).
  - Optional boosters (classification and regression): XGBoost, LightGBM, CatBoost - `pip install multimodel_analysis[boost]`. A missing package is skipped by `include='all'` and reported with an install hint when asked for by name.
- Every new model has a search space, so `tune_models`, `cross_validate_models`, `predict` and `save_model` work with them. `class_weight='balanced'` reaches Extra Trees, Ridge Classifier, Linear SVM and LightGBM.
- `ModelSpec` gained `default` (part of the standard set?) and `available` (is the optional package installed?). New `default_models()` and `available_models(installed_only=False)`.
- SVR, Linear SVR and the MLP regressor standardise the target internally.
- New notebook `examples/phase5_more_models.ipynb` with executed output.
- `plot_comparison` redesigned: one clean ranked bar chart (best model on top, value on every bar). It shows F1 Score / R2 Score by default; pick another with `metric=`. The old grouped bars were unreadable with 17+ models.

## 0.3.0 - Phase 4: hyper-parameter tuning

- `tune_models(include=, exclude=, search='random'|'grid'|'halving', n_iter=20, cv=5, scoring=, param_grids=)`: tunes each model on the training split with leak-free CV (preprocessing is refitted inside every fold) and returns the same report tuples as `run_all_models`, so `show_tabular_report`, `predict`, plots and `save_model` work on tuned models.
- Every built-in classifier and regressor ships a default `search_space` (`ModelSpec.search_space`); pass `param_grids={'knn': {...}}` to override one.
- New attributes `best_params_` (per model) and `tuning_results_` (best CV score, parameters, candidates tried, search time). Failures go to `failed_models_`; `n_iter` is capped at the grid size.
- New notebook `examples/phase4_tuning.ipynb` with executed output.
- Repo cleanup: removed stale example notebooks/CSV and an old PyPI screenshot; `.gitignore` and `MANIFEST.in` tidied.

## 0.2.0 - restructure, registry, preprocessing, prediction, cross-validation

### Phase 3: preprocessing, prediction, persistence, cross-validation

- `preprocess=True` (classifier and regressor): impute missing values, one-hot encode text columns and scale, fitted on the training split only. Default stays strict.
- `predict(X_new, model=None)`, `export_model()`, `save_model(path)` and `load_model(path)` (joblib); a saved `TrainedModel` carries its preprocessing and returns original class labels.
- `cross_validate_models(cv=5, scoring=..., include=, exclude=, custom_models=)`: leak-free CV on the training split, results in `cv_results_`.
- `class_weight='balanced'` for classification (Logistic Regression, SVM, Decision Tree, Random Forest).
- `Fit Time (s)` column in both reports.

### Phase 2: package restructure

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
