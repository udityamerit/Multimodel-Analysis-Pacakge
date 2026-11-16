"""Phase 4: hyper-parameter tuning."""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, make_regression

from multimodel_analysis import MultiModelClassifier, MultiModelRegressor
from multimodel_analysis.supervised import CLASSIFIER_REGISTRY, REGRESSOR_REGISTRY


def test_every_builtin_model_has_a_search_space_that_matches_its_estimator():
    from multimodel_analysis.core.registry import BuildContext

    ctx = BuildContext(random_state=0, n_jobs=1, n_train=100)
    for registry in (CLASSIFIER_REGISTRY, REGRESSOR_REGISTRY):
        for spec in registry:
            assert spec.search_space, f"{spec.name} has no search space"
            est = spec.build(ctx)
            valid = est.get_params()
            for param in spec.search_space:
                assert spec.param_prefix + param in valid, f"{spec.name}: unknown parameter {param}"


def test_random_search_classifier_returns_report_tuples_and_best_params():
    X, y = load_breast_cancer(return_X_y=True)
    clf = MultiModelClassifier(X, y, verbose=False)
    models = clf.tune_models(include=["logistic_regression", "decision_tree"], n_iter=4, cv=3)
    assert [m[0] for m in models] == ["Logistic Regression", "Decision Tree"]
    assert set(clf.best_params_) == {"Logistic Regression", "Decision Tree"}
    assert "C" in clf.best_params_["Logistic Regression"]
    assert clf.get_model("Logistic Regression").C == clf.best_params_["Logistic Regression"]["C"]
    df = clf.show_tabular_report(models, return_df=True)  # report works on tuned models
    assert len(df) == 2
    res = clf.tuning_results_
    assert {"Model", "Best CV accuracy", "Best Params", "Candidates", "Search Time (s)"} <= set(res.columns)
    assert res["Best CV accuracy"].is_monotonic_decreasing


def test_grid_search_tries_every_combination_and_custom_grid_overrides_default():
    X, y = load_breast_cancer(return_X_y=True)
    clf = MultiModelClassifier(X, y, verbose=False)
    clf.tune_models(include=["knn"], search="grid", cv=3, scoring="f1",
                    param_grids={"knn": {"n_neighbors": [3, 5], "weights": ["uniform", "distance"]}})
    assert clf.tuning_results_.loc[0, "Candidates"] == 4
    assert "Best CV f1" in clf.tuning_results_.columns
    assert set(clf.best_params_["KNN"]) == {"n_neighbors", "weights"}


def test_n_iter_is_capped_at_grid_size():
    X, y = load_breast_cancer(return_X_y=True)
    clf = MultiModelClassifier(X, y, verbose=False)
    clf.tune_models(include=["logistic_regression"], n_iter=50, cv=3)
    assert clf.tuning_results_.loc[0, "Candidates"] == 5  # the default C grid has 5 values


def test_halving_search_runs():
    X, y = make_regression(n_samples=300, n_features=6, noise=5, random_state=0)
    reg = MultiModelRegressor(X, y, verbose=False)
    reg.tune_models(include=["ridge"], search="halving", cv=3)
    assert "alpha" in reg.best_params_["Ridge Regression"]


def test_regressor_svr_wrapper_params_are_unprefixed_and_fitted():
    X, y = make_regression(n_samples=150, n_features=5, noise=2, random_state=0)
    reg = MultiModelRegressor(X, y, verbose=False)
    reg.tune_models(include=["svr"], n_iter=3, cv=3)
    best = reg.best_params_["SVR"]
    assert "C" in best and not any(k.startswith("regressor__") for k in best)
    assert reg.get_model("SVR").regressor.C == best["C"]
    assert "Best CV r2" in reg.tuning_results_.columns


def test_tuning_with_preprocess_on_messy_data():
    rng = np.random.default_rng(1)
    n = 150
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.choice(["u", "v"], n)})
    X.loc[rng.choice(n, 10, replace=False), "a"] = np.nan
    y = (X["b"] + (X["c"] == "u") > 0.5).astype(int)
    clf = MultiModelClassifier(X, y, preprocess=True, verbose=False)
    clf.tune_models(include=["logistic_regression"], cv=3)
    assert not clf.failed_models_
    assert len(clf.predict(X.iloc[:5], model="Logistic Regression")) == 5


def test_tuned_model_round_trips_through_predict():
    X, y = load_breast_cancer(return_X_y=True)
    clf = MultiModelClassifier(X, y, verbose=False)
    clf.tune_models(include=["random_forest"], n_iter=2, cv=3)
    assert len(clf.predict(X[:5], model="Random Forest")) == 5


def test_validation_and_failure_recording():
    X, y = load_breast_cancer(return_X_y=True)
    clf = MultiModelClassifier(X, y, verbose=False)
    with pytest.raises(ValueError):
        clf.tune_models(search="bayes")
    with pytest.raises(ValueError):
        clf.tune_models(n_iter=0)
    with pytest.raises(KeyError):
        clf.tune_models(param_grids={"no_such_model": {"a": [1]}})
    models = clf.tune_models(include=["naive_bayes"], cv=3, param_grids={"naive_bayes": {"bogus": [1, 2]}})
    assert models == [] and "Naive Bayes" in clf.failed_models_
