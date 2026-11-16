"""Phase 3: preprocessing, predict, save/load, cross-validation, class_weight, fit-time column."""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_iris, make_regression

from multimodel_analysis import MultiModelClassifier, MultiModelRegressor, TrainedModel, load_model

FAST = ["logistic_regression", "decision_tree", "naive_bayes"]


def _messy_frame(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "age": rng.integers(18, 70, n).astype(float),
        "income": rng.normal(50, 10, n),
        "city": rng.choice(["Delhi", "Pune", "Goa"], n),
    })
    y = ((X["age"] > 40) ^ (X["city"] == "Goa")).astype(int).map({0: "no", 1: "yes"})
    X.loc[rng.choice(n, 10, replace=False), "income"] = np.nan
    X.loc[rng.choice(n, 8, replace=False), "city"] = None
    return X, y


def test_strict_by_default_but_preprocess_accepts_nan_and_text():
    X, y = _messy_frame()
    with pytest.raises(ValueError):
        MultiModelClassifier(X, y, verbose=False)
    clf = MultiModelClassifier(X, y, preprocess=True, verbose=False)
    out = clf.run_all_models(include=FAST)
    assert len(out) == 3 and not clf.failed_models_
    assert not clf.X_train_scaled.isna().any().any()
    assert any(c.startswith("city_") for c in clf.feature_names_)


def test_preprocess_regressor_and_unseen_category_in_predict():
    X, _ = _messy_frame()
    y = X["income"].fillna(50) * 2 + 1
    reg = MultiModelRegressor(X, y, preprocess=True, verbose=False)
    reg.run_all_models(include=["linear_regression", "ridge"])
    new = pd.DataFrame({"age": [30.0], "income": [np.nan], "city": ["Mumbai"]})  # unseen city + NaN
    assert np.isfinite(reg.predict(new, model="Ridge Regression")).all()


def test_predict_returns_original_labels_and_uses_best_model():
    data = load_iris(as_frame=True)
    clf = MultiModelClassifier(data.data, data.target_names[data.target], verbose=False)
    clf.show_tabular_report(clf.run_all_models(include=FAST))
    pred = clf.predict(data.data.iloc[:5])
    assert set(pred) <= set(data.target_names)
    best = clf.show_tabular_report(return_df=True).iloc[0]["Model"]
    assert list(pred) == list(clf.predict(data.data.iloc[:5], model=best))


def test_predict_matches_manual_pipeline_numpy_input():
    X, y = load_breast_cancer(return_X_y=True)
    clf = MultiModelClassifier(X, y, verbose=False)
    clf.run_all_models(include=["logistic_regression"])
    est = clf.get_model("Logistic Regression")
    expected = est.predict(clf.scaler.transform(X[:10]))
    assert np.array_equal(clf.predict(X[:10], model="Logistic Regression"), expected)


def test_predict_without_choice_raises_when_ambiguous():
    X, y = load_breast_cancer(return_X_y=True)
    clf = MultiModelClassifier(X, y, verbose=False)
    clf.run_all_models(include=FAST)
    with pytest.raises(ValueError, match="Say which model"):
        clf.predict(X[:3])


def test_save_and_load_roundtrip(tmp_path):
    data = load_iris(as_frame=True)
    clf = MultiModelClassifier(data.data, data.target_names[data.target], verbose=False)
    clf.run_all_models(include=FAST)
    path = clf.save_model(str(tmp_path / "m.joblib"), model="Decision Tree")
    loaded = load_model(path)
    assert isinstance(loaded, TrainedModel) and loaded.task == "classification"
    rows = data.data.iloc[:10]
    assert list(loaded.predict(rows)) == list(clf.predict(rows, model="Decision Tree"))
    proba = loaded.predict_proba(rows)
    assert list(proba.columns) == [str(c) for c in data.target_names]
    # missing columns are reported clearly
    with pytest.raises(ValueError, match="missing column"):
        loaded.predict(rows.drop(columns=rows.columns[0]))


def test_load_model_rejects_foreign_files(tmp_path):
    import joblib

    joblib.dump({"a": 1}, tmp_path / "x.joblib")
    with pytest.raises(ValueError):
        load_model(str(tmp_path / "x.joblib"))


def test_cross_validation_classifier_multiple_scorers():
    X, y = load_breast_cancer(return_X_y=True)
    clf = MultiModelClassifier(X, y, verbose=False)
    df = clf.cross_validate_models(cv=3, scoring=["accuracy", "f1"], include=FAST)
    assert list(df["Model"].sort_values()) == sorted(["Logistic Regression", "Decision Tree", "Naive Bayes"])
    assert {"accuracy mean", "accuracy std", "f1 mean", "Fit Time (s)"} <= set(df.columns)
    assert df["accuracy mean"].is_monotonic_decreasing
    assert clf.cv_results_ is df


def test_cross_validation_regressor_with_preprocess_has_no_leakage_errors():
    X, _ = _messy_frame()
    y = X["income"].fillna(50) * 3
    reg = MultiModelRegressor(X, y, preprocess=True, verbose=False)
    df = reg.cross_validate_models(cv=3, scoring="neg_root_mean_squared_error", include=["linear_regression"])
    assert df.loc[0, "neg_root_mean_squared_error mean"] <= 0


def test_cross_validation_records_failures():
    X, y = load_breast_cancer(return_X_y=True)
    clf = MultiModelClassifier(X, y, verbose=False)
    df = clf.cross_validate_models(cv=3, include=["naive_bayes"], scoring="not_a_scorer")
    assert df.empty and "Naive Bayes" in clf.cv_failed_


def test_class_weight_balanced_applied_only_where_supported():
    X, y = make_regression(n_samples=200, n_features=5, random_state=0)
    y = (y > np.percentile(y, 90)).astype(int)  # ~10% positives
    clf = MultiModelClassifier(X, y, class_weight="balanced", verbose=False)
    clf.run_all_models(include=["logistic_regression", "random_forest", "knn", "naive_bayes"])
    assert clf.get_model("Logistic Regression").class_weight == "balanced"
    assert clf.get_model("Random Forest").class_weight == "balanced"
    assert not clf.failed_models_
    with pytest.raises(ValueError):
        MultiModelClassifier(X, y, class_weight="weird", verbose=False)


def test_report_contains_fit_time_column():
    X, y = load_breast_cancer(return_X_y=True)
    clf = MultiModelClassifier(X, y, verbose=False)
    df = clf.show_tabular_report(clf.run_all_models(include=FAST), return_df=True)
    assert "Fit Time (s)" in df.columns and (df["Fit Time (s)"] >= 0).all()
    Xr, yr = make_regression(n_samples=100, n_features=4, random_state=0)
    reg = MultiModelRegressor(Xr, yr, verbose=False)
    dfr = reg.show_tabular_report(reg.run_all_models(include=["ridge"]), return_df=True)
    assert "Fit Time (s)" in dfr.columns
