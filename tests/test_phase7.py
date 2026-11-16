"""Phase 7: anomaly detection, dimensionality reduction, explainability and the HTML report."""
import matplotlib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, load_iris

matplotlib.use("Agg")

from multimodel_analysis import (  # noqa: E402
    MultiModelAnomalyDetector,
    MultiModelClassifier,
    MultiModelClusterer,
    MultiModelReducer,
    MultiModelRegressor,
    load_model,
)
from multimodel_analysis.unsupervised import ANOMALY_REGISTRY, REDUCER_REGISTRY  # noqa: E402


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def anomaly_data():
    rng = np.random.RandomState(0)
    normal = rng.randn(300, 3)
    odd = rng.randn(15, 3) * 0.4 + 7
    X = pd.DataFrame(np.vstack([normal, odd]), columns=["a", "b", "c"])
    y = np.r_[np.zeros(300, int), np.ones(15, int)]
    return X, y


@pytest.fixture(scope="module")
def detector(anomaly_data):
    X, y = anomaly_data
    det = MultiModelAnomalyDetector(X, y=y, verbose=False)
    det.run_all_models()
    return det


@pytest.fixture(scope="module")
def digits():
    d = load_digits()
    return d.data[:600], d.target[:600]


@pytest.fixture(scope="module")
def reducer(digits):
    X, y = digits
    red = MultiModelReducer(X, y=y, verbose=False)
    red.run_all_models(include=["pca", "fastica", "isomap", "kernel_pca"])
    return red


@pytest.fixture(scope="module")
def classifier():
    d = load_breast_cancer(as_frame=True)
    clf = MultiModelClassifier(d.data, d.target, verbose=False)
    clf.run_all_models(include=["logistic_regression", "random_forest"])
    clf.show_tabular_report(return_df=True)
    return clf


@pytest.fixture(scope="module")
def regressor():
    d = load_diabetes(as_frame=True)
    reg = MultiModelRegressor(d.data, d.target, verbose=False)
    reg.run_all_models(include=["linear_regression", "random_forest"])
    reg.show_tabular_report(return_df=True)
    return reg


# --------------------------------------------------------------------------- anomaly detection
def test_anomaly_registry_sets():
    standard = [s.key for s in ANOMALY_REGISTRY.select()]
    assert {"isolation_forest", "lof", "one_class_svm", "elliptic_envelope", "knn_distance",
            "robust_zscore"} <= set(standard)
    assert "sgd_one_class_svm" not in standard
    assert "sgd_one_class_svm" in [s.key for s in ANOMALY_REGISTRY.select("all")]


def test_detectors_find_planted_outliers(detector):
    df = detector.show_tabular_report(return_df=True)
    assert len(df) >= 6 and not detector.failed_models_
    assert df["F1 Score"].iloc[0] > 0.9
    assert {"Precision", "Recall", "ROC AUC", "Agreement"} <= set(df.columns)


def test_every_model_flags_the_same_number_of_rows(detector):
    counts = {int(r.labels.sum()) for r in detector._results.values()}
    assert counts == {15}  # contamination taken from y: 15 of 315


def test_contamination_defaults(anomaly_data):
    X, _ = anomaly_data
    det = MultiModelAnomalyDetector(X, verbose=False)
    assert det.contamination == 0.05
    det.run_all_models(include=["isolation_forest", "lof"])
    df = det.show_tabular_report(return_df=True)
    assert "F1 Score" not in df.columns and "Agreement" in df.columns


def test_top_anomalies_and_consensus(detector, anomaly_data):
    top = detector.top_anomalies(5)
    assert len(top) == 5 and (top.index >= 300).all()  # planted rows are 300..314
    cons = detector.consensus()
    assert {"votes", "consensus"} <= set(cons.columns)
    assert cons["consensus"].iloc[300:].sum() >= 12
    with pytest.raises(ValueError):
        detector.consensus(min_votes=99)
    assert set(detector.assign_anomalies().columns) >= {"anomaly", "anomaly_score"}


def test_anomaly_predict_and_save_roundtrip(detector, tmp_path):
    new = pd.DataFrame({"a": [0.0, 8.0], "b": [0.1, 7.0], "c": [-0.2, 6.5]})
    labels = detector.predict(new, model="Isolation Forest")
    assert list(labels) == [0, 1]
    assert detector.score_new(new, model="Isolation Forest")[1] > detector.score_new(new, model="Isolation Forest")[0]
    path = detector.save_model(str(tmp_path / "iso.joblib"), model="Isolation Forest")
    assert list(load_model(path).predict(new)) == [0, 1]


def test_anomaly_input_validation(anomaly_data):
    X, y = anomaly_data
    with pytest.raises(ValueError):
        MultiModelAnomalyDetector(X.iloc[:5], verbose=False)
    with pytest.raises(ValueError):
        MultiModelAnomalyDetector(X, y=np.full(len(X), 2), verbose=False)
    with pytest.raises(ValueError):
        MultiModelAnomalyDetector(X, y=np.zeros(len(X), int), verbose=False)
    with pytest.raises(ValueError):
        MultiModelAnomalyDetector(X, contamination=0.9, verbose=False)
    with pytest.raises(ValueError):
        MultiModelAnomalyDetector(X, y=y[:10], verbose=False)


def test_anomaly_plots_and_summary(detector, tmp_path):
    detector.plot_comparison(save_path=str(tmp_path / "c.png"), show_plot=False)
    detector.plot_anomalies(save_path=str(tmp_path / "a.png"), show_plot=False)
    detector.plot_scores(save_path=str(tmp_path / "s.png"), show_plot=False)
    detector.get_summary(save_prefix=str(tmp_path / "run"), show_plot=False)
    assert (tmp_path / "run_report.csv").exists() and (tmp_path / "run_anomalies.png").exists()


def test_anomaly_preprocess_handles_text_and_gaps():
    rng = np.random.RandomState(1)
    df = pd.DataFrame({"x": rng.randn(120), "kind": rng.choice(["u", "v"], 120)})
    df.loc[[3, 9], "x"] = np.nan
    det = MultiModelAnomalyDetector(df, preprocess=True, verbose=False)
    det.run_all_models(include=["isolation_forest", "lof"])
    assert len(det.show_tabular_report(return_df=True)) == 2


def test_anomaly_custom_model(anomaly_data):
    from sklearn.ensemble import IsolationForest
    X, _ = anomaly_data
    det = MultiModelAnomalyDetector(X, verbose=False)
    det.run_all_models(custom_models={"My Forest": IsolationForest(random_state=0)}, include=["lof"])
    assert "My Forest" in set(det.show_tabular_report(return_df=True)["Model"])


# --------------------------------------------------------------------------- dimensionality reduction
def test_reducer_registry_sets():
    standard = [s.key for s in REDUCER_REGISTRY.select()]
    assert {"pca", "kernel_pca", "fastica", "factor_analysis", "isomap", "lle", "tsne"} <= set(standard)
    assert "mds" not in standard and "mds" in [s.key for s in REDUCER_REGISTRY.select("all")]


def test_reducer_report_and_scores(reducer):
    df = reducer.show_tabular_report(return_df=True)
    assert {"Trustworthiness", "Continuity", "Global Structure", "Variance Kept (%)",
            "Class Separation", "Can Transform New Data"} <= set(df.columns)
    assert (df["Trustworthiness"].between(0, 1)).all()
    assert df["Trustworthiness"].is_monotonic_decreasing
    assert not reducer.failed_models_


def test_reducer_embedding_loadings_and_transform(reducer, digits):
    X, _ = digits
    emb = reducer.get_embedding("PCA")
    assert emb.shape == (len(X), 2) and list(emb.columns) == ["Component 1", "Component 2"]
    load = reducer.get_loadings("PCA")
    assert load.shape == (X.shape[1], 2)
    new = reducer.transform(X[:5], model="PCA")
    assert new.shape == (5, 2)
    np.testing.assert_allclose(new.to_numpy(), emb.iloc[:5].to_numpy(), atol=1e-3)
    with pytest.raises(TypeError):
        reducer.get_loadings("Isomap")


def test_reducer_save_roundtrip_and_non_transformers(reducer, digits, tmp_path):
    X, _ = digits
    path = reducer.save_model(str(tmp_path / "pca.joblib"), model="PCA")
    again = load_model(path).transform(X[:3])
    assert again.shape == (3, 2)
    red = MultiModelReducer(X, verbose=False)
    red.run_all_models(include=["tsne"])
    with pytest.raises(TypeError):
        red.transform(X[:3], model="t-SNE")


def test_find_n_components(digits):
    X, _ = digits
    red = MultiModelReducer(X, verbose=False)
    k = red.find_n_components(0.9)
    assert 5 < k < 60 and len(red.variance_table_) >= k


def test_reducer_works_without_labels_and_with_float_target():
    iris = load_iris()
    red = MultiModelReducer(iris.data, verbose=False)
    red.run_all_models(include=["pca", "isomap"])
    assert "Class Separation" not in red.show_tabular_report(return_df=True).columns
    red2 = MultiModelReducer(iris.data, y=iris.data[:, 0], verbose=False)  # continuous y: coloured, not scored
    red2.run_all_models(include=["pca"])
    assert "Class Separation" not in red2.show_tabular_report(return_df=True).columns


def test_reducer_input_validation(digits):
    X, y = digits
    with pytest.raises(ValueError):
        MultiModelReducer(X[:, :1], verbose=False)
    with pytest.raises(ValueError):
        MultiModelReducer(X, n_components=0, verbose=False)
    with pytest.raises(ValueError):
        MultiModelReducer(X, n_components=500, verbose=False)
    with pytest.raises(ValueError):
        MultiModelReducer(X[:5], verbose=False)


def test_reducer_large_data_guard(monkeypatch, digits):
    X, _ = digits
    from multimodel_analysis.unsupervised import reduction
    monkeypatch.setitem(reduction._MAX_SAMPLES, "tsne", 100)
    red = MultiModelReducer(X, verbose=False)
    red.run_all_models(include=["pca", "tsne"])
    assert any("t-SNE" in k for k in red.failed_models_)
    red.run_all_models(include=["pca", "tsne"], allow_large=True)
    assert not red.failed_models_


def test_reducer_plots_and_summary(reducer, tmp_path):
    reducer.plot_comparison(save_path=str(tmp_path / "c.png"), show_plot=False)
    reducer.plot_embeddings(save_path=str(tmp_path / "e.png"), show_plot=False)
    reducer.find_n_components(0.9)
    reducer.plot_variance(save_path=str(tmp_path / "v.png"), show_plot=False)
    reducer.get_summary(save_prefix=str(tmp_path / "run"), show_plot=False)
    assert (tmp_path / "run_embeddings.png").exists()


# --------------------------------------------------------------------------- explainability
def test_feature_importance_classifier(classifier):
    imp = classifier.feature_importance()
    assert list(imp.columns) == ["Feature", "Importance", "Std"]
    assert imp["Importance"].is_monotonic_decreasing
    assert len(imp) == 30 and imp["Importance"].iloc[0] > 0
    assert classifier.importances_[classifier._best_model_name()] is imp


def test_feature_importance_regressor_finds_bmi(regressor):
    imp = regressor.feature_importance("Random Forest")
    assert imp["Feature"].iloc[0] in {"bmi", "s5"}
    assert imp["Importance"].iloc[0] > 0.1


def test_feature_importance_unknown_model_and_unfitted():
    d = load_iris(as_frame=True)
    fresh = MultiModelClassifier(d.data, d.target, verbose=False)
    with pytest.raises(RuntimeError):
        fresh.feature_importance()
    fresh.run_all_models(include=["logistic_regression"])
    with pytest.raises(KeyError):
        fresh.feature_importance("Nonexistent")


def test_partial_dependence_classifier(classifier):
    pdp = classifier.partial_dependence("worst radius", model="Logistic Regression")
    assert list(pdp.columns) == ["Feature", "Feature value", "Probability", "Model"]
    assert pdp["Probability"].between(0, 1).all() and pdp["Feature value"].is_monotonic_increasing
    first = classifier.partial_dependence("worst radius", model="Logistic Regression", class_label=0)
    assert np.allclose(first["Probability"] + pdp["Probability"], 1.0, atol=1e-6)
    with pytest.raises(KeyError):
        classifier.partial_dependence("nope")
    with pytest.raises(KeyError):
        classifier.partial_dependence("worst radius", class_label="cat")


def test_partial_dependence_regressor(regressor):
    pdp = regressor.partial_dependence("bmi", model="Linear Regression")
    assert list(pdp.columns)[2] == "Prediction"
    assert pdp["Prediction"].iloc[-1] > pdp["Prediction"].iloc[0]  # more BMI -> higher disease progression


def test_explain_plots(classifier, regressor, tmp_path):
    classifier.plot_feature_importance(top=8, save_path=str(tmp_path / "fi.png"), show_plot=False)
    classifier.plot_partial_dependence(["worst radius", 0], save_path=str(tmp_path / "pd.png"), show_plot=False)
    regressor.plot_partial_dependence(save_path=str(tmp_path / "pr.png"), show_plot=False)
    assert all((tmp_path / f).stat().st_size > 1000 for f in ("fi.png", "pd.png", "pr.png"))


def test_explain_with_preprocess_and_text_labels():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"num": rng.randn(200), "colour": rng.choice(["r", "g"], 200)})
    y = np.where(df["num"] + (df["colour"] == "r") > 0.5, "yes", "no")
    clf = MultiModelClassifier(df, y, preprocess=True, verbose=False)
    clf.run_all_models(include=["logistic_regression"])
    clf.show_tabular_report(return_df=True)
    assert clf.feature_importance()["Feature"].iloc[0].startswith("num")
    assert len(clf.partial_dependence("num", class_label="yes")) > 3


# --------------------------------------------------------------------------- HTML report
def _check_html(path, *needles):
    text = open(path, encoding="utf-8").read()
    assert text.startswith("<!DOCTYPE html>") and "<table>" in text
    for n in needles:
        assert n in text, n
    return text


def test_html_report_classifier(classifier, tmp_path):
    p = classifier.save_html_report(str(tmp_path / "clf.html"), title="Cancer <study>")
    text = _check_html(p, "Cancer &lt;study&gt;", "Confusion matrices", "ROC curves", "Which features matter")
    assert text.count("data:image/png;base64,") == 5
    assert "Logistic Regression" in text


def test_html_report_regressor_and_unsupervised(regressor, detector, reducer, tmp_path):
    _check_html(regressor.save_html_report(str(tmp_path / "r.html")), "True vs predicted")
    _check_html(detector.save_html_report(str(tmp_path / "a.html")), "Score distribution", "Anomaly detection")
    _check_html(reducer.save_html_report(str(tmp_path / "d.html")), "Embeddings", "Dimensionality reduction")
    d = load_iris(as_frame=True)
    cl = MultiModelClusterer(d.data, y=d.target, n_clusters=3, verbose=False)
    cl.run_all_models(include=["kmeans", "agglomerative"])
    cl.show_tabular_report(return_df=True)
    _check_html(cl.save_html_report(str(tmp_path / "c.html")), "Clusters found")


def test_html_report_needs_a_table_and_lists_failures(tmp_path):
    d = load_iris(as_frame=True)
    clf = MultiModelClassifier(d.data, d.target, verbose=False)
    clf.run_all_models(include=["logistic_regression"])
    with pytest.raises(ValueError):
        clf.save_html_report(str(tmp_path / "x.html"))
    clf.failed_models_["Broken Model"] = "it exploded"
    clf.show_tabular_report(return_df=True)
    clf.failed_models_["Broken Model"] = "it exploded"
    text = _check_html(clf.save_html_report(str(tmp_path / "f.html")), "Broken Model", "it exploded")
    assert "Models that failed" in text


# --------------------------------------------------------------------------- refactor sanity
def test_trained_model_transform_requires_transform_support(classifier):
    tm = classifier.export_model()
    with pytest.raises(TypeError):
        tm.transform(classifier.X_test_raw)


def test_clusterer_still_shares_the_unsupervised_base():
    from multimodel_analysis.unsupervised._base import _UnsupervisedBase
    assert issubclass(MultiModelClusterer, _UnsupervisedBase)
    assert issubclass(MultiModelAnomalyDetector, _UnsupervisedBase)
    assert issubclass(MultiModelReducer, _UnsupervisedBase)
