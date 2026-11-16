"""Phase 6: MultiModelClusterer."""
import matplotlib
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_moons

matplotlib.use("Agg")

from multimodel_analysis import MultiModelClusterer, load_model  # noqa: E402
from multimodel_analysis.unsupervised import CLUSTERER_REGISTRY  # noqa: E402


@pytest.fixture(scope="module")
def blobs():
    X, y = make_blobs(n_samples=240, centers=3, cluster_std=0.8, random_state=3)
    return pd.DataFrame(X, columns=["a", "b"]), y


@pytest.fixture(scope="module")
def fitted(blobs):
    X, y = blobs
    mm = MultiModelClusterer(X, y=y, n_clusters=3, verbose=False)
    mm.run_all_models()
    return mm


def test_registry_standard_and_extra_sets():
    default_keys = [s.key for s in CLUSTERER_REGISTRY.select()]
    assert default_keys == ["kmeans", "minibatch_kmeans", "agglomerative", "gaussian_mixture",
                            "birch", "spectral", "dbscan"]
    assert len(CLUSTERER_REGISTRY.select(include="all")) >= len(default_keys) + 5


def test_run_all_models_finds_the_three_blobs(fitted):
    assert not fitted.failed_models_
    names = {m.name for m in fitted.models_}
    assert {"KMeans", "Gaussian Mixture", "DBSCAN"} <= names
    km = fitted._results["KMeans"]
    assert km.n_clusters == 3 and km.n_noise == 0
    assert km.silhouette > 0.6
    assert km.external["ARI"] == pytest.approx(1.0)


def test_report_is_sorted_and_includes_external_scores(fitted):
    df = fitted.show_tabular_report(return_df=True)
    assert list(df["Silhouette"]) == sorted(df["Silhouette"], reverse=True)
    assert {"ARI", "NMI", "V-measure", "Homogeneity", "Completeness"} <= set(df.columns)
    db = fitted.show_tabular_report(return_df=True, sort_by="Davies-Bouldin")
    assert list(db["Davies-Bouldin"]) == sorted(db["Davies-Bouldin"])
    with pytest.raises(ValueError):
        fitted.show_tabular_report(return_df=True, sort_by="Accuracy")


def test_report_without_true_labels_has_no_external_columns(blobs):
    mm = MultiModelClusterer(blobs[0], n_clusters=3, verbose=False)
    df = mm.show_tabular_report(mm.run_all_models(include=["kmeans"]), return_df=True)
    assert "ARI" not in df.columns


def test_labels_assignments_and_profile(fitted):
    labels = fitted.get_labels("kmeans")
    assert labels.shape == (240,)
    out = fitted.assign_clusters("KMeans")
    assert "cluster" in out.columns and len(out) == 240
    prof = fitted.cluster_profile("KMeans")
    assert prof["Size"].sum() == 240 and prof["Share (%)"].sum() == pytest.approx(100, abs=0.5)
    assert {"a", "b"} <= set(prof.columns)
    with pytest.raises(KeyError):
        fitted.get_labels("not a model")


def test_dbscan_marks_noise_and_scores_ignore_it():
    X, _ = make_moons(n_samples=300, noise=0.08, random_state=0)
    X = np.vstack([X, [[3, 3], [-3, 3], [3, -3]]])  # three obvious outliers
    mm = MultiModelClusterer(X, n_clusters=2, verbose=False)
    res = mm.run_model("dbscan", eps=0.3, min_samples=5)
    assert res.n_noise >= 3 and (res.labels == -1).sum() == res.n_noise
    assert np.isfinite(res.silhouette)


def test_single_cluster_result_is_nan_not_a_crash(blobs):
    mm = MultiModelClusterer(blobs[0], n_clusters=3, verbose=False)
    res = mm.run_model("dbscan", eps=100.0)  # everything in one cluster
    assert res.n_clusters == 1 and np.isnan(res.silhouette)
    assert any("at least 2" in w for w in mm.warnings_["DBSCAN"])
    df = mm.show_tabular_report([res], return_df=True)
    assert len(df) == 1


def test_auto_k_picks_three_for_three_blobs(blobs):
    mm = MultiModelClusterer(blobs[0], n_clusters="auto", verbose=False)
    ks = mm.find_optimal_k()
    assert mm.optimal_k_ == 3
    assert list(ks["k"]) == list(range(2, 11))
    res = mm.run_model("kmeans")
    assert mm.n_clusters_ == 3 and res.n_clusters == 3


def test_predict_and_save_roundtrip(fitted, tmp_path, blobs):
    X = blobs[0]
    new = X.iloc[:10]
    pred = fitted.predict(new, "kmeans")
    assert pred.shape == (10,) and set(pred) <= {0, 1, 2}
    assert list(pred) == list(fitted.get_labels("kmeans")[:10])  # same scaling as in training
    path = fitted.save_model(str(tmp_path / "km.joblib"), "kmeans")
    assert list(load_model(path).predict(new)) == list(pred)


def test_predict_without_model_picks_best_that_can_predict(fitted, blobs):
    assert fitted.predict(blobs[0].iloc[:5]).shape == (5,)


def test_models_without_predict_raise_a_helpful_error(fitted, blobs):
    with pytest.raises(TypeError, match="cannot label new data"):
        fitted.predict(blobs[0].iloc[:3], "dbscan")


def test_tune_models_returns_best_params(blobs):
    mm = MultiModelClusterer(blobs[0], n_clusters=3, verbose=False)
    models = mm.tune_models(include=["kmeans", "dbscan"])
    assert {m.name for m in models} == {"KMeans", "DBSCAN"}
    assert mm.best_params_["KMeans"]["n_clusters"] == 3
    assert "eps" in mm.best_params_["DBSCAN"]
    assert list(mm.tuning_results_.columns)[:2] == ["Model", "Best Silhouette"]
    with pytest.raises(ValueError):
        mm.tune_models(scoring="accuracy")


def test_tune_with_custom_grid(blobs):
    mm = MultiModelClusterer(blobs[0], n_clusters=3, verbose=False)
    mm.tune_models(include=["kmeans"], param_grids={"kmeans": {"n_clusters": [2, 3]}})
    assert mm.best_params_["KMeans"] == {"n_clusters": 3}


def test_custom_models_and_include_exclude(blobs):
    mm = MultiModelClusterer(blobs[0], n_clusters=3, verbose=False)
    models = mm.run_all_models(custom_models={"My KMeans": KMeans(n_clusters=3, n_init=3, random_state=0)},
                               include=["kmeans", "birch"], exclude=["birch"])
    assert {m.name for m in models} == {"KMeans", "My KMeans"}


def test_large_data_guard_skips_quadratic_models(blobs, monkeypatch):
    mm = MultiModelClusterer(blobs[0], n_clusters=3, verbose=False)
    monkeypatch.setattr("multimodel_analysis.unsupervised.clustering._MAX_SAMPLES", {"spectral": 10})
    mm.run_all_models(include=["kmeans", "spectral"])
    assert "Spectral Clustering" in mm.failed_models_
    assert "skipped" in mm.failed_models_["Spectral Clustering"]
    mm.run_all_models(include=["kmeans", "spectral"], allow_large=True)
    assert not mm.failed_models_


def test_preprocess_handles_text_and_missing_values():
    df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 8.0, 9.0, 10.0, 1.5, 9.5],
                       "kind": ["a", "a", "b", "b", "b", "a", "a", "b"]})
    mm = MultiModelClusterer(df, n_clusters=2, preprocess=True, verbose=False)
    res = mm.run_model("kmeans")
    assert res.n_clusters == 2
    assert mm.predict(df.iloc[:3], "kmeans").shape == (3,)


def test_invalid_inputs():
    X = np.random.default_rng(0).normal(size=(30, 2))
    with pytest.raises(ValueError):
        MultiModelClusterer(None)
    with pytest.raises(ValueError, match="n_clusters"):
        MultiModelClusterer(X, n_clusters=1)
    with pytest.raises(ValueError, match="n_clusters"):
        MultiModelClusterer(X, n_clusters=99)
    with pytest.raises(ValueError, match="different lengths"):
        MultiModelClusterer(X, y=np.zeros(5))
    bad = X.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        MultiModelClusterer(bad)
    with pytest.raises(ValueError):
        MultiModelClusterer(X[:3])


def test_plots_are_saved(fitted, tmp_path):
    for name, call in {
        "cmp": lambda p: fitted.plot_comparison(save_path=p, show_plot=False),
        "cmp_db": lambda p: fitted.plot_comparison(save_path=p, show_plot=False, metric="Davies-Bouldin"),
        "clusters": lambda p: fitted.plot_clusters(save_path=p, show_plot=False, top=4),
        "sil": lambda p: fitted.plot_silhouette("kmeans", save_path=p, show_plot=False),
        "k": lambda p: fitted.plot_k_search(save_path=p, show_plot=False),
    }.items():
        path = str(tmp_path / f"{name}.png")
        call(path)
        assert (tmp_path / f"{name}.png").stat().st_size > 1000


def test_plot_clusters_projects_many_features_with_pca():
    X, _ = make_blobs(n_samples=120, n_features=6, centers=3, random_state=1)
    mm = MultiModelClusterer(X, n_clusters=3, verbose=False)
    mm.run_all_models(include=["kmeans"])
    mm.plot_clusters(show_plot=False)  # must not raise


def test_get_summary_writes_files(fitted, tmp_path):
    prefix = str(tmp_path / "run")
    fitted.get_summary(save_prefix=prefix, show_plot=False)
    for suffix in ("report.csv", "comparison.png", "clusters.png"):
        assert (tmp_path / f"run_{suffix}").exists()
