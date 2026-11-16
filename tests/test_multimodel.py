import os
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from multimodel_analysis import (
    MultiModelClassifier,
    MultiModelRegressior,
    MultiModelRegressor,
    save_report,
)


@pytest.fixture(autouse=True)
def _headless_matplotlib(monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")


# ----------------------------------------------------------------- global state
def test_import_does_not_add_global_warning_filter():
    """Importing the package must not install a blanket warnings 'ignore' filter or change plot style."""
    code = textwrap.dedent("""
        import warnings, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn, sklearn  # these add their own filters; baseline AFTER them
        before = list(warnings.filters)
        style_before = dict(plt.rcParams)
        import multimodel_analysis
        after = list(warnings.filters)
        assert before == after, "multimodel_analysis changed warnings.filters on import"
        assert style_before == dict(plt.rcParams), "multimodel_analysis changed matplotlib rcParams on import"
        print("OK")
    """)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_run_all_models_does_not_mutate_warning_filters():
    X, y = make_classification(n_samples=80, n_features=5, random_state=0)
    clf = MultiModelClassifier(X, y, verbose=False)
    before = list(warnings.filters)
    clf.run_all_models()
    assert list(warnings.filters) == before


# ----------------------------------------------------------------- classifier basics
def test_classifier_binary_end_to_end(tmp_path):
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    df_X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(5)])

    clf = MultiModelClassifier(df_X, y, random_state=42)
    models = clf.run_all_models()
    assert len(models) + len(clf.failed_models_) == 8
    assert len(models) >= 1

    df_report = clf.show_tabular_report(models, return_df=True)
    assert isinstance(df_report, pd.DataFrame) and not df_report.empty
    assert df_report["ROC AUC"].dtype.kind == "f"  # numeric, never "N/A" strings

    clf.plot_confusion_matrices(models, save_path=str(tmp_path / "cm.png"), show_plot=False)
    clf.plot_roc_curves(models, save_path=str(tmp_path / "roc.png"), show_plot=False)
    clf.plot_comparison(models, save_path=str(tmp_path / "cmp.png"), show_plot=False)
    for name in ("cm.png", "roc.png", "cmp.png"):
        assert (tmp_path / name).exists()


def test_defaults_are_scaled_and_rbf_svm():
    X, y = make_classification(n_samples=60, n_features=4, random_state=1)
    clf = MultiModelClassifier(X, y)
    assert clf.scaled_data is True
    assert abs(np.mean(clf.X_train_scaled)) < 1e-8
    svc_res = clf.Support_vector_model()
    assert svc_res[2] >= 0


def test_string_labels_and_confusion_matrix_report_use_original_names():
    X, y = make_classification(n_samples=90, n_features=5, n_classes=3, n_informative=4, n_redundant=0, random_state=3)
    labels = np.array(["cat", "dog", "bird"])[y]
    clf = MultiModelClassifier(X, labels, verbose=False)
    res = clf.Logistic_model()
    report, matrix = res[0], res[1]
    assert "cat" in report and "dog" in report and "bird" in report
    assert matrix.shape == (3, 3)


def test_multiclass_roc_has_macro_curve():
    X, y = make_classification(n_samples=150, n_features=6, n_classes=3, n_informative=4, n_redundant=0, random_state=42)
    clf = MultiModelClassifier(X, y, verbose=False)
    res = clf.RandomForest_model()
    fpr_dict, roc_auc = res[6], res[8]
    assert "macro" in fpr_dict
    assert 0.0 <= roc_auc <= 1.0
    clf.plot_roc_curves([("RF", *res)], show_plot=False)


# ----------------------------------------------------------------- bug regressions
def test_continuous_target_rejected_by_classifier():
    X, y = make_regression(n_samples=60, n_features=3, random_state=0)
    with pytest.raises(ValueError, match="continuous"):
        MultiModelClassifier(X, y)


def test_input_validation_errors():
    X, y = make_classification(n_samples=40, n_features=4, random_state=0)
    with pytest.raises(ValueError, match="different lengths"):
        MultiModelClassifier(X, y[:-3])
    with pytest.raises(ValueError, match="test_size"):
        MultiModelClassifier(X, y, test_size=1.5)
    with pytest.raises(ValueError, match="missing"):
        y_bad = y.astype(float)
        y_bad[0] = np.nan
        MultiModelClassifier(X, y_bad)
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        MultiModelClassifier(X_nan, y)
    df_text = pd.DataFrame({"a": np.arange(40.0), "city": ["x", "y"] * 20})
    with pytest.raises(ValueError, match="city"):
        MultiModelClassifier(df_text, y)
    with pytest.raises(ValueError, match="only one class"):
        MultiModelClassifier(X, np.zeros(40, dtype=int))
    with pytest.raises(ValueError):
        MultiModelClassifier(None, None)


def test_confusion_matrix_has_full_shape_when_test_split_misses_a_class():
    """A class absent from the test split must not shrink the confusion matrix or crash the report."""
    rng = np.random.RandomState(0)
    X = rng.randn(60, 3)
    y = np.array([0] * 28 + [1] * 28 + [2] * 4)
    clf = MultiModelClassifier(X, y, test_size=0.2, stratify=False, random_state=2, verbose=False)
    res = clf.Logistic_model()
    assert res[1].shape == (clf.n_classes_, clf.n_classes_)
    assert not np.isnan(res[8]) or len(np.unique(clf.y_test)) < 2


def test_roc_auc_nan_when_probabilities_fail():
    X, y = make_classification(n_samples=40, n_features=4, random_state=42)
    clf = MultiModelClassifier(X, y, test_size=0.3, verbose=False)

    class CorruptModel:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            raise RuntimeError("Simulated probability failure")

    res = clf.evaluate_model(CorruptModel(), clf.X_test_scaled, clf.y_test)
    assert np.isnan(res[8])


def test_failed_models_are_recorded_not_hidden(capsys):
    X, y = make_classification(n_samples=60, n_features=4, random_state=0)
    clf = MultiModelClassifier(X, y, verbose=False)

    class Broken:
        def fit(self, X, y):
            raise RuntimeError("boom")

    models = clf.run_all_models(custom_models={"Broken": Broken()})
    assert "Broken" not in [m[0] for m in models]
    assert "Broken" in clf.failed_models_
    assert "boom" in clf.failed_models_["Broken"]
    clf.show_tabular_report(models)
    assert "Broken" in capsys.readouterr().out


def test_convergence_warnings_are_captured():
    X, y = make_classification(n_samples=100, n_features=5, random_state=0)
    clf = MultiModelClassifier(X * 1000, y, scaled_data=False, verbose=False)
    clf.Logistic_model(max_iter=1)
    assert any("Convergence" in w for ws in clf.warnings_.values() for w in ws)


def test_no_seaborn_palette_futurewarning(tmp_path):
    X, y = make_classification(n_samples=60, n_features=4, random_state=0)
    clf = MultiModelClassifier(X, y, verbose=False)
    models = [("LR", *clf.Logistic_model()), ("DT", *clf.DecisionTree_model())]
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        clf.plot_comparison(models, show_plot=False)


def test_sort_by_option_and_validation():
    X, y = make_classification(n_samples=80, n_features=4, random_state=0)
    clf = MultiModelClassifier(X, y, verbose=False)
    clf.run_all_models()
    df = clf.show_tabular_report(return_df=True, sort_by="F1 Score")
    assert df["F1 Score"].is_monotonic_decreasing
    with pytest.raises(ValueError, match="sort_by"):
        clf.show_tabular_report(sort_by="Nope")


# ----------------------------------------------------------------- regressor
def test_regressor_2d_target_and_plots(tmp_path):
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    reg = MultiModelRegressor(X, pd.DataFrame({"target": y}), random_state=42)
    models = reg.run_all_models()

    names = [m[0] for m in models]
    assert "KNN Regressor" in names
    assert len(models) + len(reg.failed_models_) == 9

    df_report = reg.show_tabular_report(models, return_df=True)
    assert df_report["R2 Score"].is_monotonic_decreasing

    reg.plot_true_vs_predicted(models, save_path=str(tmp_path / "tvp.png"), show_plot=False)
    reg.plot_comparison(models, save_path=str(tmp_path / "cmp.png"), show_plot=False)
    assert (tmp_path / "tvp.png").exists() and (tmp_path / "cmp.png").exists()


def test_regressor_error_metric_sorts_ascending():
    X, y = make_regression(n_samples=80, n_features=4, noise=1.0, random_state=0)
    reg = MultiModelRegressor(X, y, verbose=False)
    df = reg.show_tabular_report(reg.run_all_models(), return_df=True, sort_by="RMSE")
    assert df["RMSE"].is_monotonic_increasing


def test_svr_is_competitive_on_large_scale_target():
    """SVR on an unscaled large-valued target used to give R2 near 0; the target is now standardised internally."""
    X, y = make_regression(n_samples=150, n_features=5, noise=0.5, random_state=0)
    reg = MultiModelRegressor(X, y * 1000, verbose=False)
    r2 = reg.SVR_model()[3]
    assert r2 > 0.5


def test_regressor_rejects_non_numeric_target():
    X = np.random.RandomState(0).randn(20, 2)
    with pytest.raises(ValueError, match="numeric"):
        MultiModelRegressor(X, np.array(["a", "b"] * 10))


# ----------------------------------------------------------------- misc API
def test_custom_models_and_report_saving(tmp_path):
    X, y = make_classification(n_samples=80, n_features=4, random_state=42)
    clf = MultiModelClassifier(X, y, verbose=False)
    models = clf.run_all_models(custom_models={"My Custom RF": RandomForestClassifier(n_estimators=10, random_state=42)})
    assert "My Custom RF" in [m[0] for m in models]

    df_report = clf.show_tabular_report(models, return_df=True)
    save_report(df_report, str(tmp_path / "report.csv"))
    save_report(df_report, str(tmp_path / "report.json"))
    save_report(df_report, str(tmp_path / "report.html"))
    clf.save_report(str(tmp_path / "report_method.csv"))
    for name in ("report.csv", "report.json", "report.html", "report_method.csv"):
        assert (tmp_path / name).exists()


def test_legacy_xls_rejected(tmp_path):
    with pytest.raises(ValueError, match="xlsx"):
        save_report(pd.DataFrame({"a": [1]}), str(tmp_path / "r.xls"))


def test_backward_compatibility_alias():
    assert MultiModelRegressior is MultiModelRegressor


def test_random_state_accepted_by_individual_models():
    X, y = make_classification(n_samples=60, n_features=4, random_state=42)
    clf = MultiModelClassifier(X, y, random_state=42, verbose=False)
    for res in (clf.Logistic_model(random_state=1), clf.RandomForest_model(random_state=1, n_estimators=20),
                clf.DecisionTree_model(random_state=1), clf.KNN_model(random_state=1),
                clf.Naive_Bayes_model(random_state=1)):
        assert res[2] > 0.0

    X_r, y_r = make_regression(n_samples=60, n_features=4, random_state=42)
    reg = MultiModelRegressor(X_r, y_r, random_state=42, verbose=False)
    for res in (reg.Lasso_model(random_state=9, alpha=0.05), reg.Ridge_model(random_state=9),
                reg.LinearRegression_model(random_state=9), reg.AdaBoost_model(random_state=9),
                reg.KNN_model(random_state=9)):
        assert len(res) == 5


def test_same_random_state_is_reproducible():
    X, y = make_classification(n_samples=100, n_features=5, random_state=0)
    a = MultiModelClassifier(X, y, random_state=7, verbose=False).RandomForest_model()
    b = MultiModelClassifier(X, y, random_state=7, verbose=False).RandomForest_model()
    assert a[2] == b[2] and np.array_equal(a[1], b[1])


def test_omitted_models_auto_run_and_default_report_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    X, y = make_classification(n_samples=80, n_features=4, random_state=42)
    clf = MultiModelClassifier(X, y, verbose=False)

    df_report = clf.show_tabular_report(return_df=True)
    assert df_report is not None and not df_report.empty
    clf.get_summary(show_plot=False)

    save_report(df_report)
    assert (tmp_path / "report.csv").exists()
    os.remove(tmp_path / "report.csv")
    clf.save_report()
    assert (tmp_path / "report.csv").exists()


def test_custom_sklearn_estimator_instance_is_used():
    X, y = make_classification(n_samples=60, n_features=4, random_state=0)
    clf = MultiModelClassifier(X, y, verbose=False)
    models = clf.run_all_models(custom_models={"LR-C0.1": LogisticRegression(C=0.1)})
    assert models[-1][0] == "LR-C0.1"


# ----------------------------------------------------------------- Phase 2: registry / structure
def test_old_import_paths_still_work():
    from multimodel_analysis.multimodel import MultiModelClassifier as OldClf
    from multimodel_analysis.multimodel import MultiModelRegressior as OldReg
    from multimodel_analysis.multimodel import save_report as old_save
    from multimodel_analysis.supervised import MultiModelClassifier as NewClf
    assert OldClf is NewClf is MultiModelClassifier
    assert OldReg is MultiModelRegressor
    assert old_save is save_report


def test_available_models_and_include_exclude():
    assert "random_forest" in MultiModelClassifier.available_models()
    assert "knn" in MultiModelRegressor.available_models()
    X, y = make_classification(n_samples=80, n_features=4, random_state=0)
    clf = MultiModelClassifier(X, y, verbose=False)

    only = clf.run_all_models(include=["random_forest", "Logistic Regression"])
    assert [m[0] for m in only] == ["Logistic Regression", "Random Forest"]

    fewer = clf.run_all_models(exclude=["svm", "adaboost"])
    assert len(fewer) == 6 and "SVM" not in [m[0] for m in fewer]

    with pytest.raises(KeyError, match="Unknown model"):
        clf.run_all_models(include=["not_a_model"])


def test_run_model_by_name_with_params_and_get_fitted_model():
    X, y = make_classification(n_samples=80, n_features=4, random_state=0)
    clf = MultiModelClassifier(X, y, verbose=False)
    res = clf.run_model("random_forest", n_estimators=7)
    assert res[2] > 0
    est = clf.get_model("Random Forest")
    assert est.n_estimators == 7
    assert clf.get_model("random_forest") is est
    assert clf.fit_times_["Random Forest"] >= 0
    preds = est.predict(clf.X_test_scaled)
    assert len(preds) == len(clf.y_test)
    with pytest.raises(KeyError):
        clf.get_model("nope")


def test_run_all_models_stores_estimators_for_regressor():
    X, y = make_regression(n_samples=80, n_features=4, random_state=0)
    reg = MultiModelRegressor(X, y, verbose=False)
    models = reg.run_all_models(custom_models={"MyLR": __import__("sklearn.linear_model", fromlist=["Ridge"]).Ridge()})
    assert set(reg.estimators_) == {m[0] for m in models}
    assert "MyLR" in reg.estimators_
