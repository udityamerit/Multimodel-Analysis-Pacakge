"""Phase 5: extra models and optional boosters."""
import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_iris, make_regression

from multimodel_analysis import MultiModelClassifier, MultiModelRegressor
from multimodel_analysis.core.registry import ModelRegistry, ModelSpec
from multimodel_analysis.supervised import CLASSIFIER_REGISTRY, REGRESSOR_REGISTRY

CLF_DEFAULT = ["Logistic Regression", "SVM", "Decision Tree", "KNN", "Naive Bayes", "Random Forest",
               "Gradient Boosting", "AdaBoost"]
REG_DEFAULT = ["Linear Regression", "Lasso Regression", "Ridge Regression", "SVR", "KNN Regressor",
               "Decision Tree Regressor", "Random Forest Regressor", "Gradient Boosting Regressor",
               "AdaBoost Regressor"]


@pytest.fixture(scope="module")
def clf():
    X, y = load_breast_cancer(return_X_y=True)
    return MultiModelClassifier(X, y, random_state=0, verbose=False)


@pytest.fixture(scope="module")
def reg():
    X, y = make_regression(n_samples=200, n_features=6, noise=5, random_state=0)
    return MultiModelRegressor(X, y, random_state=0, verbose=False)


def test_default_run_is_unchanged_and_extras_are_listed_separately(clf, reg):
    assert [m[0] for m in clf.run_all_models()] == CLF_DEFAULT
    assert [m[0] for m in reg.run_all_models()] == REG_DEFAULT
    assert MultiModelClassifier.default_models() == [
        "logistic_regression", "svm", "decision_tree", "knn", "naive_bayes", "random_forest",
        "gradient_boosting", "adaboost"]
    for extra in ("extra_trees", "hist_gradient_boosting", "mlp", "xgboost", "lightgbm", "catboost"):
        assert extra in MultiModelClassifier.available_models()
        assert extra not in MultiModelClassifier.default_models()
    assert "elastic_net" in MultiModelRegressor.available_models()


def test_include_all_runs_every_installed_classifier(clf):
    models = clf.run_all_models(include="all")
    assert not clf.failed_models_, clf.failed_models_
    installed = [s.name for s in CLASSIFIER_REGISTRY if s.available]
    assert [m[0] for m in models] == installed
    df = clf.show_tabular_report(models, return_df=True)
    assert df["Accuracy"].min() > 0.85  # every extra model is sensible on an easy dataset


def test_include_all_runs_every_installed_regressor(reg):
    models = reg.run_all_models(include="all")
    assert not reg.failed_models_, reg.failed_models_
    assert [m[0] for m in models] == [s.name for s in REGRESSOR_REGISTRY if s.available]
    df = reg.show_tabular_report(models, return_df=True)
    assert df["R2 Score"].min() > 0.5


def test_extras_can_be_picked_by_name_and_mixed_with_defaults(clf):
    models = clf.run_all_models(include=["random_forest", "extra_trees", "Linear SVM"])
    assert [m[0] for m in models] == ["Random Forest", "Extra Trees", "Linear SVM"]


def test_multiclass_with_string_labels_for_extras():
    iris = load_iris(as_frame=True)
    y = iris.target.map(dict(enumerate(iris.target_names)))
    c = MultiModelClassifier(iris.data, y, random_state=1, verbose=False)
    models = c.run_all_models(include=["extra_trees", "lda", "ridge_classifier", "linear_svm", "mlp"])
    assert not c.failed_models_, c.failed_models_
    assert set(c.predict(iris.data.head(10), model="Extra Trees")) <= set(iris.target_names)
    assert len(models) == 5


def test_missing_optional_package_gives_install_hint_and_is_skipped_by_all():
    fake = ModelSpec("ghost", "Ghost Booster", lambda ctx, **p: None, requires="no_such_package_xyz", default=False)

    class Ghosted(MultiModelClassifier):
        registry = ModelRegistry(list(CLASSIFIER_REGISTRY) + [fake])

    X, y = load_breast_cancer(return_X_y=True)
    g = Ghosted(X, y, verbose=False)
    assert not fake.available
    assert "ghost" in Ghosted.available_models() and "ghost" not in Ghosted.available_models(installed_only=True)
    names = [m[0] for m in g.run_all_models(include="all")]
    assert "Ghost Booster" not in names and not g.failed_models_
    assert g.run_all_models(include=["ghost"]) == []
    assert "pip install no_such_package_xyz" in g.failed_models_["Ghost Booster"]


@pytest.mark.parametrize("key", ["extra_trees", "ridge_classifier", "linear_svm"])
def test_class_weight_reaches_extra_models(key):
    from multimodel_analysis.core.registry import BuildContext
    ctx = BuildContext(random_state=0, n_jobs=1, n_train=100, class_weight="balanced")
    assert CLASSIFIER_REGISTRY.resolve(key).build(ctx).class_weight == "balanced"


def test_xgboost_and_lightgbm_work_through_predict_with_original_labels():
    pytest.importorskip("xgboost")
    pytest.importorskip("lightgbm")
    iris = load_iris(as_frame=True)
    y = iris.target.map(dict(enumerate(iris.target_names)))
    c = MultiModelClassifier(iris.data, y, random_state=0, verbose=False)
    c.run_all_models(include=["xgboost", "lightgbm"])
    assert not c.failed_models_, c.failed_models_
    for name in ("XGBoost", "LightGBM"):
        assert set(c.predict(iris.data.head(8), model=name)) <= set(iris.target_names)


def test_tuning_extras_including_target_scaled_wrappers(reg, clf):
    reg.tune_models(include=["linear_svr", "mlp", "elastic_net"], n_iter=2, cv=3)
    assert not reg.failed_models_, reg.failed_models_
    assert "C" in reg.best_params_["Linear SVR"]
    assert not any(k.startswith("regressor__") for k in reg.best_params_["Neural Network Regressor (MLP)"])
    clf.tune_models(include=["extra_trees", "lda"], n_iter=2, cv=3)
    assert not clf.failed_models_, clf.failed_models_


@pytest.mark.parametrize("package,key,name", [("xgboost", "xgboost", "XGBoost"), ("lightgbm", "lightgbm", "LightGBM"),
                                              ("catboost", "catboost", "CatBoost")])
def test_tuning_boosters_if_installed(clf, package, key, name):
    pytest.importorskip(package)
    clf.tune_models(include=[key], n_iter=2, cv=3, param_grids={key: {
        {"catboost": "iterations"}.get(key, "n_estimators"): [20, 40], "learning_rate": [0.1, 0.2]}})
    assert not clf.failed_models_, clf.failed_models_
    assert "learning_rate" in clf.best_params_[name]


def test_cross_validation_and_save_work_for_extras(clf, tmp_path):
    clf.run_all_models(include=["hist_gradient_boosting"])
    cv = clf.cross_validate_models(include=["hist_gradient_boosting", "mlp"], cv=3)
    assert len(cv) == 2 and not clf.cv_failed_
    path = clf.save_model(str(tmp_path / "hgb.joblib"), model="Hist Gradient Boosting")
    from multimodel_analysis import load_model
    X, _ = load_breast_cancer(return_X_y=True)
    assert np.array_equal(load_model(path).predict(X[:5]), clf.predict(X[:5], model="Hist Gradient Boosting"))


def test_comparison_plot_ranks_models_on_one_metric(clf, reg, tmp_path):
    clf_models = clf.run_all_models(include=["logistic_regression", "knn", "extra_trees", "mlp"])
    clf.plot_comparison(clf_models, save_path=str(tmp_path / "clf.png"), show_plot=False)
    clf.plot_comparison(clf_models, save_path=str(tmp_path / "clf_acc.png"), show_plot=False, metric="Accuracy")
    reg_models = reg.run_all_models(include=["linear_regression", "elastic_net", "huber"])
    reg.plot_comparison(reg_models, save_path=str(tmp_path / "reg.png"), show_plot=False, metric="RMSE")
    for name in ("clf.png", "clf_acc.png", "reg.png"):
        assert (tmp_path / name).stat().st_size > 3000
    with pytest.raises(ValueError):
        clf.plot_comparison(clf_models, show_plot=False, metric="Nope")
    with pytest.raises(ValueError):
        reg.plot_comparison(reg_models, show_plot=False, metric="Fit Time (s)")


def test_comparison_figure_handles_ties_missing_values_and_one_model():
    import matplotlib.pyplot as plt

    from multimodel_analysis.core.plotting import comparison_figure
    for names, values, lower in [(["a", "b", "c"], [0.9, 0.9, 0.9], False),
                                 (["a", "b", "c"], [np.nan, 0.5, -0.2], False),
                                 (["a", "b"], [3.0, 1.0], True),
                                 (["only"], [0.5], False)]:
        fig = comparison_figure(names, values, metric="m", title="t", lower_is_better=lower)
        labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        if lower and values == [3.0, 1.0]:
            assert labels[0] == "b"
        plt.close(fig)
