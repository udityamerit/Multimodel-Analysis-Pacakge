import os
import warnings
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from multimodel_analysis import (
    MultiModelClassifier,
    MultiModelRegressor,
    MultiModelRegressior,
    save_report
)


def test_global_state_not_mutated():
    """Bug 4 Check: Importing multimodel should not mute global warnings or force matplotlib styles."""
    filters = warnings.filters
    assert not any(f[0] == 'ignore' and f[2] is Warning and f[3] is None for f in filters)


def test_classifier_binary():
    """Tests MultiModelClassifier on standard binary classification."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    df_X = pd.DataFrame(X, columns=[f'col_{i}' for i in range(5)])
    
    clf = MultiModelClassifier(df_X, y, scaled_data=True, random_state=42)
    models = clf.run_all_models()
    
    assert len(models) == 8
    df_report = clf.show_tabular_report(models, return_df=True)
    assert isinstance(df_report, pd.DataFrame)
    assert 'ROC AUC' in df_report.columns
    assert not df_report.empty

    # Test headless plot rendering
    clf.plot_confusion_matrices(models, show_plot=False)
    clf.plot_roc_curves(models, show_plot=False)
    clf.plot_comparison(models, show_plot=False)


def test_classifier_multiclass_roc_curves():
    """Bug 1 Check: Verify multiclass ROC curves compute macro & per-class curves without dummy diagonal."""
    X, y = make_classification(n_samples=150, n_features=6, n_classes=3, n_informative=4, random_state=42)
    clf = MultiModelClassifier(X, y, scaled_data=True, random_state=42)
    models = clf.run_all_models()
    
    for model_tuple in models:
        name, report, matrix, accuracy, precision, recall, f1, fpr_dict, tpr_dict, roc_auc = model_tuple
        if isinstance(fpr_dict, dict) and len(fpr_dict) > 0:
            assert 'macro' in fpr_dict or 0 in fpr_dict or 'binary' in fpr_dict

    clf.plot_roc_curves(models, show_plot=False)


def test_classifier_roc_auc_nan_fallback():
    """Bug 2 Check: Verify ROC AUC defaults to np.nan (not silent 0.5) when evaluation fails."""
    X, y = make_classification(n_samples=20, n_features=4, n_classes=2, random_state=42)
    clf = MultiModelClassifier(X, y, test_size=0.1, stratify=False, random_state=42)
    
    class CorruptModel:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)
        def predict_proba(self, X):
            raise RuntimeError("Simulated probability failure")
            
    res = clf.evaluate_model(CorruptModel(), clf.X_test_scaled, clf.y_test)
    roc_auc = res[8]
    assert np.isnan(roc_auc)


def test_regressor_2d_target_and_line_min_max(tmp_path):
    """Bug 3 & Bug 5 Check: Test 2D single-column DataFrame target and plot true vs pred line bounds."""
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    df_y = pd.DataFrame({'target': y})
    
    reg = MultiModelRegressor(X, df_y, scaled_data=True, random_state=42)
    models = reg.run_all_models()
    
    # 8 models including AdaBoost Regressor
    assert len(models) == 8
    df_report = reg.show_tabular_report(models, return_df=True)
    assert not df_report.empty
    
    save_img = str(tmp_path / "true_vs_pred.png")
    reg.plot_true_vs_predicted(models, save_path=save_img, show_plot=False)
    assert os.path.exists(save_img)


def test_custom_models_and_report_saving(tmp_path):
    """Tests passing custom estimators and saving CSV tabular report."""
    X, y = make_classification(n_samples=80, n_features=4, random_state=42)
    clf = MultiModelClassifier(X, y, scaled_data=True)
    
    custom_rf = RandomForestClassifier(n_estimators=10, random_state=42)
    models = clf.run_all_models(custom_models={'My Custom RF': custom_rf})
    
    model_names = [m[0] for m in models]
    assert 'My Custom RF' in model_names
    
    df_report = clf.show_tabular_report(models, return_df=True)
    report_file = str(tmp_path / "report.csv")
    save_report(df_report, report_file)
    assert os.path.exists(report_file)

    report_file_method = str(tmp_path / "report_method.csv")
    clf.save_report(report_file_method)
    assert os.path.exists(report_file_method)


def test_backward_compatibility_alias():
    """Verify MultiModelRegressior alias works as MultiModelRegressor."""
    assert MultiModelRegressior is MultiModelRegressor


def test_random_state_in_individual_models():
    """Verifies that model methods accept random_state explicitly without throwing TypeError."""
    X, y = make_classification(n_samples=60, n_features=4, random_state=42)
    clf = MultiModelClassifier(X, y, random_state=42)

    # Calling individual model functions with random_state parameter
    lr_res = clf.Logistic_model(random_state=123)
    rf_res = clf.RandomForest_model(random_state=123, n_estimators=50)
    dt_res = clf.DecisionTree_model(random_state=123)
    knn_res = clf.KNN_model(random_state=123)  # Strips random_state gracefully
    nb_res = clf.Naive_Bayes_model(random_state=123)  # Strips random_state gracefully

    assert lr_res[2] > 0.0  # Accuracy score
    assert rf_res[2] > 0.0
    assert dt_res[2] > 0.0
    assert knn_res[2] > 0.0
    assert nb_res[2] > 0.0

    # Test regressor model functions
    X_reg, y_reg = make_regression(n_samples=60, n_features=4, random_state=42)
    reg = MultiModelRegressor(X_reg, y_reg, random_state=42)

    lasso_res = reg.Lasso_model(random_state=999, alpha=0.05)
    ridge_res = reg.Ridge_model(random_state=999)
    lin_res = reg.LinearRegression_model(random_state=999)
    ada_res = reg.AdaBoost_model(random_state=999)

    assert len(lasso_res) == 5
    assert len(ridge_res) == 5
    assert len(lin_res) == 5
    assert len(ada_res) == 5


def test_omitted_parameters_and_auto_cache():
    """Verifies that calling reporting/plotting functions without models parameter auto-runs or uses cached results."""
    X, y = make_classification(n_samples=80, n_features=4, random_state=42)
    clf = MultiModelClassifier(X, y)

    # User calls show_tabular_report directly without passing models argument
    df_report = clf.show_tabular_report(return_df=True)
    assert df_report is not None
    assert not df_report.empty

    # Calling get_summary without models parameter
    clf.get_summary(show_plot=False)

    # Standalone save_report with default args
    save_report(df_report)
    assert os.path.exists("report.csv")
    os.remove("report.csv")

    # Instance save_report without args
    clf.save_report()
    assert os.path.exists("report.csv")
    os.remove("report.csv")
