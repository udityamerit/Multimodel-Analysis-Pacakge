"""Helpers shared by every analyser (supervised and unsupervised): warning capture, model lookup,
saving plots / reports and listing failed models. Needs ``registry``, ``verbose``, ``warnings_``,
``failed_models_``, ``estimators_`` and ``_last_df_report`` on the class using it."""
import contextlib
import io
import os
import tempfile
import warnings
from typing import Any, Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd

from .registry import ModelRegistry
from .utils import _VISIBLE_WARNINGS, _safe_print, save_report


class _ReportingMixin:
    registry: ModelRegistry = ModelRegistry()

    # ------------------------------------------------------------------ helpers
    def _warn(self, label: str, message: str, show: bool = True) -> None:
        """Records a warning for ``label`` (deduplicated) and optionally prints it."""
        bucket = self.warnings_.setdefault(label, [])
        if message not in bucket:
            bucket.append(message)
            if show and self.verbose:
                _safe_print(f"[WARNING] {label}: {message}")

    @contextlib.contextmanager
    def _track_warnings(self, label: str):
        """Captures warnings raised inside the block (instead of silencing them) and records them."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            yield
        for w in caught:
            text = str(w.message).strip()
            first_line = text.splitlines()[0] if text else ""
            self._warn(label, f"{w.category.__name__}: {first_line}",
                       show=w.category.__name__ in _VISIBLE_WARNINGS)

    @classmethod
    def available_models(cls, installed_only: bool = False) -> List[str]:
        """Keys of the built-in models, usable in ``run_model`` / ``include`` / ``exclude``.

        Includes the extra models that ``run_all_models()`` skips by default; with ``installed_only=True``
        models whose optional package (xgboost, lightgbm, catboost) is missing are left out.
        """
        return [s.key for s in cls.registry if s.available or not installed_only]

    @classmethod
    def default_models(cls) -> List[str]:
        """Keys of the models ``run_all_models()`` runs when ``include`` is not given."""
        return [s.key for s in cls.registry if s.default]

    def get_model(self, name: str) -> Any:
        """Returns the FITTED estimator for ``name`` (display name, as shown in the report)."""
        if name in self.estimators_:
            return self.estimators_[name]
        try:
            name = self.registry.resolve(name).name
            if name in self.estimators_:
                return self.estimators_[name]
        except KeyError:
            pass
        raise KeyError(f"No fitted model called '{name}'. Fitted so far: {list(self.estimators_)}")

    @staticmethod
    def _save_and_show(fig: plt.Figure, save_path: Optional[str] = None, show_plot: bool = True):
        """Helper method to handle saving and displaying matplotlib figures."""
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            _safe_print(f"[INFO] Plot saved to '{save_path}'.")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    def save_report(self, df_or_filepath: Optional[Union[pd.DataFrame, str]] = None, filepath: Optional[str] = None) -> None:
        """
        Saves tabular report to CSV/Excel/HTML/JSON.
        Can be called as:
          - save_report(df, "metrics.csv")           [standalone function]
          - model.save_report(df, "metrics.csv")     [method on classifier/regressor instance]
          - model.save_report("metrics.csv")         [method using auto-stored report DataFrame]
          - model.save_report()                      [method using auto-stored report DataFrame and default path 'report.csv']
        """
        if df_or_filepath is None:
            target_df = getattr(self, '_last_df_report', None)
            target_path = filepath if filepath else "report.csv"
            if target_df is None:
                _safe_print("[WARNING] No previous tabular report DataFrame found. Run show_tabular_report() first or pass a DataFrame.")
                return
        elif isinstance(df_or_filepath, str):
            target_path = df_or_filepath
            target_df = getattr(self, '_last_df_report', None)
            if target_df is None:
                _safe_print("[WARNING] No previous tabular report DataFrame found. Run show_tabular_report() first or pass a DataFrame.")
                return
        else:
            target_df = df_or_filepath
            target_path = filepath if filepath else "report.csv"

        save_report(target_df, target_path)

    # ------------------------------------------------------------------ HTML report
    def _report_plots(self) -> List[Tuple[str, Callable[[str], Any]]]:
        """(heading, function(save_path)) pairs drawn into the HTML report; each class overrides this."""
        return []

    def _report_subtitle(self) -> str:
        return type(self).__name__

    def save_html_report(self, path: str = "report.html", title: Optional[str] = None) -> str:
        """Writes ONE self-contained HTML file: the results table, the plots and any warnings / failed models.

        Run ``show_tabular_report()`` first; the file opens in any browser and can be e-mailed as it is
        (plots are embedded, nothing else is needed). Returns the path.
        """
        from .htmlreport import build_html_report

        df = getattr(self, "_last_df_report", None)
        if df is None or not len(df):
            raise ValueError("Nothing to report yet: run the models and call show_tabular_report() first.")
        plots: List[Tuple[str, bytes]] = []
        notes: List[Tuple[str, List[str]]] = []
        problems: List[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            for i, (heading, draw) in enumerate(self._report_plots()):
                png_path = os.path.join(tmp, f"plot{i}.png")
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        draw(png_path)
                    with open(png_path, "rb") as fh:
                        plots.append((heading, fh.read()))
                except Exception as exc:  # a plot that cannot be drawn must not lose the whole report
                    problems.append(f"'{heading}' could not be drawn: {exc}")
        if self.failed_models_:
            notes.append(("Models that failed or were skipped", [f"{k}: {v}" for k, v in self.failed_models_.items()]))
        warn_lines = [f"{k}: {m}" for k, msgs in self.warnings_.items() for m in msgs]
        notes.append(("Warnings", warn_lines + problems))
        from .. import __version__

        html_text = build_html_report(
            title or f"{type(self).__name__} report", self._report_subtitle(), df,
            str(df.iloc[0]["Model"]), plots, notes, __version__)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html_text)
        _safe_print(f"[INFO] HTML report saved to '{path}'.")
        return path

    def _print_failures(self) -> None:
        if self.failed_models_:
            _safe_print("Models that FAILED (not in the table above):")
            for name, reason in self.failed_models_.items():
                _safe_print(f"   - {name}: {reason}")
            _safe_print(f"{'='*60}\n")
