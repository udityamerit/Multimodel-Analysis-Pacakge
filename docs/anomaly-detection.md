# Anomaly detection

`MultiModelAnomalyDetector` finds rows that look unlike the rest: faulty sensors, fraud, typos,
unusual customers. It needs **no labels**. It runs 6 standard detectors (7 in total) and tells you
where they agree.

```python
import numpy as np, pandas as pd
from multimodel_analysis import MultiModelAnomalyDetector

rng = np.random.RandomState(0)
normal = rng.randn(300, 3)
faulty = rng.randn(15, 3) * 0.5 + [4, -4, 4]
sensors = pd.DataFrame(np.vstack([normal, faulty]), columns=["temperature", "pressure", "vibration"])

det = MultiModelAnomalyDetector(sensors, contamination=0.05, random_state=0)
models = det.run_all_models()
det.show_tabular_report(models)
det.plot_anomalies(models)
```

**Convention:** `1` = anomaly, `0` = normal.

## `contamination`: how many rows to flag

Anomaly detection always needs a cut-off. `contamination` is *the share of rows you expect to be
unusual* (between 0 and 0.5). Every detector then flags exactly the top share of rows by anomaly
score, so the models are comparable.

* Default: the share of `1`s in `y` if you pass labels, otherwise **5%**.
* Think about your data: fraud might be 0.5%, a sensor fault 2%.

## Reading the report

| Column | Meaning |
|---|---|
| `Anomalies`, `Anomaly Rate (%)` | how many rows the model flagged |
| `Agreement` | how closely this model's *ranking* of rows matches the other models (1 = identical). A consensus hint, **not** accuracy |
| `Precision`, `Recall`, `F1 Score`, `ROC AUC`, `Avg Precision` | only when you pass `y=` (1 = known anomaly) |

Without labels the table is ranked by `Agreement`; with labels, by `F1 Score`.

## Look at the actual rows

```python
det.top_anomalies(10)               # the 10 strangest rows with their scores
det.assign_anomalies()              # all rows + `anomaly` (0/1) and `anomaly_score` columns
det.get_labels("Isolation Forest")  # 0/1 array
det.get_scores("Isolation Forest")  # higher = stranger
det.plot_scores()                   # score histogram with the cut-off line
```

## Consensus: trust rows many models agree on

```python
votes = det.consensus()             # adds `votes` and `consensus`
votes[votes["consensus"]]           # flagged by more than half of the models
det.consensus(min_votes=5)          # stricter
```

## Judge new rows

```python
det.predict(new_rows, model="Isolation Forest")      # 1 = anomaly
det.score_new(new_rows, model="Isolation Forest")    # the raw scores
```

The cut-off learned on the training data is reused, so scores are comparable over time. Local
Outlier Factor, KNN Distance and the others work too; `det.save_model(path)` keeps the cut-off.

## Mixed and messy data

```python
od = MultiModelAnomalyDetector(orders, contamination=0.02, preprocess=True)
```

Text columns (country, category) are encoded and missing numbers filled. Add your own detector with
`custom_models={"My Forest": IsolationForest(n_estimators=300)}`.

## Big data

One-Class SVM (above 20,000 rows) and Elliptic Envelope (above 100,000) are skipped unless
`allow_large=True`; skipped models are listed in `failed_models_`.

## Notebook

[`examples/phase7_anomaly_reduction_explain.ipynb`](../examples/phase7_anomaly_reduction_explain.ipynb)
