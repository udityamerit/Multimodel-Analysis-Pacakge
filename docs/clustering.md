# Clustering

`MultiModelClusterer` finds groups in data that has **no target column**. It runs 7 standard methods
(14 in total), scores each one, and shows the groups on a 2-D map.

```python
from sklearn.datasets import make_blobs
import pandas as pd
from multimodel_analysis import MultiModelClusterer

X, y = make_blobs(n_samples=450, centers=3, random_state=11)
X = pd.DataFrame(X, columns=["feature 1", "feature 2"])

cl = MultiModelClusterer(X, n_clusters=3, random_state=0)
models = cl.run_all_models()
cl.show_tabular_report(models)
cl.plot_comparison(models)     # ranked bars
cl.plot_clusters(models)       # scatter plots of the best methods
```

## How many clusters?

* Pass a number: `n_clusters=3`.
* Or let the tool search: `n_clusters="auto"` picks *k* by Silhouette.
* Or look first:

```python
cl.find_optimal_k()      # table of k vs Inertia and Silhouette
cl.plot_k_search()       # elbow curve + silhouette curve, best k marked
```

Methods that decide the number themselves (DBSCAN, HDBSCAN, OPTICS, Mean Shift, Affinity Propagation)
ignore `n_clusters`; their settings are chosen automatically and shown in `cl.auto_params_`.
Rows they leave unassigned are **noise** and get label `-1`.

## Reading the scores

| Column | Meaning | Better |
|---|---|---|
| `Silhouette` | how well each row fits its own cluster vs the next one (-1 to 1) | higher |
| `Calinski-Harabasz` | between-cluster spread vs within-cluster spread | higher |
| `Davies-Bouldin` | how similar clusters are to each other | **lower** |
| `Clusters`, `Noise Points` | what the method found | — |
| `ARI`, `NMI`, `V-measure`, `Homogeneity`, `Completeness` | agreement with *known* labels; only filled when you pass `y=` | higher |

No true labels? Trust Silhouette and look at the plot. If you do have labels (for example to test
the tool on a dataset like iris), pass them as `y=` and sort by `ARI`:

```python
cl = MultiModelClusterer(iris.data, y=iris.target, n_clusters=3)
cl.show_tabular_report(cl.run_all_models(include="all"), sort_by="ARI")
```

## Using the result

```python
cl.assign_clusters("KMeans")      # your data + a `cluster` column
cl.get_labels("KMeans")           # just the labels (numpy array)
cl.cluster_profile("KMeans")      # size and average of every feature per cluster, in your own units
cl.plot_silhouette("KMeans")      # per-row silhouette, grouped by cluster
```

`cluster_profile` is the quickest way to *name* your groups ("young, low spend" / "older, high spend").

## Label new rows

```python
cl.predict(new_rows, model="KMeans")
```

Works for KMeans, MiniBatch KMeans, Gaussian Mixture, Birch, Bisecting KMeans, Mean Shift and Affinity
Propagation. The others (the three Agglomerative variants, Spectral, DBSCAN, HDBSCAN, OPTICS) cannot place
unseen points and raise a clear `TypeError` instead of guessing. With no `model=`, the best-ranked model that *can* predict is used.

## Tuning

```python
cl.tune_models(include=["kmeans", "gaussian_mixture", "dbscan"])
cl.tuning_results_          # best settings and scores
cl.best_params_
cl.tune_models(include=["dbscan"], scoring="Calinski-Harabasz",
               param_grids={"dbscan": {"eps": [0.2, 0.3, 0.5], "min_samples": [4, 8]}})
```

## Big data

Methods that need memory proportional to *n²* (Spectral, Agglomerative, Mean Shift, Affinity
Propagation, OPTICS) are **skipped above a size limit** and listed in `failed_models_`. Pass
`allow_large=True` to run them anyway. The fast methods (KMeans, MiniBatch KMeans, Birch, Gaussian
Mixture, DBSCAN) always run.

## Messy data

```python
cc = MultiModelClusterer(customers, n_clusters=2, preprocess=True)
```

Missing values are filled and text columns are encoded before clustering.

## Notebook

[`examples/phase6_clustering.ipynb`](../examples/phase6_clustering.ipynb)
