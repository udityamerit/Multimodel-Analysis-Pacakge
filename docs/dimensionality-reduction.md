# Dimensionality reduction

`MultiModelReducer` squeezes many columns into a few (usually 2, to draw a map). It runs 7 standard
methods (12 in total), and scores how faithfully each one keeps the structure of your data.

```python
from sklearn.datasets import load_digits
from multimodel_analysis import MultiModelReducer

digits = load_digits()
red = MultiModelReducer(digits.data[:1000], y=digits.target[:1000], n_components=2, random_state=0)

models = red.run_all_models()
red.show_tabular_report(models)
red.plot_comparison(models)
red.plot_embeddings(models, top=4)    # 2-D maps, coloured by y if you gave it
```

`y` is optional here; it only colours the maps and feeds `Class Separation`.

## Reading the scores

| Column | Question it answers | Better |
|---|---|---|
| `Trustworthiness` | are points that look close on the map *really* close in the original data? | higher (max 1) |
| `Continuity` | are points that were close originally still close on the map? | higher (max 1) |
| `Global Structure` | do far-apart rows stay far apart? | higher |
| `Variance Kept (%)` | share of the data's spread that survives (PCA and other reversible methods) | higher |
| `Class Separation` | how well classes are separated on the map (needs `y`) | higher |
| `Can Transform New Data` | `yes` if the method can map unseen rows | — |

Local scores favour t-SNE and Isomap; PCA keeps the global shape and is the easiest to explain.
Pick by purpose: **PCA** to interpret or to feed another model, **t-SNE / UMAP** for a picture.

## How many components do I need?

```python
k = red.find_n_components(0.95)     # smallest number of PCA components keeping 95% of the variance
red.plot_variance()                 # scree plot: variance per component + running total
```

## What do the components mean? (PCA)

```python
loads = red.get_loadings("PCA")
loads["Component 1"].abs().sort_values(ascending=False).head(5)   # columns that drive component 1
```

## Use the result

```python
red.get_embedding("PCA")                    # new columns: Component 1, Component 2, ...
red.transform(new_rows, model="PCA")        # map unseen rows
```

PCA, Kernel PCA, FastICA, Factor Analysis, Truncated SVD, Isomap, LLE and Random Projection (and UMAP,
if installed) can transform new rows. **t-SNE, Spectral Embedding and MDS cannot** (they only embed the data they
were fitted on), so they raise a clear `TypeError`.

Reduce first, then cluster:

```python
emb = red.get_embedding("PCA")
MultiModelClusterer(emb, n_clusters=3).run_all_models()
```

## Notes

* Constant columns are tolerated (FastICA ignores them).
* t-SNE, Kernel PCA, Isomap, LLE, Spectral Embedding and MDS are skipped on large data (limits: t-SNE and LLE 10,000 rows, Kernel
  PCA, Isomap and Spectral Embedding 8,000, MDS 1,500) unless `allow_large=True`.
* UMAP needs the extra: `pip install "multimodel-analysis[umap]"`.

## Notebook

[`examples/phase7_anomaly_reduction_explain.ipynb`](../examples/phase7_anomaly_reduction_explain.ipynb)
