# k-Means Clustering

The k-means objective recast as continuous global optimisation. A fixed
dataset of 75 points in the unit square forms three well-separated blobs
(≈25 points each). Place `k = 3` centroids to minimise the within-cluster
sum of squares (WCSS): every point contributes the squared Euclidean
distance to its nearest centroid.

The decision vector is the three centroids' `(x, y)` coordinates, so
`N_DIM = 6`. The global optimum drops one centroid into each blob, giving
a low SSE.

## What this stresses

- **Multimodality.** This is the textbook reason k-means is run with
  random restarts. Many local minima exist — the most common is a
  "2-1" split where two centroids carve up a single blob while the third
  is left to cover the other two. An optimiser stuck there reports a
  visibly higher SSE than one that found the one-per-blob assignment.

- **Non-smoothness.** Each point is assigned to its nearest centroid, so
  the assignment flips discontinuously across the perpendicular bisectors
  between centroids. The WCSS surface is piecewise-quadratic with creases,
  which frustrates methods that assume a smooth landscape.

- **Symmetry / permutation.** The three centroids are interchangeable, so
  every global optimum has `3! = 6` equivalent relabellings — extra
  basins that a search has to navigate without help.

## Running

```bash
python -m example_applications.kmeans_clustering.run
```

Output is a small comparison table of optimiser → best SSE → centroid
coordinates. Optimisers that escape the local minima land one centroid
per blob and report a low SSE; those that don't reveal the classic
k-means restart pathology.
