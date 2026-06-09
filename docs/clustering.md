# Clustering (Module 3)

Groups peptides by condition profile similarity on proteins where Module 2 found at least one discordant peptide. The default path uses Ward hierarchical clustering on Euclidean distances between median condition profiles, then a dendrogram cut. `assign_proteoforms()` maps clusters to differential proteoform IDs (dPF).

## Entry point

```python
from proteoforge import (
    Config,
    assign_proteoforms,
    prepare_from_parquet,
    run_cluster,
    run_discordance,
)

config = Config.from_yaml_path("config.yaml")
dataset = prepare_from_parquet("peptides.parquet", config)
discordance = run_discordance(dataset)
clusters = run_cluster(dataset, discordance)
mapping = assign_proteoforms(dataset, discordance, clusters)

clusters.table
mapping.table
mapping.n_differential_peptides
```

Both functions take the same `PreparedDataset` and `DiscordanceResult` as [Discordance](discordance.md). The frozen `Config` on all three handoff objects must match.

Optional arguments for `run_cluster()`:

- `n_jobs`: overrides `config.n_jobs` for this call
- `show_progress=True`: peptide-weighted tqdm bar (auto-suppressed in CI or when `TQDM_DISABLE` is set)

## Scope

Clustering runs on proteins with at least one discordant peptide. For each such protein, **all peptides on that protein** enter the profile matrix and linkage, not discordant-only subsets. Non-discordant peptides that co-move with a discordant sibling can land in the same cluster.

Proteins with no discordant peptides are skipped. Their peptides are absent from `ClusterResult.table` and receive `dpf_id = 0` after assignment.

Degenerate cases on a discordant protein:

- **One peptide:** `cluster_id = 1` without linkage
- **Fewer than `cluster_min_peptides` peptides:** single cluster without linkage (default `cluster_min_peptides` is `2`)

## Method

Per discordant protein:

1. **Profiles:** median `intensity_normalized` per `(peptide_id, condition)`; matrix shape `(n_peptides, n_conditions)` with columns in `condition_levels` order
2. **Distance:** condensed Euclidean distances between peptide rows
3. **Linkage:** Ward (only supported linkage in this release)
4. **Cut:** strategy from `config.cut` (default `hybrid_outlier`)

### Cut strategies

- **`hybrid_outlier`** (default): choose cluster count `k` from the merge-height curve (second difference peak on linkage distances), clamped to `[cluster_min_clusters, cluster_max_clusters]`, then compute per-peptide silhouette scores on the condensed distance matrix. Peptides with silhouette below `hybrid_outlier_threshold` (default `0.05`) move to a new singleton cluster when their current cluster has more than one member
- **`dynamic_tree`:** dendrogram k-selection only (same k rule as the first step of hybrid)
- **`fixed_height`:** cut to `max(cluster_min_clusters, fixed_n_clusters)` clusters, capped by peptide count

Set `cut`, `linkage`, and related fields in [Configuration](config.md). Only `linkage: ward` is accepted at runtime.

## Parallelism

Clustering parallelizes at protein grain when `n_jobs > 1`, using the same spawn pool pattern as discordance.

- `n_jobs=-1` (default) maps to `min(8, cpu_count // 2)`
- `n_jobs=1` runs serially (useful for debugging)
- Serial fallback applies in notebooks or when a worker pool cannot start

## Cluster output

`ClusterResult.table` columns:

- `protein_id`, `peptide_id`: peptide keys
- `cluster_id`: 1-based per protein after the cut
- `cut_method`, `linkage_method`: snapshots from the run

`clusters.metadata` includes `n_discordant_proteins`, `n_clustered_peptides`, parallelism fields, and the `linkage` / `cut` values used.

**Join on keys, not row index** (same rule as discordance):

```python
joined = dataset.peptides.join(
    clusters.table,
    on=["protein_id", "peptide_id"],
    how="left",
)
```

## dPF assignment

`assign_proteoforms(prepared, discordance, cluster)` returns `ProteoformMappingResult` with one row per `(protein_id, peptide_id)` in the prepared scope.

`mapping.table` columns:

- `protein_id`, `peptide_id`
- `is_discordant`
- `cluster_id`: null for peptides on proteins without discordance
- `dpf_id`: differential proteoform ID

Assignment is cluster-first: every peptide in a cluster inherits that cluster's dPF.

| Cluster | `dpf_id` |
| ------- | -------- |
| No discordant member | `0` (`dPF_0`, canonical) |
| One peptide, discordant | `-1` (`dPF_-1`, singleton differential) |
| Two or more peptides, at least one discordant | `1`, `2`, … (`dPF_1`, …; numbered per protein among qualifying clusters, ordered by `cluster_id`) |

Peptides on proteins with no discordant members always receive `dpf_id = 0` and `cluster_id = null`.

`mapping.metadata` reports counts by dPF class (`n_singleton_peptides`, `n_differential_peptides`, `dpf_counts`, and related fields).

## Related pages

- [Discordance](discordance.md): upstream `is_discordant` flags
- [PreparedDataset](prepared-dataset.md): `intensity_normalized` used for profiles
- [Configuration](config.md): `linkage`, `cut`, cluster limits, hybrid threshold
