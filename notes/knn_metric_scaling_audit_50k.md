# KNN metric / scaling audit (50k train rows)

- **Dataset:** Small-HI
- **Rows:** 50000
- **k:** 15
- **Baseline for Jaccard:** baseline_cosine

## baseline_cosine

- feature_set=`edge_native+degree_fan` dims=12 metric=cosine scaling=none l2=True group_weights=False
- sim_mean=0.9966, sim_p50=1.0000, sim_p90=1.0000
- unique_neighbor_fraction=0.0666, hub1=0.0001, hub10=0.0006
- Jaccard vs baseline=nan
- endpoint sender=0.0161, pair=0.0007

## richer_v1_cosine_robust

- feature_set=`richer_v1` dims=67 metric=cosine scaling=none l2=True group_weights=True
- sim_mean=0.9932, sim_p50=1.0000, sim_p90=1.0000
- unique_neighbor_fraction=0.0665, hub1=0.0001, hub10=0.0006
- Jaccard vs baseline=0.7550
- endpoint sender=0.0140, pair=0.0000

## richer_v1_euclidean_robust

- feature_set=`richer_v1` dims=67 metric=euclidean scaling=none l2=False group_weights=False
- dist_mean=-0.4444, dist_p50=-0.0219, dist_p90=-0.0017
- unique_neighbor_fraction=0.0665, hub1=0.0001, hub10=0.0005
- Jaccard vs baseline=0.7608
- endpoint sender=0.0145, pair=0.0000

## richer_v1_euclidean_robust_weighted

- feature_set=`richer_v1` dims=67 metric=euclidean scaling=none l2=False group_weights=True
- dist_mean=-0.4264, dist_p50=-0.0231, dist_p90=-0.0020
- unique_neighbor_fraction=0.0665, hub1=0.0001, hub10=0.0005
- Jaccard vs baseline=0.7541
- endpoint sender=0.0142, pair=0.0000

## richer_v1_cosine_no_static

- feature_set=`richer_v1_no_static` dims=51 metric=cosine scaling=none l2=True group_weights=True
- sim_mean=0.9958, sim_p50=1.0000, sim_p90=1.0000
- unique_neighbor_fraction=0.0666, hub1=0.0001, hub10=0.0005
- Jaccard vs baseline=0.6657
- endpoint sender=0.0136, pair=0.0000

## richer_v1_causal_only_cosine

- feature_set=`richer_v1_causal_only` dims=48 metric=cosine scaling=none l2=True group_weights=True
- sim_mean=0.9959, sim_p50=1.0000, sim_p90=1.0000
- unique_neighbor_fraction=0.0666, hub1=0.0001, hub10=0.0005
- Jaccard vs baseline=0.6656
- endpoint sender=0.0137, pair=0.0000
