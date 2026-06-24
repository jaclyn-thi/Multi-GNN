# KNN mutual / hub-filter audit (50k train rows)

- **Dataset:** Small-HI
- **Rows:** 50000
- **k:** 15

Label enrichment is analysis-only.

## baseline_edge_native_degree_fan (`edge_native+degree_fan`)

### ordinary_topk: sim_mean=0.9966, uniq=0.0666, hub1=0.0001
- endpoint: sender=0.0161, pair=0.0007

### mutual_knn: sim_mean=0.9979, uniq=0.0801, hub1=0.0000
- mutual coverage=0.9971, avg mutual neighbors=12.46
- endpoint: sender=0.0152, pair=0.0008

### Hub filters (vs ordinary)
- `hub_filter_0.001`: uniq=0.0666, hub1=0.0001, lost_all=0.0000, banned=0
- `hub_filter_0.005`: uniq=0.0666, hub1=0.0001, lost_all=0.0000, banned=0
- `hub_filter_0.01`: uniq=0.0666, hub1=0.0001, lost_all=0.0000, banned=0

## richer_v1_one_hot (`richer_v1`)

### ordinary_topk: sim_mean=0.9932, uniq=0.0665, hub1=0.0001
- endpoint: sender=0.0140, pair=0.0000

### mutual_knn: sim_mean=0.9969, uniq=0.0814, hub1=0.0000
- mutual coverage=0.9932, avg mutual neighbors=12.20
- endpoint: sender=0.0129, pair=0.0000

### Hub filters (vs ordinary)
- `hub_filter_0.001`: uniq=0.0665, hub1=0.0001, lost_all=0.0000, banned=0
- `hub_filter_0.005`: uniq=0.0665, hub1=0.0001, lost_all=0.0000, banned=0
- `hub_filter_0.01`: uniq=0.0665, hub1=0.0001, lost_all=0.0000, banned=0

## richer_v1_no_pair (`richer_v1_no_pair`)

### ordinary_topk: sim_mean=0.9932, uniq=0.0665, hub1=0.0001
- endpoint: sender=0.0141, pair=0.0000

### mutual_knn: sim_mean=0.9970, uniq=0.0814, hub1=0.0000
- mutual coverage=0.9932, avg mutual neighbors=12.20
- endpoint: sender=0.0130, pair=0.0000

### Hub filters (vs ordinary)
- `hub_filter_0.001`: uniq=0.0665, hub1=0.0001, lost_all=0.0000, banned=0
- `hub_filter_0.005`: uniq=0.0665, hub1=0.0001, lost_all=0.0000, banned=0
- `hub_filter_0.01`: uniq=0.0665, hub1=0.0001, lost_all=0.0000, banned=0
