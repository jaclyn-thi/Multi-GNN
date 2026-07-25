# D+ partial fine-tune smoke

**PASS**

Bounded GPU smoke for the locked partial fine-tune protocol ([`final_dplus_experiment_preflight.md`](final_dplus_experiment_preflight.md)). Full fine-tune job was **not** submitted.

## Locked inputs

- Device: `cuda:0`
- Source ckpt: `/orcd/home/002/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/saved-models/checkpoint_gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_8192neg_queue0_40ep_seed2.tar`
- Source sha256: `a320920141f585c5825cbd63ce760a845fb434a9b162d4c87270dc72b0442b87`
- Loaded epoch: 40
- MLP init path: **from_scratch_identical_recipe** (18678029 weights found: False)
- Note: Job 18678029 did not persist PaperStyleMLP weights; classifier is initialized from scratch with seed=2 / lr=1e-3 / BCE recipe.
- Stack dim: 227 (H+X+TF)
- TF leakage audit ok: True
- Pre-3h equivalence ok: True (matched=16378, max_abs=9.5367431640625e-06)

## Stage 1 (frozen encoder / classifier warmup)

- Trainable encoder params: []
- Frozen encoder tensors: 58
- Optimizer groups: `[{'index': 0, 'name': 'classifier', 'lr': 0.001, 'n_params': 29313}]`
- Warmup losses: [0.03295207401017379]
- Encoder param deltas during warmup (must be empty): {}
- Val after warmup: `{'auprc': 0.18839350864648627, 'auroc': 0.9652249685325058, 'n': 965462.0, 'f1_at_selected': 0.2784810126582279, 'threshold': 0.060000000000000005}`
- Test blocked during selection: True

## Stage 2 (partial unfreeze)

- Trainable encoder n_params: 61776
- Reverse trainable count: 10
- Optimizer groups: `[{'index': 0, 'name': 'classifier', 'lr': 0.001, 'n_params': 29313}, {'index': 1, 'name': 'encoder_final_block', 'lr': 0.0001, 'n_params': 61776}]`
- Partial stats (loss / grad_norm / steps): `{'loss': 0.005866331164725125, 'grad_norm': 0.01836262285034112, 'steps': 2.0}`
- Trainable deltas: 18 (examples: `{'convs.1.node__to__node.nn.0.weight': 0.0002000797539949417, 'convs.1.node__to__node.nn.0.bias': 0.00019869208335876465, 'convs.1.node__to__node.nn.2.weight': 0.00020007789134979248, 'convs.1.node__to__node.nn.2.bias': 1.4901161193847656e-08, 'convs.1.node__to__node.lin.weight': 0.00020009279251098633, 'convs.1.node__to__node.lin.bias': 0.00019993633031845093, 'convs.1.node__rev_to__node.nn.0.weight': 0.00020010769367218018, 'convs.1.node__rev_to__node.nn.0.bias': 0.0001990795135498047}`)
- Frozen convs.0 deltas (must be empty): `{}`
- Smoke val metrics: `{'auprc': 0.17582227319882432, 'auroc': 0.9598672855137063, 'n': 965462.0, 'f1_at_selected': 0.27387387387387385, 'threshold': 0.08}`
- Checkpoint save/reload ok: True

## Resources / projection

- Peak GPU GiB: 5.551838397979736
- Sec/batch (partial): 0.6793840609898325
- Train/val batches: 397 / 118
- Projected full runtime h: 1.7620312209796813
- 6h safe: True → single_6h_job
- Wall s: 658.8874925400014

## Full-job command (NOT submitted)

```bash
python scripts/run_dplus_partial_finetune.py --seed 2 --warmup_epochs 5 --max_epochs 20 --early_stop_patience 5 --loader_num_workers 0 --unique_name dplus_partial_finetune_hxxtf_seed2 --init_checkpoint /orcd/home/002/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/saved-models/checkpoint_gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_8192neg_queue0_40ep_seed2.tar --output_dir saved-models/dplus_partial_finetune_hxxtf_seed2
```

