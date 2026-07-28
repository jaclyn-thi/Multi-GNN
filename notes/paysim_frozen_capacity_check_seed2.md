# PaySim frozen capacity check (seed 2)

> Exploratory / post-hoc. `table_eligible=false`. Validation only; test not accessed.
> Twin: `results/diagnostics/paysim_frozen_capacity_check_seed2.json`

## Coverage

- Train: n=3792812, positives=3175
- Val: n=1276274, positives=780
- ID hashes match expected P1 / random control: **yes**

## Val AUPRC @ 0.5

| Stack | Logistic | MLP |
|-------|----------|-----|
| X | 0.003339 (reused) | 0.065598 |
| H | 0.022977 (reused) | 0.007594 |
| H+X | 0.041084 (reused) | 0.057616 |
| random_H | 0.011445 (reused) | 0.159559 |
| random_H+X | n/a (no exact cell) | 0.153151 |

## Answers

1. Pretrained H+X > X? logistic=True (Δ=0.03774537268988898, material=True); mlp=False (Δ=-0.007981798392328286, material=False)
2. Pretrained H+X > random H+X? logistic=False (Δ=None, material=None); mlp=False (Δ=-0.0955349523863172, material=False)
3. MLP materially > logistic on pretrained H+X? True (Δ=0.01653180612254395)
4. H useful above random H (Δ≥0.003)? logistic=True, mlp=False
5. Interpretation: **probe_undercapacity_possible_but_H_still_weak_vs_random**

