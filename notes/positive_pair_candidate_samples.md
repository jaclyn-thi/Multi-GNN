# Positive-pair candidate samples (label-free diagnostic)

Dataset: **Small-HI** | anchors: **2000** | window: **86400.0s**

## endpoint_role_temporal_v1

Shared endpoint u; u plays the same role (sender/receiver) on both edges; |Δt| <= W. Label-free; exclusion of identity only.

Anchors: 2000 | zero: 39.1% | one: 23.3% | mean: 73.39 | max: 4353

### Sample 1
- anchor EdgeID `34830` → positive `333641`
- roles: (28841→24297) vs (28841→98199)
- Δt=3300s | amounts 44 vs 1.1e+03 | log Δ=3.240
- pair_past_count(anchor)=0.0

### Sample 2
- anchor EdgeID `34978` → positive `350073`
- roles: (28959→28959) vs (28959→274742)
- Δt=4020s | amounts 16 vs 8.9e+03 | log Δ=6.248
- pair_past_count(anchor)=0.0

### Sample 3
- anchor EdgeID `35071` → positive `35083`
- roles: (29035→29035) vs (29035→29035)
- Δt=960s | amounts 2e+03 vs 23 | log Δ=4.417
- pair_past_count(anchor)=0.0

### Sample 4
- anchor EdgeID `40450` → positive `346739`
- roles: (33437→33436) vs (33437→39104)
- Δt=4500s | amounts 86 vs 2.2e+03 | log Δ=3.236
- pair_past_count(anchor)=0.0

### Sample 5
- anchor EdgeID `55802` → positive `55801`
- roles: (45891→45891) vs (45891→45891)
- Δt=780s | amounts 16 vs 5.8e+02 | log Δ=3.536
- pair_past_count(anchor)=0.0

### Sample 6
- anchor EdgeID `68348` → positive `68347`
- roles: (56061→56060) vs (56060→56060)
- Δt=420s | amounts 8.8e+02 vs 5.4e+03 | log Δ=1.803
- pair_past_count(anchor)=0.0

### Sample 7
- anchor EdgeID `80125` → positive `9017`
- roles: (65568→7534) vs (7534→7534)
- Δt=1620s | amounts 6 vs 13 | log Δ=0.700
- pair_past_count(anchor)=0.0

### Sample 8
- anchor EdgeID `81399` → positive `81393`
- roles: (66587→66587) vs (66587→66587)
- Δt=420s | amounts 25 vs 1.5e+03 | log Δ=4.104
- pair_past_count(anchor)=0.0

### Sample 9
- anchor EdgeID `118231` → positive `118201`
- roles: (95623→95623) vs (95623→95623)
- Δt=420s | amounts 4.9 vs 1.6e+03 | log Δ=5.575
- pair_past_count(anchor)=0.0

### Sample 10
- anchor EdgeID `121913` → positive `121912`
- roles: (98496→98497) vs (98496→98496)
- Δt=60s | amounts 3.8e+03 vs 1.7e+03 | log Δ=0.813
- pair_past_count(anchor)=0.0

### Sample 11
- anchor EdgeID `123792` → positive `146747`
- roles: (100080→100080) vs (100080→118879)
- Δt=1380s | amounts 5.1e+02 vs 7.5e+02 | log Δ=0.382
- pair_past_count(anchor)=0.0

### Sample 12
- anchor EdgeID `138301` → positive `335950`
- roles: (112096→112096) vs (112096→134897)
- Δt=3480s | amounts 20 vs 2.4e+02 | log Δ=2.421
- pair_past_count(anchor)=0.0

### Sample 13
- anchor EdgeID `147973` → positive `153468`
- roles: (119864→119883) vs (121628→119883)
- Δt=120s | amounts 0.09 vs 1.4e+04 | log Δ=9.485
- pair_past_count(anchor)=0.0

### Sample 14
- anchor EdgeID `148457` → positive `188385`
- roles: (120292→120293) vs (120293→120293)
- Δt=0s | amounts 1.1e+02 vs 1e+05 | log Δ=6.827
- pair_past_count(anchor)=0.0

### Sample 15
- anchor EdgeID `156230` → positive `156228`
- roles: (126571→126571) vs (108055→126571)
- Δt=600s | amounts 16 vs 9.9e+02 | log Δ=4.079
- pair_past_count(anchor)=0.0

### Sample 16
- anchor EdgeID `157374` → positive `157376`
- roles: (127483→127483) vs (127483→127483)
- Δt=420s | amounts 2.1e+03 vs 4.8 | log Δ=5.902
- pair_past_count(anchor)=0.0

### Sample 17
- anchor EdgeID `158264` → positive `404321`
- roles: (128216→128216) vs (128216→295628)
- Δt=9060s | amounts 3.7e+04 vs 2.5e+03 | log Δ=2.709
- pair_past_count(anchor)=0.0

### Sample 18
- anchor EdgeID `169342` → positive `169341`
- roles: (117364→137213) vs (117364→137213)
- Δt=600s | amounts 2.4e+03 vs 4.1e+03 | log Δ=0.561
- pair_past_count(anchor)=0.0

### Sample 19
- anchor EdgeID `178259` → positive `178260`
- roles: (144327→144327) vs (98992→144327)
- Δt=1200s | amounts 5.9e+04 vs 87 | log Δ=6.509
- pair_past_count(anchor)=0.0

### Sample 20
- anchor EdgeID `199237` → positive `49500`
- roles: (40746→160898) vs (40746→40745)
- Δt=1380s | amounts 7.3e+02 vs 5.5e+02 | log Δ=0.293
- pair_past_count(anchor)=0.0

## shared_sender_temporal_amount_v1

Same sender; |Δt| <= W; |log1p(amount)-log1p(anchor_amount)| <= δ.

Anchors: 2000 | zero: 84.9% | one: 8.9% | mean: 7.35 | max: 690

### Sample 1
- anchor EdgeID `68348` → positive `68349`
- roles: (56061→56060) vs (56061→56060)
- Δt=1020s | amounts 8.8e+02 vs 1.1e+03 | log Δ=0.224
- pair_past_count(anchor)=0.0

### Sample 2
- anchor EdgeID `123792` → positive `146747`
- roles: (100080→100080) vs (100080→118879)
- Δt=1380s | amounts 5.1e+02 vs 7.5e+02 | log Δ=0.382
- pair_past_count(anchor)=0.0

### Sample 3
- anchor EdgeID `199237` → positive `49500`
- roles: (40746→160898) vs (40746→40745)
- Δt=1380s | amounts 7.3e+02 vs 5.5e+02 | log Δ=0.293
- pair_past_count(anchor)=0.0

### Sample 4
- anchor EdgeID `34180` → positive `34182`
- roles: (26744→28314) vs (26744→28314)
- Δt=240s | amounts 1.4e+03 vs 1.6e+03 | log Δ=0.133
- pair_past_count(anchor)=0.0

### Sample 5
- anchor EdgeID `128476` → positive `128477`
- roles: (103949→103950) vs (103949→103950)
- Δt=60s | amounts 3.2e+03 vs 2.5e+03 | log Δ=0.273
- pair_past_count(anchor)=0.0

### Sample 6
- anchor EdgeID `131493` → positive `367578`
- roles: (106490→106490) vs (106490→108649)
- Δt=6240s | amounts 14 vs 10 | log Δ=0.271
- pair_past_count(anchor)=0.0

### Sample 7
- anchor EdgeID `135699` → positive `135698`
- roles: (109972→109971) vs (109972→109971)
- Δt=1320s | amounts 5.4e+04 vs 4.3e+04 | log Δ=0.220
- pair_past_count(anchor)=0.0

### Sample 8
- anchor EdgeID `156615` → positive `156616`
- roles: (110934→126865) vs (110934→126865)
- Δt=1020s | amounts 1.4e+03 vs 1.7e+03 | log Δ=0.255
- pair_past_count(anchor)=0.0

### Sample 9
- anchor EdgeID `188761` → positive `140259`
- roles: (113687→152615) vs (113687→113687)
- Δt=840s | amounts 7.7e+03 vs 1e+04 | log Δ=0.257
- pair_past_count(anchor)=0.0

### Sample 10
- anchor EdgeID `275100` → positive `276042`
- roles: (221493→222928) vs (221493→223706)
- Δt=1080s | amounts 9.5e+05 vs 1.4e+06 | log Δ=0.371
- pair_past_count(anchor)=0.0

### Sample 11
- anchor EdgeID `318792` → positive `318975`
- roles: (258419→258420) vs (258419→258573)
- Δt=1620s | amounts 0.59 vs 0.09 | log Δ=0.376
- pair_past_count(anchor)=0.0

### Sample 12
- anchor EdgeID `668` → positive `60223`
- roles: (585→585) vs (585→49444)
- Δt=300s | amounts 1.1e+05 vs 1e+05 | log Δ=0.051
- pair_past_count(anchor)=0.0

### Sample 13
- anchor EdgeID `53788` → positive `53786`
- roles: (44255→44256) vs (44255→44256)
- Δt=1020s | amounts 8e+02 vs 1.2e+03 | log Δ=0.365
- pair_past_count(anchor)=1.0

### Sample 14
- anchor EdgeID `153938` → positive `352402`
- roles: (124752→124752) vs (124752→120963)
- Δt=5220s | amounts 8.7e+06 vs 1.4e+07 | log Δ=0.441
- pair_past_count(anchor)=0.0

### Sample 15
- anchor EdgeID `181080` → positive `181079`
- roles: (146596→146594) vs (146596→146596)
- Δt=0s | amounts 1.5e+04 vs 1.5e+04 | log Δ=0.000
- pair_past_count(anchor)=0.0

### Sample 16
- anchor EdgeID `212943` → positive `212945`
- roles: (172014→172123) vs (172014→172123)
- Δt=960s | amounts 9.2e+06 vs 1e+07 | log Δ=0.085
- pair_past_count(anchor)=0.0

### Sample 17
- anchor EdgeID `219691` → positive `219692`
- roles: (171894→177772) vs (171894→177772)
- Δt=-120s | amounts 1.8e+04 vs 2.6e+04 | log Δ=0.371
- pair_past_count(anchor)=1.0

### Sample 18
- anchor EdgeID `259174` → positive `259176`
- roles: (210182→210181) vs (210182→210181)
- Δt=0s | amounts 4.3e+04 vs 4.3e+04 | log Δ=0.002
- pair_past_count(anchor)=0.0

### Sample 19
- anchor EdgeID `260088` → positive `260087`
- roles: (210898→210899) vs (210898→210899)
- Δt=1380s | amounts 9.9e+02 vs 8.3e+02 | log Δ=0.170
- pair_past_count(anchor)=0.0

### Sample 20
- anchor EdgeID `323151` → positive `323150`
- roles: (258437→261890) vs (258437→261890)
- Δt=0s | amounts 4.5 vs 6.1 | log Δ=0.255
- pair_past_count(anchor)=1.0

## repeat_pair_forward_temporal_v1

Same ordered (sender, receiver); forward repeat 0 < Δt <= W (prior edge earlier).

Anchors: 2000 | zero: 75.8% | one: 18.8% | mean: 0.31 | max: 4

### Sample 1
- anchor EdgeID `34978` → positive `34977`
- roles: (28959→28959) vs (28959→28959)
- Δt=240s | amounts 16 vs 1e+07 | log Δ=13.312
- pair_past_count(anchor)=0.0

### Sample 2
- anchor EdgeID `35071` → positive `35083`
- roles: (29035→29035) vs (29035→29035)
- Δt=960s | amounts 2e+03 vs 23 | log Δ=4.417
- pair_past_count(anchor)=0.0

### Sample 3
- anchor EdgeID `40450` → positive `40448`
- roles: (33437→33436) vs (33437→33436)
- Δt=720s | amounts 86 vs 3.3e+02 | log Δ=1.325
- pair_past_count(anchor)=0.0

### Sample 4
- anchor EdgeID `55802` → positive `55801`
- roles: (45891→45891) vs (45891→45891)
- Δt=780s | amounts 16 vs 5.8e+02 | log Δ=3.536
- pair_past_count(anchor)=0.0

### Sample 5
- anchor EdgeID `68348` → positive `68349`
- roles: (56061→56060) vs (56061→56060)
- Δt=1020s | amounts 8.8e+02 vs 1.1e+03 | log Δ=0.224
- pair_past_count(anchor)=0.0

### Sample 6
- anchor EdgeID `81399` → positive `81393`
- roles: (66587→66587) vs (66587→66587)
- Δt=420s | amounts 25 vs 1.5e+03 | log Δ=4.104
- pair_past_count(anchor)=0.0

### Sample 7
- anchor EdgeID `118231` → positive `118201`
- roles: (95623→95623) vs (95623→95623)
- Δt=420s | amounts 4.9 vs 1.6e+03 | log Δ=5.575
- pair_past_count(anchor)=0.0

### Sample 8
- anchor EdgeID `123792` → positive `350580`
- roles: (100080→100080) vs (100080→100080)
- Δt=4260s | amounts 5.1e+02 vs 5.8 | log Δ=4.320
- pair_past_count(anchor)=0.0

### Sample 9
- anchor EdgeID `157374` → positive `157376`
- roles: (127483→127483) vs (127483→127483)
- Δt=420s | amounts 2.1e+03 vs 4.8 | log Δ=5.902
- pair_past_count(anchor)=0.0

### Sample 10
- anchor EdgeID `169342` → positive `169341`
- roles: (117364→137213) vs (117364→137213)
- Δt=600s | amounts 2.4e+03 vs 4.1e+03 | log Δ=0.561
- pair_past_count(anchor)=0.0

### Sample 11
- anchor EdgeID `202247` → positive `337811`
- roles: (163364→163364) vs (163364→163364)
- Δt=2760s | amounts 5.8e+03 vs 83 | log Δ=4.235
- pair_past_count(anchor)=0.0

### Sample 12
- anchor EdgeID `203744` → positive `203741`
- roles: (164611→164611) vs (164611→164611)
- Δt=900s | amounts 70 vs 1.9e+04 | log Δ=5.591
- pair_past_count(anchor)=0.0

### Sample 13
- anchor EdgeID `232126` → positive `232129`
- roles: (188045→188045) vs (188045→188045)
- Δt=1560s | amounts 1.5e+07 vs 1.2e+03 | log Δ=9.485
- pair_past_count(anchor)=0.0

### Sample 14
- anchor EdgeID `260511` → positive `260496`
- roles: (211216→211216) vs (211216→211216)
- Δt=540s | amounts 32 vs 1.7e+04 | log Δ=6.243
- pair_past_count(anchor)=0.0

### Sample 15
- anchor EdgeID `262952` → positive `262953`
- roles: (213173→213174) vs (213173→213174)
- Δt=720s | amounts 7.3e+03 vs 2.3e+07 | log Δ=8.048
- pair_past_count(anchor)=0.0

### Sample 16
- anchor EdgeID `314773` → positive `314775`
- roles: (255249→255250) vs (255249→255250)
- Δt=60s | amounts 2.3e+04 vs 3.7e+03 | log Δ=1.809
- pair_past_count(anchor)=0.0

### Sample 17
- anchor EdgeID `325137` → positive `325136`
- roles: (263445→263445) vs (263445→263445)
- Δt=960s | amounts 0.0019 vs 6.7 | log Δ=2.040
- pair_past_count(anchor)=0.0

### Sample 18
- anchor EdgeID `34180` → positive `34182`
- roles: (26744→28314) vs (26744→28314)
- Δt=240s | amounts 1.4e+03 vs 1.6e+03 | log Δ=0.133
- pair_past_count(anchor)=0.0

### Sample 19
- anchor EdgeID `38793` → positive `38794`
- roles: (25156→32104) vs (25156→32104)
- Δt=420s | amounts 3.6e+03 vs 2.2e+03 | log Δ=0.509
- pair_past_count(anchor)=0.0

### Sample 20
- anchor EdgeID `60056` → positive `60058`
- roles: (49317→49317) vs (49317→49317)
- Δt=420s | amounts 1.3e+06 vs 13 | log Δ=11.429
- pair_past_count(anchor)=0.0
