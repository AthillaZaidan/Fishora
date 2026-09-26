# Evaluation summary: `lp_vit_s`

Model `vit_small_patch16_dinov3.lvd1689m`, temperature 0.0500, production abstain threshold 0.0.

Out of scope (not scored): `field_live`: out of operational scope: Fishora classifies fish after landing, not live fish under water

## Labelled slices

| slice | n | accuracy | macro-F1 | mean conf | wrong & conf>=0.9 | ECE |
|---|---|---|---|---|---|---|
| clean | 849 | 99.6% [99.2%, 100.0%] | 0.995 | 0.999 | 0.0% | 0.003 |
| dark | 2547 | 99.8% | 0.997 | 0.999 | 0.0% | 0.001 |
| bright | 2547 | 98.8% | 0.985 | 0.996 | 0.7% | 0.009 |
| blur | 2547 | 84.2% | 0.826 | 0.985 | 12.7% | 0.143 |
| jpeg | 2547 | 84.7% | 0.839 | 0.981 | 11.8% | 0.133 |
| lowres | 2547 | 87.8% | 0.874 | 0.986 | 9.4% | 0.109 |
| rotate | 2547 | 99.4% | 0.994 | 0.999 | 0.5% | 0.006 |
| occlusion | 2547 | 91.1% | 0.913 | 0.990 | 7.3% | 0.079 |
| bg_removed | 848 | 98.3% [97.4%, 99.2%] | 0.980 | 0.998 | 1.3% | 0.016 |
| bg_swap | 848 | 98.1% [97.2%, 99.1%] | 0.978 | 0.997 | 1.4% | 0.017 |
| field | 258 | 51.9% [46.1%, 58.5%] | 0.412 | 0.958 | 38.8% | 0.439 |

## Background tests

- **fish_erased** (n=848): still predicted as the true class 34.3% of the time (chance 9.1%), mean confidence 0.94, confident 82.2%.
  - FISH_GRES: shortcut rate 20.4% (n=489), mean conf 0.95
  - ROBOFLOW: shortcut rate 53.2% (n=359), mean conf 0.93
- **bg_swap** (n=848): accuracy 98.1%, flip to the new background's family 0.6%.
  - fish from FISH_GRES: accuracy 97.1%, flip 1.0%, top predictions {'nila': 92, 'kuniran': 87, 'kembung': 81}
  - fish from ROBOFLOW: accuracy 99.4%, flip 0.0%, top predictions {'gembolo': 150, 'tuna': 106, 'tenggiri': 103}

## Out-of-distribution

| set | n | mean conf | conf>=0.9 | accepted at production threshold | AUROC | FPR@95TPR |
|---|---|---|---|---|---|---|
| ood_unknown_fish | 89 | 0.932 | 80.9% | 100.0% | 0.805 | 73.0% |
| ood_nonfish | 83 | 0.931 | 80.7% | 100.0% | 0.854 | 71.1% |
| ood_synthetic | 22 | 0.867 | 59.1% | 100.0% | 0.879 | 40.9% |

## Field photos per class

| class | n | recall | top confusions |
|---|---|---|---|
| tuna | 29 | 6.9% | {'kembung': 17, 'senangin': 4} |
| tenggiri | 37 | 27.0% | {'kembung': 10, 'tuna': 6} |
| kuniran | 16 | 43.8% | {'gembolo': 4, 'bandeng': 3} |
| kembung | 22 | 45.5% | {'gembolo': 8, 'bandeng': 3} |
| gelama_bunga | 24 | 45.8% | {'kuniran': 6, 'nila': 3} |
| mujair | 19 | 52.6% | {'nila': 7, 'senangin': 2} |
| bandeng | 34 | 58.8% | {'senangin': 7, 'kembung': 2} |
| senangin | 48 | 77.1% | {'tenggiri': 3, 'gelama_bunga': 2} |
| nila | 29 | 93.1% | {'kuniran': 1, 'tuna': 1} |

Figures: `figures/confusion.png`, `figures/corruptions.png`, `figures/confidence.png`, `figures/reliability.png`.
