# Evaluation summary: `lp_vit_l`

Model `vit_large_patch16_dinov3.lvd1689m`, temperature 0.0771, production abstain threshold 0.0.

Out of scope (not scored): `field_live`: out of operational scope: Fishora classifies fish after landing, not live fish under water

## Labelled slices

| slice | n | accuracy | macro-F1 | mean conf | wrong & conf>=0.9 | ECE |
|---|---|---|---|---|---|---|
| clean | 849 | 100.0% [100.0%, 100.0%] | 1.000 | 1.000 | 0.0% | 0.000 |
| dark | 2547 | 100.0% | 1.000 | 1.000 | 0.0% | 0.000 |
| bright | 2547 | 99.9% | 0.998 | 0.999 | 0.1% | 0.001 |
| blur | 2547 | 95.3% | 0.946 | 0.992 | 3.4% | 0.039 |
| jpeg | 2547 | 97.7% | 0.977 | 0.994 | 1.5% | 0.017 |
| lowres | 2547 | 98.0% | 0.978 | 0.996 | 1.5% | 0.016 |
| rotate | 2547 | 100.0% | 1.000 | 1.000 | 0.0% | 0.000 |
| occlusion | 2547 | 98.1% | 0.983 | 0.996 | 1.5% | 0.017 |
| bg_removed | 848 | 98.9% [98.2%, 99.5%] | 0.989 | 0.998 | 0.7% | 0.009 |
| bg_swap | 848 | 98.7% [97.9%, 99.4%] | 0.987 | 0.998 | 0.8% | 0.012 |
| field | 258 | 81.8% [76.7%, 86.0%] | 0.667 | 0.984 | 15.1% | 0.166 |

## Background tests

- **fish_erased** (n=848): still predicted as the true class 42.5% of the time (chance 9.1%), mean confidence 0.83, confident 51.4%.
  - FISH_GRES: shortcut rate 41.1% (n=489), mean conf 0.85
  - ROBOFLOW: shortcut rate 44.3% (n=359), mean conf 0.80
- **bg_swap** (n=848): accuracy 98.7%, flip to the new background's family 0.7%.
  - fish from FISH_GRES: accuracy 99.0%, flip 0.0%, top predictions {'nila': 89, 'kuniran': 87, 'kembung': 82}
  - fish from ROBOFLOW: accuracy 98.3%, flip 1.7%, top predictions {'gembolo': 142, 'tuna': 106, 'tenggiri': 105}

## Out-of-distribution

| set | n | mean conf | conf>=0.9 | accepted at production threshold | AUROC | FPR@95TPR |
|---|---|---|---|---|---|---|
| ood_unknown_fish | 89 | 0.922 | 76.4% | 100.0% | 0.778 | 56.2% |
| ood_nonfish | 83 | 0.827 | 48.2% | 100.0% | 0.983 | 8.4% |
| ood_synthetic | 22 | 0.754 | 36.4% | 100.0% | 0.988 | 0.0% |

## Field photos per class

| class | n | recall | top confusions |
|---|---|---|---|
| mujair | 19 | 57.9% | {'nila': 8} |
| kembung | 22 | 59.1% | {'gembolo': 8, 'senangin': 1} |
| tenggiri | 37 | 62.2% | {'tuna': 11, 'kembung': 2} |
| kuniran | 16 | 75.0% | {'gembolo': 2, 'tenggiri': 1} |
| bandeng | 34 | 85.3% | {'tenggiri': 4, 'senangin': 1} |
| gelama_bunga | 24 | 87.5% | {'kuniran': 2, 'senangin': 1} |
| tuna | 29 | 89.7% | {'tenggiri': 2, 'gulamah': 1} |
| nila | 29 | 96.6% | {'mujair': 1} |
| senangin | 48 | 100.0% | {} |

Figures: `figures/confusion.png`, `figures/corruptions.png`, `figures/confidence.png`, `figures/reliability.png`.
