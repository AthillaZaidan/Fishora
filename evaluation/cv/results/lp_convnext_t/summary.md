# Evaluation summary: `lp_convnext_t`

Model `convnext_tiny.dinov3_lvd1689m`, temperature 0.1073, production abstain threshold 0.0.

Out of scope (not scored): `field_live`: out of operational scope: Fishora classifies fish after landing, not live fish under water

## Labelled slices

| slice | n | accuracy | macro-F1 | mean conf | wrong & conf>=0.9 | ECE |
|---|---|---|---|---|---|---|
| clean | 849 | 99.5% [99.1%, 100.0%] | 0.993 | 0.998 | 0.4% | 0.005 |
| dark | 2547 | 99.5% | 0.992 | 0.997 | 0.4% | 0.005 |
| bright | 2547 | 98.0% | 0.970 | 0.995 | 1.4% | 0.016 |
| blur | 2547 | 83.1% | 0.805 | 0.977 | 12.1% | 0.146 |
| jpeg | 2547 | 88.1% | 0.859 | 0.979 | 8.7% | 0.098 |
| lowres | 2547 | 86.1% | 0.837 | 0.981 | 10.2% | 0.121 |
| rotate | 2547 | 98.9% | 0.984 | 0.997 | 0.7% | 0.009 |
| occlusion | 2547 | 83.4% | 0.813 | 0.973 | 11.4% | 0.139 |
| bg_removed | 848 | 95.6% [94.1%, 96.9%] | 0.955 | 0.989 | 2.7% | 0.037 |
| bg_swap | 848 | 87.0% [84.8%, 89.4%] | 0.881 | 0.976 | 9.2% | 0.110 |
| field | 258 | 53.1% [46.9%, 59.3%] | 0.401 | 0.946 | 34.1% | 0.419 |

## Background tests

- **fish_erased** (n=848): still predicted as the true class 23.1% of the time (chance 9.1%), mean confidence 0.91, confident 73.2%.
  - FISH_GRES: shortcut rate 23.1% (n=489), mean conf 0.93
  - ROBOFLOW: shortcut rate 23.1% (n=359), mean conf 0.88
- **bg_swap** (n=848): accuracy 87.0%, flip to the new background's family 10.7%.
  - fish from FISH_GRES: accuracy 94.7%, flip 1.4%, top predictions {'kuniran': 93, 'nila': 92, 'kembung': 81}
  - fish from ROBOFLOW: accuracy 76.6%, flip 23.4%, top predictions {'tuna': 103, 'gembolo': 97, 'tenggiri': 75}

## Out-of-distribution

| set | n | mean conf | conf>=0.9 | accepted at production threshold | AUROC | FPR@95TPR |
|---|---|---|---|---|---|---|
| ood_unknown_fish | 89 | 0.942 | 82.0% | 100.0% | 0.784 | 80.9% |
| ood_nonfish | 83 | 0.921 | 78.3% | 100.0% | 0.852 | 78.3% |
| ood_synthetic | 22 | 0.850 | 54.5% | 100.0% | 0.953 | 40.9% |

## Field photos per class

| class | n | recall | top confusions |
|---|---|---|---|
| kuniran | 16 | 6.2% | {'kembung': 9, 'gembolo': 4} |
| tuna | 29 | 20.7% | {'tenggiri': 8, 'kembung': 8} |
| tenggiri | 37 | 35.1% | {'mujair': 7, 'bandeng': 6} |
| gelama_bunga | 24 | 45.8% | {'gulamah': 5, 'bandeng': 2} |
| mujair | 19 | 47.4% | {'nila': 7, 'bandeng': 1} |
| bandeng | 34 | 64.7% | {'senangin': 5, 'kembung': 3} |
| senangin | 48 | 70.8% | {'bandeng': 6, 'mujair': 2} |
| nila | 29 | 72.4% | {'mujair': 3, 'kuniran': 2} |
| kembung | 22 | 90.9% | {'bandeng': 2} |

Figures: `figures/confusion.png`, `figures/corruptions.png`, `figures/confidence.png`, `figures/reliability.png`.
