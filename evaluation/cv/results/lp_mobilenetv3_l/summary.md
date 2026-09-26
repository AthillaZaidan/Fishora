# Evaluation summary: `lp_mobilenetv3_l`

Model `mobilenetv3_large_100.ra_in1k`, temperature 0.5346, production abstain threshold 0.0.

Out of scope (not scored): `field_live`: out of operational scope: Fishora classifies fish after landing, not live fish under water

## Labelled slices

| slice | n | accuracy | macro-F1 | mean conf | wrong & conf>=0.9 | ECE |
|---|---|---|---|---|---|---|
| clean | 849 | 91.5% [89.6%, 93.3%] | 0.899 | 0.926 | 1.3% | 0.018 |
| dark | 2547 | 84.5% | 0.822 | 0.870 | 2.7% | 0.030 |
| bright | 2547 | 84.5% | 0.820 | 0.887 | 3.5% | 0.045 |
| blur | 2547 | 42.0% | 0.378 | 0.815 | 22.1% | 0.395 |
| jpeg | 2547 | 50.6% | 0.463 | 0.787 | 13.5% | 0.281 |
| lowres | 2547 | 43.6% | 0.387 | 0.784 | 17.2% | 0.348 |
| rotate | 2547 | 84.7% | 0.817 | 0.904 | 3.5% | 0.057 |
| occlusion | 2547 | 72.0% | 0.692 | 0.844 | 7.1% | 0.126 |
| bg_removed | 848 | 59.7% [56.5%, 63.0%] | 0.577 | 0.807 | 7.4% | 0.210 |
| bg_swap | 848 | 55.1% [51.9%, 58.3%] | 0.584 | 0.830 | 19.2% | 0.285 |
| field | 258 | 29.8% [24.4%, 35.7%] | 0.239 | 0.763 | 22.9% | 0.464 |

## Background tests

- **fish_erased** (n=848): still predicted as the true class 29.0% of the time (chance 9.1%), mean confidence 0.80, confident 46.7%.
  - FISH_GRES: shortcut rate 19.4% (n=489), mean conf 0.85
  - ROBOFLOW: shortcut rate 42.1% (n=359), mean conf 0.72
- **bg_swap** (n=848): accuracy 55.1%, flip to the new background's family 38.1%.
  - fish from FISH_GRES: accuracy 83.6%, flip 5.7%, top predictions {'nila': 96, 'kuniran': 90, 'kembung': 79}
  - fish from ROBOFLOW: accuracy 16.2%, flip 82.2%, top predictions {'kembung': 256, 'tenggiri': 45, 'kuniran': 20}

## Out-of-distribution

| set | n | mean conf | conf>=0.9 | accepted at production threshold | AUROC | FPR@95TPR |
|---|---|---|---|---|---|---|
| ood_unknown_fish | 89 | 0.781 | 41.6% | 100.0% | 0.687 | 86.5% |
| ood_nonfish | 83 | 0.763 | 39.8% | 100.0% | 0.690 | 85.5% |
| ood_synthetic | 22 | 0.633 | 0.0% | 100.0% | 0.870 | 77.3% |

## Field photos per class

| class | n | recall | top confusions |
|---|---|---|---|
| kembung | 22 | 9.1% | {'bandeng': 6, 'kuniran': 5} |
| mujair | 19 | 15.8% | {'tenggiri': 3, 'gelama_bunga': 3} |
| tuna | 29 | 17.2% | {'senangin': 4, 'tenggiri': 4} |
| tenggiri | 37 | 21.6% | {'senangin': 9, 'tuna': 5} |
| bandeng | 34 | 35.3% | {'senangin': 6, 'gulamah': 4} |
| senangin | 48 | 35.4% | {'bandeng': 10, 'kuniran': 5} |
| nila | 29 | 41.4% | {'gelama_bunga': 5, 'senangin': 4} |
| gelama_bunga | 24 | 41.7% | {'kuniran': 5, 'mujair': 3} |
| kuniran | 16 | 50.0% | {'bandeng': 4, 'tenggiri': 3} |

Figures: `figures/confusion.png`, `figures/corruptions.png`, `figures/confidence.png`, `figures/reliability.png`.
