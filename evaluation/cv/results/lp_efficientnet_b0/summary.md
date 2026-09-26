# Evaluation summary: `lp_efficientnet_b0`

Model `efficientnet_b0.ra_in1k`, temperature 0.4958, production abstain threshold 0.0.

Out of scope (not scored): `field_live`: out of operational scope: Fishora classifies fish after landing, not live fish under water

## Labelled slices

| slice | n | accuracy | macro-F1 | mean conf | wrong & conf>=0.9 | ECE |
|---|---|---|---|---|---|---|
| clean | 849 | 94.9% [93.5%, 96.2%] | 0.937 | 0.939 | 0.8% | 0.026 |
| dark | 2547 | 83.3% | 0.818 | 0.874 | 2.1% | 0.041 |
| bright | 2547 | 89.8% | 0.872 | 0.908 | 1.6% | 0.017 |
| blur | 2547 | 47.5% | 0.457 | 0.834 | 21.6% | 0.359 |
| jpeg | 2547 | 59.6% | 0.538 | 0.794 | 8.1% | 0.198 |
| lowres | 2547 | 54.5% | 0.508 | 0.806 | 13.0% | 0.261 |
| rotate | 2547 | 91.2% | 0.888 | 0.927 | 1.7% | 0.016 |
| occlusion | 2547 | 83.4% | 0.807 | 0.883 | 3.1% | 0.050 |
| bg_removed | 848 | 59.4% [56.2%, 63.0%] | 0.544 | 0.842 | 12.9% | 0.248 |
| bg_swap | 848 | 55.0% [51.9%, 58.3%] | 0.558 | 0.781 | 8.8% | 0.234 |
| field | 258 | 25.6% [20.5%, 31.4%] | 0.183 | 0.751 | 20.9% | 0.497 |

## Background tests

- **fish_erased** (n=848): still predicted as the true class 20.8% of the time (chance 9.1%), mean confidence 0.75, confident 30.0%.
  - FISH_GRES: shortcut rate 21.5% (n=489), mean conf 0.73
  - ROBOFLOW: shortcut rate 19.8% (n=359), mean conf 0.76
- **bg_swap** (n=848): accuracy 55.0%, flip to the new background's family 40.2%.
  - fish from FISH_GRES: accuracy 61.1%, flip 32.3%, top predictions {'tenggiri': 153, 'nila': 82, 'kuniran': 82}
  - fish from ROBOFLOW: accuracy 46.5%, flip 51.0%, top predictions {'tuna': 104, 'bandeng': 71, 'kembung': 70}

## Out-of-distribution

| set | n | mean conf | conf>=0.9 | accepted at production threshold | AUROC | FPR@95TPR |
|---|---|---|---|---|---|---|
| ood_unknown_fish | 89 | 0.808 | 41.6% | 100.0% | 0.671 | 95.5% |
| ood_nonfish | 83 | 0.761 | 38.6% | 100.0% | 0.721 | 86.7% |
| ood_synthetic | 22 | 0.592 | 13.6% | 100.0% | 0.863 | 63.6% |

## Field photos per class

| class | n | recall | top confusions |
|---|---|---|---|
| kuniran | 16 | 0.0% | {'bandeng': 8, 'gembolo': 3} |
| tuna | 29 | 0.0% | {'kembung': 13, 'bandeng': 6} |
| tenggiri | 37 | 10.8% | {'bandeng': 10, 'kembung': 6} |
| senangin | 48 | 14.6% | {'bandeng': 11, 'gelama_bunga': 6} |
| kembung | 22 | 22.7% | {'bandeng': 9, 'mujair': 3} |
| gelama_bunga | 24 | 33.3% | {'kuniran': 6, 'bandeng': 3} |
| mujair | 19 | 36.8% | {'nila': 7, 'bandeng': 3} |
| bandeng | 34 | 47.1% | {'gelama_bunga': 4, 'nila': 4} |
| nila | 29 | 65.5% | {'bandeng': 3, 'mujair': 2} |

Figures: `figures/confusion.png`, `figures/corruptions.png`, `figures/confidence.png`, `figures/reliability.png`.
