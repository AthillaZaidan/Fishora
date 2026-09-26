# Evaluation summary: `lp_vit_b`

Model `vit_base_patch16_dinov3.lvd1689m`, temperature 0.0900, production abstain threshold 0.0.

Out of scope (not scored): `field_live`: out of operational scope: Fishora classifies fish after landing, not live fish under water

## Labelled slices

| slice | n | accuracy | macro-F1 | mean conf | wrong & conf>=0.9 | ECE |
|---|---|---|---|---|---|---|
| clean | 849 | 100.0% [100.0%, 100.0%] | 1.000 | 1.000 | 0.0% | 0.000 |
| dark | 2547 | 100.0% | 1.000 | 1.000 | 0.0% | 0.000 |
| bright | 2547 | 99.8% | 0.997 | 0.999 | 0.2% | 0.002 |
| blur | 2547 | 87.0% | 0.857 | 0.981 | 9.8% | 0.111 |
| jpeg | 2547 | 86.1% | 0.870 | 0.979 | 9.7% | 0.119 |
| lowres | 2547 | 91.4% | 0.906 | 0.988 | 6.4% | 0.074 |
| rotate | 2547 | 100.0% | 0.999 | 0.999 | 0.0% | 0.001 |
| occlusion | 2547 | 95.3% | 0.949 | 0.988 | 2.9% | 0.036 |
| bg_removed | 848 | 99.4% [98.8%, 99.9%] | 0.993 | 0.999 | 0.4% | 0.005 |
| bg_swap | 848 | 99.4% [98.8%, 99.9%] | 0.993 | 0.999 | 0.5% | 0.006 |
| field | 258 | 67.1% [61.6%, 72.5%] | 0.543 | 0.935 | 21.3% | 0.271 |

## Background tests

- **fish_erased** (n=848): still predicted as the true class 38.7% of the time (chance 9.1%), mean confidence 0.83, confident 52.9%.
  - FISH_GRES: shortcut rate 39.5% (n=489), mean conf 0.88
  - ROBOFLOW: shortcut rate 37.6% (n=359), mean conf 0.76
- **bg_swap** (n=848): accuracy 99.4%, flip to the new background's family 0.1%.
  - fish from FISH_GRES: accuracy 99.0%, flip 0.2%, top predictions {'nila': 89, 'kuniran': 87, 'kembung': 82}
  - fish from ROBOFLOW: accuracy 100.0%, flip 0.0%, top predictions {'gembolo': 148, 'tuna': 106, 'tenggiri': 105}

## Out-of-distribution

| set | n | mean conf | conf>=0.9 | accepted at production threshold | AUROC | FPR@95TPR |
|---|---|---|---|---|---|---|
| ood_unknown_fish | 89 | 0.938 | 82.0% | 100.0% | 0.829 | 75.3% |
| ood_nonfish | 83 | 0.863 | 61.4% | 100.0% | 0.945 | 54.2% |
| ood_synthetic | 22 | 0.814 | 27.3% | 100.0% | 0.958 | 18.2% |

## Field photos per class

| class | n | recall | top confusions |
|---|---|---|---|
| tuna | 29 | 13.8% | {'kembung': 20, 'tenggiri': 4} |
| tenggiri | 37 | 48.6% | {'kembung': 10, 'gulamah': 6} |
| kuniran | 16 | 56.2% | {'gembolo': 3, 'mujair': 2} |
| gelama_bunga | 24 | 66.7% | {'kuniran': 5, 'bandeng': 1} |
| kembung | 22 | 72.7% | {'gembolo': 6} |
| mujair | 19 | 73.7% | {'nila': 4, 'senangin': 1} |
| senangin | 48 | 77.1% | {'gelama_bunga': 5, 'bandeng': 4} |
| bandeng | 34 | 88.2% | {'senangin': 3, 'tenggiri': 1} |
| nila | 29 | 100.0% | {} |

Figures: `figures/confusion.png`, `figures/corruptions.png`, `figures/confidence.png`, `figures/reliability.png`.
