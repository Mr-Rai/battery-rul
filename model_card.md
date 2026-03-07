# Model Card — Battery RUL Estimation

## Model Details
- **Task:** Remaining Useful Life (RUL) regression for Li-ion battery cells
- **Output:** Point estimate + 90% prediction interval (cycles to End of Life)
- **EOL Definition:** Discharge capacity < 80% of nominal rated capacity
- **Model versions:** XGBoost baseline, TCN with MC Dropout uncertainty

## Training Data
- **Primary:** CALCE CS2 and CX2 cells (LiCoO₂ chemistry, 1.1Ah and 1.35Ah)
- **Conditions:** Room temperature (~25°C), standard charge/discharge profiles
- **Validation:** NASA PCoE B0005/B0006/B0007 (cross-dataset generalization)

## Performance (to be filled after training)
| Metric | CALCE Test Set | NASA Validation |
|--------|---------------|-----------------|
| RMSE (cycles) | — | — |
| MAE (cycles) | — | — |
| 90% PI Coverage | — | — |
| Early Warning Rate | — | — |

## Known Limitations
- **Chemistry:** Trained on LiCoO₂ only. Do not apply to LFP, NMC, or NCA cells without retraining.
- **Temperature:** Training data collected near 25°C. Predictions at extreme temperatures (< 5°C or > 40°C) are unreliable.
- **Cycling profile:** Optimized for standard constant-current charge/discharge. Dynamic profiles (EV drive cycles) may degrade accuracy.
- **Early life:** Model accuracy is lower in the first 20 cycles before the degradation trend stabilizes.
- **Cell-to-cell variation:** Predictions assume cells within the same chemistry/form-factor. Manufacturing outliers may not be captured.

## Intended Use
- Proactive battery replacement scheduling in controlled lab or fleet environments
- Warranty risk quantification for cell cohorts under known operating conditions
- Research and benchmarking — not for safety-critical deployment without additional validation

## Misuse Warning
Do not use this model as the sole basis for safety decisions (e.g. aviation, medical devices) without independent validation against your specific cell chemistry, operating conditions, and failure mode definitions.

## Authors
Shubham Shekhar Rai — [LinkedIn](https://in.linkedin.com/in/shubhamshekharrai)
