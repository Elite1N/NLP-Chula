DeBERTa V1 (base) Experiment Log

Issues Encountered:

1. DeBERTa V3 (microsoft/deberta-v3-base) caused instability (Loss 0, Grad NaN, Predictions 'conflict').
2. BF16 precision caused RuntimeError on Windows: "value cannot be converted to type at::BFloat16 without overflow".

Fixes Applied:

1. Switched model to `microsoft/deberta-base` (V1).
2. Disabled BF16 and FP16 (using FP32 default precision) to avoid overflow errors.
3. Reset Learning Rate to 2e-5.

Current Status:

- Training in progress (FP32).
- Initial loss decreasing (~0.6 -> ~0.4).
- Speed: ~6.3 it/s on RTX 4060.
