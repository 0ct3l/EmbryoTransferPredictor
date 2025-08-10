![Python](https://img.shields.io/badge/python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

# Embryo Transfer Outcome Prediction (Fresh vs Frozen)

> **Research prototype — Not for clinical use**

This repository implements a PyTorch model to predict IVF outcomes for **Fresh vs Frozen Embryo Transfer (ET)** using **synthetic data** derived from a published RCT (n=838; 419 Fresh, 419 Frozen).  
The model combines a **rule-based risk engine** with a neural network trained on Age, BMI, and Infertility Duration to recommend transfer type.

---

## Trial Summary

| Outcome | Fresh | Frozen | RR (95% CI) | p-value |
|---------|------:|-------:|-------------|---------|
| Live birth | 40% | 32% | 1.25 (1.06–1.48) | 0.009 |
| Clinical pregnancy | 47% | 39% | 1.19 (1.02–1.38) | 0.02 |
| Ongoing pregnancy | 43% | 36% | 1.19 (1.01–1.40) | 0.04 |
| Pregnancy loss† | 10% | 17% | 0.60 (0.35–1.03) | 0.06 |
| Ectopic† | 4% | 1% | 3.52 (0.75–16.63) | 0.11 |

† % of clinical pregnancies.

---

## Method

1. **Rule-based engine**  
   - Encodes trial outcome rates as priors.  
   - Adjusts for Age, BMI, Duration via multiplicative risk factors.  
   - Outputs composite risk score + recommendation.

2. **Synthetic data generation**  
   - Samples features to match trial baselines.  
   - Labels via rule-based engine (weak supervision).

3. **Neural network**  
   - 3-layer MLP, CrossEntropy loss, Adam optimiser.  
   - Learns to reproduce rule-based recommendations.

4. **Visualisation**  
   - Training curves, outcome probabilities, risk factor multipliers.

---

## Limitations

- Labels from heuristics, not patient-level data.  
- Multipliers illustrative, not validated.  
- Probabilities uncalibrated.  
- Educational use only.

---

## Usage

```bash
pip install torch numpy matplotlib
python embryo_transfer.py

