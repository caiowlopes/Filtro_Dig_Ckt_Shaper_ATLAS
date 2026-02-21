# Digital Filter Simulator for Circuit Shapers

Research-oriented implementation of linear and adaptive digital filters applied to signal reconstruction in circuit shaper systems.

This project evaluates different filtering strategies for recovering an original signal from a distorted shaper readout, including automatic model order and delay selection.

---

## Overview

Given:

- Original signal (ground truth)
- Shaper readout (distorted measurement)

The goal is to reconstruct the original signal using:

- Least Squares (Moore–Penrose pseudo-inverse)
- Improved LS with delay, bias and clipping
- LMS (Least Mean Squares)
- NLMS (Normalized LMS)

Performance is evaluated using:

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

Both filter order and delay are automatically optimized.

---
## Research Context

This work supports statistical modeling and signal reconstruction studies in circuit shaping systems, forming part of a broader research line involving:

- Monte Carlo simulation
- Pole variability analysis
- Adaptive filtering for physical systems

---

## Implemented Methods

### 1. Least Squares (LS)
Closed-form solution using Moore–Penrose pseudo-inverse.

### 2. LS with Delay and Bias
Includes:
- Explicit delay alignment
- Bias learning
- Output clipping
- Improved temporal consistency

### 3. LMS
Adaptive gradient-based update:
w(k+1) = w(k) + μ e(k) x(k)

### 4. NLMS
Normalized LMS with adaptive step size:
μ / (ε + ||x||²)

More stable when input energy varies.

---

## Model Selection Strategy

Two search procedures are implemented:

- Optimal filter order selection (based on RMSE / MAE)
- Optimal delay selection

Evaluation can be performed using:
- Independent window per order
- Common evaluation window across all orders

This avoids artificial bias from non-estimated samples.

---

## Structure
├── Filter implementations (LS, LMS, NLMS)
├── Automatic order search
├── Automatic delay search
├── Signal comparison plots
├── Error metrics (RMSE / MAE)


---

## Author
Caio W. Lopes  
Electrical Engineer – UERJ  
M.Sc. Candidate in Intelligent Systems