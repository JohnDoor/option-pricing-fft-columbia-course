# Option Pricing: Numerical Integration and FFT

Implementation of two classical methods for pricing European options, developed as part of the
[Financial Engineering and Risk Management Specialization](https://www.coursera.org/specializations/financialengineering)
by Columbia University on Coursera (course: *Computational Methods in Pricing and Model Calibration*).

---

## Core idea: Risk-neutral pricing

Under the risk-neutral measure, the price of a European option with payoff $g(S_T)$ is:

$$V_0 = e^{-rT} \int_0^\infty g(S) \, f^Q(S) \, dS$$

where $f^Q(S)$ is the risk-neutral density of the stock price at maturity $T$, and $r$ is the
risk-free rate. The two methods below differ in how they evaluate this integral.

---

## Method 1 — Trapezoidal numerical integration

When the risk-neutral density is known in closed form (e.g. log-normal under Black-Merton-Scholes),
the integral can be discretized directly using the **trapezoidal rule**.

The domain $[0, K]$ is partitioned into $N$ equally spaced nodes with step size $\eta = K/N$.
The put price becomes:

$$P_0 \approx e^{-rT} \sum_{j=0}^{N-1} (K - S_j)^+ \, f^Q(S_j) \, w_j$$

where $w_0 = \eta/2$ (half-weight on the boundary) and $w_j = \eta$ otherwise.

As $N$ grows (equivalently, $\eta \to 0$), the approximation converges to the true price.
The notebook `optionPricingViaNumericalIntegration.ipynb` runs this for $N = 2^n$,
$n = 1, \ldots, 15$, and shows convergence of the put price.

**Limitation:** requires knowing $f^Q$ in closed form — not always available.

---

## Method 2 — Fast Fourier Transform (Carr-Madan)

When the density is not available but the **characteristic function** of $\ln S_T$ is known,
option prices can be computed via FFT. This covers a wide class of models including stochastic
volatility and jump-diffusion models.

The characteristic function is defined as:

$$\phi(u) = \mathbb{E}^Q\!\left[e^{iu \ln S_T}\right]$$

To ensure integrability, a damping factor $e^{\alpha k}$ is applied to the option price as a
function of log-strike $k = \ln K$. Its Fourier transform has the closed-form expression:

$$\Psi(u) = \frac{e^{-rT} \, \phi(u - (\alpha+1)i)}{(\alpha + iu)(\alpha + 1 + iu)}$$

Inverting via FFT over a uniform frequency grid with spacing $\eta$ and $N = 2^n$ points yields
option prices at $N$ log-strikes simultaneously, with log-strike spacing $\lambda = 2\pi / (N\eta)$.

**Sign convention for $\alpha$:**
- $\alpha < -1$ — prices **puts**
- $\alpha > 0$ — prices **calls**

Numerical stability is assessed by checking that the price is consistent across different
combinations of $(\alpha, \eta, n)$.

### Supported models

| Model | Parameters | Key feature |
|---|---|---|
| Black-Merton-Scholes (BMS) | $\sigma$ | Log-normal, analytical benchmark |
| Heston | $\kappa, \theta, \sigma, \rho, v_0$ | Stochastic volatility, captures vol smile |
| Variance Gamma (VG) | $\sigma, \nu, \theta$ | Pure-jump process, heavier tails and skew |

---

## Repository structure

```
option_pricing.py                          # Clean standalone implementation
optionPricingViaNumericalIntegration.ipynb # Numerical integration walkthrough
optionPricingViaFFT.ipynb                  # FFT pricing walkthrough
exampleUsingFFT_1.ipynb                    # FFT example: call pricing
exampleUsingFFT_2.ipynb                    # FFT example: sensitivity analysis
plotLogNormalForVariousSpot.ipynb          # Log-normal density visualisation
plot_put_call.ipynb                        # Put/call payoff visualisation
```

---

## Usage

```python
from option_pricing import price_put_numerical, price_put_fft
import numpy as np

# --- Numerical integration (BMS) ---
S0, K, r, q, sig, T = 100, 90, 0.04, 0.02, 0.25, 1.0
eta, put = price_put_numerical(S0, K, r, q, sig, T, N=2**12)
print(f"Put price (numerical): {put:.4f}")

# --- FFT pricing ---
S0, K, r, q, T = 100, 80, 0.05, 0.01, 1.0
alpha_vec = np.array([-1.5, -2.0, -5.0])
eta_vec   = np.array([0.1, 0.25])
n_vec     = np.array([6, 10])

# BMS
bms_matrix = price_put_fft([0.3], S0, K, r, q, T, 'BMS', alpha_vec, eta_vec, n_vec)

# Heston: [kappa, theta, sigma, rho, v0]
heston_matrix = price_put_fft([2.0, 0.05, 0.3, -0.7, 0.04], S0, K, r, q, T, 'Heston', alpha_vec, eta_vec, n_vec)

# Variance Gamma: [sigma, nu, theta]
vg_matrix = price_put_fft([0.3, 0.5, -0.4], S0, K, r, q, T, 'VG', alpha_vec, eta_vec, n_vec)
```

Run the full demo:

```bash
python option_pricing.py
```

---

## Dependencies

```
numpy
```

---

## Reference

Carr, P. & Madan, D. (1999). *Option valuation using the fast Fourier transform.*
Journal of Computational Finance, 2(4), 61–73.
