"""
European Put Option Pricing: Numerical Integration and FFT
==========================================================
Implements two methods for pricing European put options:

1. Numerical Integration — assumes log-normal (Black-Merton-Scholes) stock price dynamics.
2. FFT-based Pricing (Carr-Madan) — model-agnostic approach requiring only the
   characteristic function of the log-price. Supports BMS, Heston, and Variance Gamma models.

References:
    Carr, P. & Madan, D. (1999). "Option valuation using the fast Fourier transform."
    Journal of Computational Finance, 2(4), 61-73.
"""

import numpy as np
from time import time


# ---------------------------------------------------------------------------
# Section 1: Numerical Integration (Black-Merton-Scholes)
# ---------------------------------------------------------------------------

def lognormal_density(S, r, q, sig, S0, T):
    """
    Log-normal probability density of the stock price S at time T.

    Parameters
    ----------
    S   : float or array -- stock price(s) at time T
    r   : float -- continuously compounded risk-free rate
    q   : float -- continuous dividend yield
    sig : float -- volatility
    S0  : float -- current stock price
    T   : float -- time to maturity (years)

    Returns
    -------
    f : float or array -- density value(s)

    Note: Evaluating at S=0 produces NaN; use S=1e-8 as a practical lower bound.
    """
    drift = (r - q - 0.5 * sig**2) * T
    vol_sqrt_T = sig * np.sqrt(T)
    z = (np.log(S / S0) - drift) / vol_sqrt_T
    f = np.exp(-0.5 * z**2) / (S * vol_sqrt_T * np.sqrt(2 * np.pi))
    return f


def price_put_numerical(S0, K, r, q, sig, T, N):
    """
    Price a European put via trapezoidal-rule numerical integration under BMS.

    The put payoff (K - S)^+ is integrated against the risk-neutral log-normal
    density over [epsilon, K], then discounted.

    Parameters
    ----------
    S0  : float -- current stock price
    K   : float -- strike price
    r   : float -- risk-free rate
    q   : float -- dividend yield
    sig : float -- volatility
    T   : float -- time to maturity
    N   : int   -- number of grid points (higher N -> more accurate)

    Returns
    -------
    eta   : float -- grid spacing (K / N)
    price : float -- put price
    """
    eta = K / N
    discount = np.exp(-r * T)

    # Grid from epsilon to K in N steps (offset avoids S=0 singularity)
    S = 1.0 + np.arange(N) * eta
    density = lognormal_density(S, r, q, sig, S0, T)

    # Trapezoidal weights: half-weight on the first node
    weights = np.full(N, eta)
    weights[0] = eta / 2.0

    # Only nodes where S < K contribute to the put payoff
    payoff = np.where(S < K, K - S, 0.0)

    price = discount * np.sum(payoff * density * weights)
    return eta, price


# ---------------------------------------------------------------------------
# Section 2: Characteristic Functions (BMS, Heston, Variance Gamma)
# ---------------------------------------------------------------------------

def characteristic_function(u, params, S0, r, q, T, model):
    """
    Characteristic function phi(u) = E[e^{i*u * ln(S_T)}] under the risk-neutral measure.

    Parameters
    ----------
    u      : float or array -- frequency argument (may be complex)
    params : list           -- model-specific parameters (see below)
    S0     : float          -- current stock price
    r      : float          -- risk-free rate
    q      : float          -- dividend yield
    T      : float          -- time to maturity
    model  : str            -- 'BMS', 'Heston', or 'VG'

    Model parameters
    ----------------
    BMS    : [sigma]
    Heston : [kappa, theta, sigma, rho, v0]
               kappa -- mean-reversion speed of variance
               theta -- long-run variance
               sigma -- vol-of-vol
               rho   -- correlation between stock and variance Brownian motions
               v0    -- initial variance
    VG     : [sigma, nu, theta]
               sigma -- BMS-like volatility of the Gamma-time-changed Brownian motion
               nu    -- variance of the Gamma subordinator (controls kurtosis)
               theta -- drift of the Brownian motion (controls skewness)

    Returns
    -------
    phi : complex float or array
    """
    if model == 'BMS':
        sig = params[0]
        mu = np.log(S0) + (r - q - 0.5 * sig**2) * T
        a = sig * np.sqrt(T)
        phi = np.exp(1j * mu * u - 0.5 * (a * u)**2)

    elif model == 'Heston':
        kappa, theta, sigma, rho, v0 = params
        tmp = kappa - 1j * rho * sigma * u
        g = np.sqrt(sigma**2 * (u**2 + 1j * u) + tmp**2)
        pow1 = 2 * kappa * theta / sigma**2
        numer = (kappa * theta * T * tmp) / sigma**2 + 1j * u * T * r + 1j * u * np.log(S0)
        log_denom = pow1 * np.log(np.cosh(g * T / 2) + (tmp / g) * np.sinh(g * T / 2))
        tmp2 = (u**2 + 1j * u) * v0 / (g / np.tanh(g * T / 2) + tmp)
        phi = np.exp(numer - log_denom - tmp2)

    elif model == 'VG':
        sigma, nu, theta = params
        if nu == 0:
            # Limiting case: reduces to a Brownian motion with drift
            mu = np.log(S0) + (r - q - theta - 0.5 * sigma**2) * T
            phi = np.exp(1j * u * mu) * np.exp((1j * theta * u - 0.5 * sigma**2 * u**2) * T)
        else:
            mu = np.log(S0) + (r - q + np.log(1 - theta * nu - 0.5 * sigma**2 * nu) / nu) * T
            phi = np.exp(1j * u * mu) * (1 - 1j * nu * theta * u + 0.5 * nu * sigma**2 * u**2) ** (-T / nu)

    else:
        raise ValueError(f"Unknown model '{model}'. Choose from 'BMS', 'Heston', 'VG'.")

    return phi


# ---------------------------------------------------------------------------
# Section 3: FFT Option Pricing (Carr-Madan)
# ---------------------------------------------------------------------------

def fft_option_price(params, S0, K, r, q, T, alpha, eta, n, model):
    """
    Price European options using the Carr-Madan FFT method.

    The method exploits the fact that the (modified) option price has a known
    Fourier transform expressed in terms of the characteristic function of ln(S_T).
    A damping factor e^{alpha * k} is applied to ensure integrability:
      - alpha < -1  prices puts  (psi is integrable for alpha < -1)
      - alpha >  0  prices calls (psi is integrable for alpha > 0)

    Grid relationship (uncertainty principle):
        lambda * eta = 2 * pi / N   where lambda = log-strike spacing, eta = frequency spacing

    Parameters
    ----------
    params : list  -- model-specific parameters passed to characteristic_function()
    S0     : float -- current stock price
    K      : float -- target strike (returned as first element of km)
    r      : float -- risk-free rate
    q      : float -- dividend yield
    T      : float -- time to maturity
    alpha  : float -- damping exponent (< -1 for puts, > 0 for calls)
    eta    : float -- frequency-domain grid spacing
    n      : int   -- log2 of FFT size (N = 2^n)
    model  : str   -- 'BMS', 'Heston', or 'VG'

    Returns
    -------
    km     : array (N,) -- log-strikes
    prices : array (N,) -- option prices at each log-strike
    """
    N = 2**n
    lda = (2 * np.pi / N) / eta          # log-strike spacing
    beta = np.log(K)                      # anchor: K is the first log-strike

    discount = np.exp(-r * T)
    nu = np.arange(N) * eta              # frequency nodes

    # Damped characteristic function evaluated at (nu - i*(alpha+1))
    cf_vals = characteristic_function(
        nu - (alpha + 1) * 1j, params, S0, r, q, T, model
    )
    psi = cf_vals / ((alpha + 1j * nu) * (alpha + 1 + 1j * nu))

    # Trapezoidal weights (half-weight on first node)
    w = eta * np.ones(N)
    w[0] = eta / 2.0

    x = np.exp(-1j * beta * nu) * discount * psi * w
    y = np.fft.fft(x)

    km = beta + lda * np.arange(N)
    multiplier = np.exp(-alpha * km) / np.pi
    prices = multiplier * np.real(y)

    return km, prices


def price_put_fft(params, S0, K, r, q, T, model, alpha_vec, eta_vec, n_vec):
    """
    Compute put prices via FFT across a grid of (eta, n, alpha) combinations.

    Sweeping these parameters is a standard way to assess numerical stability:
    a stable price appears across a broad range of (alpha, eta, n) values.

    Returns
    -------
    matrix : ndarray, shape (M, 4) -- columns: [eta, n, alpha, put_price]
    """
    rows = []
    for eta in eta_vec:
        for n in n_vec:
            for alpha in alpha_vec:
                _, price_vec = fft_option_price(params, S0, K, r, q, T, alpha, eta, n, model)
                put = price_vec[0]       # price at strike K (first element since beta = log(K))
                rows.append([eta, n, alpha, put])
    return np.array(rows)


# ---------------------------------------------------------------------------
# Section 4: Demo
# ---------------------------------------------------------------------------

def demo_numerical_integration():
    """Convergence study: trapezoidal numerical integration for a BMS put."""
    print("=" * 60)
    print("NUMERICAL INTEGRATION -- European Put (BMS)")
    print("=" * 60)

    S0, K, r, q, sig, T = 100, 90, 0.04, 0.02, 0.25, 1.0
    print(f"S0={S0}, K={K}, r={r}, q={q}, sig={sig}, T={T}\n")
    print(f"{'n':>4}  {'N':>6}  {'eta':>8}  {'Put Price':>10}")
    print("-" * 36)

    t0 = time()
    for n in range(1, 16):
        N = 2**n
        eta, put = price_put_numerical(S0, K, r, q, sig, T, N)
        print(f"{n:>4}  {N:>6}  {eta:>8.4f}  {put:>10.4f}")
    print(f"\nElapsed: {time() - t0:.3f}s\n")


def demo_fft_pricing():
    """FFT put pricing stability check across BMS, Heston, and Variance Gamma."""
    print("=" * 60)
    print("FFT PRICING -- European Put (BMS / Heston / VG)")
    print("=" * 60)

    S0, K, r, q, T = 100, 80, 0.05, 0.01, 1.0
    eta_vec   = np.array([0.1, 0.25])
    n_vec     = np.array([6, 10])
    alpha_vec = np.array([-1.01, -1.25, -1.5, -1.75, -2.0, -5.0])

    models = {
        'BMS':    ([0.3],                         "sig=0.30"),
        'Heston': ([2.0, 0.05, 0.3, -0.7, 0.04], "kappa=2, theta=0.05, sigma=0.30, rho=-0.70, v0=0.04"),
        'VG':     ([0.3, 0.5, -0.4],              "sigma=0.30, nu=0.50, theta=-0.40"),
    }

    for model, (params, desc) in models.items():
        print(f"\nModel: {model}  ({desc})")
        print(f"{'eta':>6}  {'N':>6}  {'alpha':>6}  {'Put':>10}")
        print("-" * 36)
        t0 = time()
        matrix = price_put_fft(params, S0, K, r, q, T, model, alpha_vec, eta_vec, n_vec)
        for row in matrix:
            eta, n, alpha, put = row
            print(f"{eta:>6.2f}  2^{int(n):>2}  {alpha:>6.2f}  {put:>10.4f}")
        print(f"Elapsed: {time() - t0:.4f}s")


if __name__ == "__main__":
    demo_numerical_integration()
    demo_fft_pricing()
