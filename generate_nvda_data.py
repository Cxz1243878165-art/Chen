"""
Generate synthetic NVDA options data based on thesis calibration parameters.
Uses Heston characteristic function pricing to create realistic option surfaces.

Reference: Table 4.4 in thesis
  v0=0.0625, theta*=0.0324, kappa*=3.2, sigma=0.52, rho=-0.81
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime, timedelta

# ─── Heston characteristic function pricing ───────────────────────────
def heston_cf(phi, S0, r, q, T, v0, kappa, theta, sigma, rho, j=1):
    """Heston characteristic function (Little Heston Trap formulation)."""
    if j == 1:
        u = 0.5
        b = kappa - rho * sigma
    else:
        u = -0.5
        b = kappa

    a = kappa * theta
    x = np.log(S0)
    d = np.sqrt((rho * sigma * 1j * phi - b)**2 - sigma**2 * (2 * u * 1j * phi - phi**2))
    g = (b - rho * sigma * 1j * phi + d) / (b - rho * sigma * 1j * phi - d)

    C = (r - q) * 1j * phi * T + (a / sigma**2) * (
        (b - rho * sigma * 1j * phi + d) * T
        - 2.0 * np.log((1 - g * np.exp(d * T)) / (1 - g))
    )
    D = ((b - rho * sigma * 1j * phi + d) / sigma**2) * (
        (1 - np.exp(d * T)) / (1 - g * np.exp(d * T))
    )
    return np.exp(C + D * v0 + 1j * phi * x)


def heston_price(S0, K, r, q, T, v0, kappa, theta, sigma, rho, option_type='call'):
    """European option price via Heston model using numerical integration."""
    dphi = 0.01
    phi_max = 100.0
    phi = np.arange(dphi, phi_max, dphi)

    P1 = 0.5 + (1.0 / np.pi) * np.sum(
        np.real(np.exp(-1j * phi * np.log(K)) * heston_cf(phi, S0, r, q, T, v0, kappa, theta, sigma, rho, j=1) / (1j * phi))
    ) * dphi

    P2 = 0.5 + (1.0 / np.pi) * np.sum(
        np.real(np.exp(-1j * phi * np.log(K)) * heston_cf(phi, S0, r, q, T, v0, kappa, theta, sigma, rho, j=2) / (1j * phi))
    ) * dphi

    call_price = S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
    call_price = max(call_price, 0.0)

    if option_type == 'call':
        return call_price
    else:
        # Put-call parity
        return call_price - S0 * np.exp(-q * T) + K * np.exp(-r * T)


def bs_price(S0, K, r, q, T, sigma, option_type='call'):
    """Black-Scholes European option price."""
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)


def bs_iv(price, S0, K, r, q, T, option_type='call'):
    """Implied volatility from BS model via Brent's method."""
    try:
        iv = brentq(lambda s: bs_price(S0, K, r, q, T, s, option_type) - price, 0.01, 5.0, xtol=1e-6)
        return iv
    except Exception:
        return np.nan


def generate_nvda_options():
    """Generate realistic NVDA options data for multiple dates and expiries."""
    # --- Market parameters from thesis ---
    S0 = 183.12          # ATM price from Figure 3.1
    r = 0.043            # risk-free rate (~4.3%)
    q = 0.0              # NVDA dividend yield ≈ 0

    # Heston calibrated parameters (Table 4.4)
    v0 = 0.0625          # initial variance (25% vol)
    kappa = 3.2          # mean reversion speed
    theta = 0.0324       # long-run variance (18% vol)
    sigma = 0.52         # vol of vol
    rho = -0.81          # correlation

    # Quote dates and expiry dates
    quote_date = datetime(2026, 1, 10)
    expiries = [
        (datetime(2026, 2, 9),  30),    # 30 DTE
        (datetime(2026, 3, 11), 60),    # 60 DTE
        (datetime(2026, 4, 10), 90),    # 90 DTE
        (datetime(2026, 6, 9),  150),   # 150 DTE
    ]

    # Strikes: range around S0
    strike_range = np.arange(155, 215, 2.5)

    rows = []
    rng = np.random.default_rng(42)

    for expiry_date, dte in expiries:
        T = dte / 365.0
        for K in strike_range:
            # Heston prices
            c_heston = heston_price(S0, K, r, q, T, v0, kappa, theta, sigma, rho, 'call')
            p_heston = heston_price(S0, K, r, q, T, v0, kappa, theta, sigma, rho, 'put')

            # Add small noise to simulate real market bid-ask
            spread_c = max(0.05, c_heston * 0.02)
            spread_p = max(0.05, p_heston * 0.02)
            noise_c = rng.uniform(-0.15, 0.15)
            noise_p = rng.uniform(-0.15, 0.15)

            c_mid = max(c_heston + noise_c, 0.01)
            p_mid = max(p_heston + noise_p, 0.01)
            c_bid = max(c_mid - spread_c / 2, 0.01)
            c_ask = c_mid + spread_c / 2
            p_bid = max(p_mid - spread_p / 2, 0.01)
            p_ask = p_mid + spread_p / 2

            # Implied volatilities
            c_iv = bs_iv(c_mid, S0, K, r, q, T, 'call')
            p_iv = bs_iv(p_mid, S0, K, r, q, T, 'put')

            # Greeks (approximate)
            if not np.isnan(c_iv) and c_iv > 0:
                d1 = (np.log(S0/K) + (r - q + 0.5*c_iv**2)*T) / (c_iv*np.sqrt(T))
                d2 = d1 - c_iv * np.sqrt(T)
                c_delta = np.exp(-q*T) * norm.cdf(d1)
                c_gamma = np.exp(-q*T) * norm.pdf(d1) / (S0 * c_iv * np.sqrt(T))
                c_vega = S0 * np.exp(-q*T) * norm.pdf(d1) * np.sqrt(T) / 100
                c_theta_val = -(S0 * np.exp(-q*T) * norm.pdf(d1) * c_iv) / (2*np.sqrt(T)) / 365
            else:
                c_delta = c_gamma = c_vega = c_theta_val = np.nan

            if not np.isnan(p_iv) and p_iv > 0:
                d1p = (np.log(S0/K) + (r - q + 0.5*p_iv**2)*T) / (p_iv*np.sqrt(T))
                p_delta = np.exp(-q*T) * (norm.cdf(d1p) - 1)
                p_gamma = c_gamma if not np.isnan(c_gamma) else np.nan
                p_vega = c_vega if not np.isnan(c_vega) else np.nan
                p_theta_val = c_theta_val if not np.isnan(c_theta_val) else np.nan
            else:
                p_delta = p_gamma = p_vega = p_theta_val = np.nan

            # Volume (synthetic)
            moneyness = abs(K - S0) / S0
            base_vol = int(max(10, 5000 * np.exp(-10 * moneyness)))
            c_volume = rng.poisson(base_vol)
            p_volume = rng.poisson(int(base_vol * 0.8))

            rows.append({
                'QUOTE_DATE': quote_date,
                'EXPIRE_DATE': expiry_date,
                'UNDERLYING_LAST': S0,
                'DTE': dte,
                'STRIKE': K,
                'C_BID': round(c_bid, 2),
                'C_ASK': round(c_ask, 2),
                'C_LAST': round(c_mid, 2),
                'C_IV': round(c_iv, 4) if not np.isnan(c_iv) else np.nan,
                'C_DELTA': round(c_delta, 4) if not np.isnan(c_delta) else np.nan,
                'C_GAMMA': round(c_gamma, 6) if not np.isnan(c_gamma) else np.nan,
                'C_VEGA': round(c_vega, 4) if not np.isnan(c_vega) else np.nan,
                'C_THETA': round(c_theta_val, 4) if not np.isnan(c_theta_val) else np.nan,
                'C_RHO': 0.0,
                'C_VOLUME': c_volume,
                'P_BID': round(p_bid, 2),
                'P_ASK': round(p_ask, 2),
                'P_LAST': round(p_mid, 2),
                'P_IV': round(p_iv, 4) if not np.isnan(p_iv) else np.nan,
                'P_DELTA': round(p_delta, 4) if not np.isnan(p_delta) else np.nan,
                'P_GAMMA': round(p_gamma, 6) if not np.isnan(p_gamma) else np.nan,
                'P_VEGA': round(p_vega, 4) if not np.isnan(p_vega) else np.nan,
                'P_THETA': round(p_theta_val, 4) if not np.isnan(p_theta_val) else np.nan,
                'P_RHO': 0.0,
                'P_VOLUME': p_volume,
                'STRIKE_DISTANCE': round(K - S0, 2),
                'STRIKE_DISTANCE_PCT': round((K - S0) / S0 * 100, 2),
            })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    print("Generating NVDA options data...")
    df = generate_nvda_options()
    outpath = "data/nvda_options_jan2026.csv.gz"
    df.to_csv(outpath, index=False, compression='gzip')
    print(f"Saved {len(df)} rows to {outpath}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample:\n{df.head(3).to_string()}")
    print(f"\nExpiry dates: {df['EXPIRE_DATE'].unique()}")
    print(f"Strike range: {df['STRIKE'].min()} — {df['STRIKE'].max()}")
    print(f"DTE values: {sorted(df['DTE'].unique())}")
