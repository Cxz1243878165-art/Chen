# ══════════════════════════════════════════════════════════════════════
# NVDA Volatility & SDE Explorer
# "Comparing Black–Scholes and Heston Models Using NVDA Option Data"
# ══════════════════════════════════════════════════════════════════════

import os, time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm
from scipy.optimize import brentq, least_squares

# ─── App config ───────────────────────────────────────────────────────
st.set_page_config(page_title="NVDA Volatility & SDE Explorer", layout="wide", page_icon="📈")

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

DEFAULT_DATA_DIR = BASE_DIR
DEFAULT_DATA_FILE = "nvda_options_jan2026.csv.gz"

# ─── Navigation ───────────────────────────────────────────────────────
NAV_LABELS = [
    "🏠 Home",
    "📖 Theory and Notes",
    "📈 SDE Visualiser",
    "⏱️ Performance and Benchmark",
    "📉 Vol Smile Explorer",
]

if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = NAV_LABELS[0]

if "__page_override" in st.session_state:
    ov = st.session_state.pop("__page_override")
    if ov in NAV_LABELS:
        st.session_state["nav_page"] = ov

def _navigate_to(label: str):
    st.session_state["__page_override"] = label

def set_state(**kwargs):
    for k, v in kwargs.items():
        st.session_state[k] = v

# ─── Shared utilities ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True, on_bad_lines="skip")
    df.columns = [c.strip().lstrip("[").rstrip("]") for c in df.columns]
    for col in ["QUOTE_DATE", "EXPIRE_DATE"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    numeric_cols = [
        "UNDERLYING_LAST", "DTE", "C_DELTA", "C_GAMMA", "C_VEGA", "C_THETA",
        "C_RHO", "C_IV", "C_VOLUME", "C_LAST", "C_BID", "C_ASK",
        "STRIKE", "P_BID", "P_ASK", "P_LAST", "P_DELTA", "P_GAMMA",
        "P_VEGA", "P_THETA", "P_RHO", "P_IV", "P_VOLUME",
        "STRIKE_DISTANCE", "STRIKE_DISTANCE_PCT",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def _subset(df, date, expiry):
    d = pd.Timestamp(date).normalize()
    e = pd.Timestamp(expiry).normalize()
    return df.loc[
        (df["QUOTE_DATE"].dt.normalize() == d)
        & (df["EXPIRE_DATE"].dt.normalize() == e)
    ].copy()

def _atm_iv(sub, S0):
    mny = sub["STRIKE"] / S0
    near = sub.loc[mny.between(0.95, 1.05) & sub["C_IV"].notna(), "C_IV"]
    return float(np.median(near)) if not near.empty else float(np.nanmedian(sub["C_IV"]))

def help_block(title, body_md, presets=None):
    with st.expander(title):
        st.markdown(body_md)
        if presets:
            cols = st.columns(min(4, len(presets)))
            for i, p in enumerate(presets):
                with cols[i % len(cols)]:
                    st.button(p.get("label", "Use preset"), on_click=p.get("on_click"))

# ─── CSS styling ──────────────────────────────────────────────────────
def _inject_css():
    st.markdown("""
    <style>
    .panel {
        padding: 0.9rem 1.1rem;
        margin: 0.8rem 0;
        border-radius: 8px;
        border: 1px solid #d9dee7;
        background: #fafbfc;
    }

    .math-panel {
        padding: 0.9rem 1.1rem;
        margin: 0.8rem 0;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        background: #f4f8ff;
    }

    .warning-panel {
        padding: 0.9rem 1.1rem;
        margin: 0.8rem 0;
        border-radius: 8px;
        border-left: 4px solid #f59e0b;
        background: #fff8eb;
    }

    .app-banner {
        background: #eef3f8;
        border: 1px solid #d6e0ea;
        border-radius: 12px;
        padding: 1.6rem 1.4rem;
        text-align: center;
        margin-bottom: 1.2rem;
    }

    .app-banner h1 {
        margin: 0 0 0.35rem 0;
        font-size: 2rem;
    }

    .app-banner p {
        margin: 0;
        color: #4b5563;
        font-size: 1rem;
        line-height: 1.6;
    }

    .feature-tile {
        border: 1px solid #e1e6ee;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        background: white;
        min-height: 100%;
    }

    .feature-tile h4 {
        margin: 0 0 0.45rem 0;
    }

    .feature-tile p {
        margin: 0;
        color: #5b6470;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    </style>
    """, unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════════════
# PRICING ENGINES
# ══════════════════════════════════════════════════════════════════════

def bs_price(S0, K, r, q, T, sigma, option_type='call'):
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)

def bs_iv(price, S0, K, r, q, T, option_type='call'):
    try:
        return brentq(lambda s: bs_price(S0, K, r, q, T, s, option_type) - price, 0.01, 5.0, xtol=1e-6)
    except Exception:
        return np.nan

def heston_cf(phi, S0, r, q, T, v0, kappa, theta, sigma, rho, j=1):
    """Heston characteristic function — Little Heston Trap formulation."""
    u = 0.5 if j == 1 else -0.5
    b = (kappa - rho * sigma) if j == 1 else kappa
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

def heston_call_price(S0, K, r, q, T, v0, kappa, theta, sigma, rho):
    dphi = 0.01
    phi = np.arange(dphi, 100.0, dphi)
    P1 = 0.5 + (1.0 / np.pi) * np.sum(
        np.real(np.exp(-1j * phi * np.log(K))
                * heston_cf(phi, S0, r, q, T, v0, kappa, theta, sigma, rho, j=1) / (1j * phi))
    ) * dphi
    P2 = 0.5 + (1.0 / np.pi) * np.sum(
        np.real(np.exp(-1j * phi * np.log(K))
                * heston_cf(phi, S0, r, q, T, v0, kappa, theta, sigma, rho, j=2) / (1j * phi))
    ) * dphi
    return max(S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2, 0.0)


# ══════════════════════════════════════════════════════════════════════
# SIMULATION ENGINES
# ══════════════════════════════════════════════════════════════════════

def simulate_gbm_paths(S0, T, mu, sigma, M, N):
    dt = T / N
    t_local = np.linspace(0, T, N)
    paths = np.zeros((M, N))
    for i in range(M):
        W = np.cumsum(np.random.randn(N)) * np.sqrt(dt)
        paths[i] = S0 * np.exp((mu - 0.5 * sigma**2) * t_local + sigma * W)
    return paths

def simulate_heston_paths(S0, T, r, v0, kappa, theta, sigma_v, rho, M, N, return_variance=False):
    dt = T / N
    paths = np.zeros((M, N))
    vars_ = np.zeros((M, N)) if return_variance else None
    for j in range(M):
        S = np.zeros(N); v = np.zeros(N)
        S[0], v[0] = S0, max(v0, 1e-12)
        for i in range(1, N):
            z1 = np.random.normal()
            z2 = rho * z1 + np.sqrt(max(1 - rho**2, 0.0)) * np.random.normal()
            v_prev = max(v[i-1], 1e-12)
            v[i] = abs(v_prev + kappa * (theta - v_prev) * dt + sigma_v * np.sqrt(v_prev * dt) * z2)
            S[i] = S[i-1] * np.exp((r - 0.5 * v_prev) * dt + np.sqrt(v_prev * dt) * z1)
        paths[j] = S
        if return_variance:
            vars_[j] = v
    return (paths, vars_) if return_variance else paths


# ══════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════

def page_home():
    _inject_css()

    st.markdown("""
    <div class="app-banner">
        <h1>NVDA Volatility and Option Pricing Explorer</h1>
        <p>
            This interactive application was developed alongside the dissertation to
            support real-time exploration of the Black--Scholes and Heston models
            using NVIDIA option data. The tool combines theoretical notes, simulation,
            model comparison, and volatility-surface visualisation in one interface.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Explore the application")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="feature-tile">
            <h4>📖 Theory and Notes</h4>
            <p>
                Concise explanations of Brownian motion, Itô calculus, geometric
                Brownian motion, Black--Scholes pricing, Heston dynamics, and
                calibration ideas used throughout the dissertation.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.button(
            "Go to Theory",
            key="go_theory",
            on_click=_navigate_to,
            args=("📖 Theory and Notes",)
        )

    with col2:
        st.markdown("""
        <div class="feature-tile">
            <h4>📈 SDE Visualiser</h4>
            <p>
                Simulate and compare Black--Scholes and Heston paths, and inspect
                how stochastic volatility changes return dispersion and distributional shape.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.button(
            "Go to Visualiser",
            key="go_visualiser",
            on_click=_navigate_to,
            args=("📈 SDE Visualiser",)
        )

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("""
        <div class="feature-tile">
            <h4>⏱️ Performance and Benchmark</h4>
            <p>
                Compare runtime and computational cost across model settings, with
                a focus on the trade-off between analytical simplicity and empirical realism.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.button(
            "Go to Benchmark",
            key="go_benchmark",
            on_click=_navigate_to,
            args=("⏱️ Performance and Benchmark",)
        )

    with col4:
        st.markdown("""
        <div class="feature-tile">
            <h4>📉 Volatility Smile Explorer</h4>
            <p>
                Load NVDA option data and compare market implied volatility with
                Black--Scholes and calibrated Heston outputs across strike prices.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.button(
            "Go to Smile Explorer",
            key="go_smile",
            on_click=_navigate_to,
            args=("📉 Vol Smile Explorer",)
        )

    st.markdown("---")
    st.caption("BSc Thesis Companion Tool: NVDA options, stochastic differential equations, and volatility modelling")


# ══════════════════════════════════════════════════════════════════════
# PAGE: THEORY & NOTES
# ══════════════════════════════════════════════════════════════════════
THEORY_SECTIONS = [
    ("brownian",     "Standard Brownian Motion",    "Foundation of continuous-time randomness.",  "🌊"),
    ("ito",          "Itô's Lemma",                 "The stochastic chain rule and its role in model derivation.",       "🧮"),
    ("gbm_def",      "Geometric Brownian Motion",   "Definition, closed-form solution, and lognormal price dynamics.",      "📈"),
    ("gbm_var",      "GBM: Different Variances",    "Same noise, different σ to compare paths.",  "📊"),
    ("bs_theory",    "Black–Scholes (Theory)",      "Risk-neutral pricing, PDE intuition, and the European call formula.",        "📘"),
    ("bs_tools",     "BS Mini-Calculator",          "Quick prices for C/P with d₁, d₂.",        "🛠️"),
    ("limitations",  "Limitations & Smile",         "Why BS is too rigid; NVDA smile evidence.", "⚠️"),
    ("meanrev",   "Mean-Reverting Process",      "OU intuition & link to Heston.",            "🔁"),
    ("ou_explorer",      "OU Explorer",                 "Interactive simulation of mean-reverting dynamics.",    "🧪"),
    ("heston", "Heston Model",                "Two-factor dynamics, characteristic-function pricing, and market intuition.", "🌀"),
    ("heston_calib", "Heston Calibration",          "Parameter fitting, objective functions, and practical calibration issues.", "🔧"),
    ("glossary", "Glossary", "Summary definitions for notation and financial concepts used in the app.", "🔎")
]


def _set_theory_view(sid):
    st.session_state["theory_view"] = sid

# ─── Individual theory sections ───────────────────────────────────────

def _section_brownian():
    st.header("Brownian Motion: Path Simulation")
    st.markdown(
        '<div class="definition box">Brownian motion starts at zero, has independent Gaussian increments, and continuous sample paths.</div>',
        unsafe_allow_html=True
    )
    st.latex(r"W_0 = 0,\qquad W_t - W_s \sim \mathcal{N}(0, t-s)")
    st.write(
        "Use the controls below to explore how the time horizon, number of paths, "
        "and discretisation level affect simulated Brownian trajectories."
    )

    left, right = st.columns(2)
    with left:
        T = st.slider("Time horizon T", 0.25, 5.0, 1.0, 0.25, key="bm_T")
        n_paths = st.slider("Number of paths", 1, 30, 5, 1, key="bm_paths")
    with right:
        n_steps = st.slider("Number of time steps", 100, 4000, 800, 100, key="bm_steps")
        seed = st.number_input("Random seed", min_value=0, value=0, step=1, key="bm_seed")

    show_hist = st.checkbox("Show increment distribution", value=True, key="bm_hist")

    rng = np.random.default_rng(int(seed))
    dt = T / n_steps
    t_grid = np.linspace(0, T, n_steps + 1)
    increments = rng.normal(0, np.sqrt(dt), (n_paths, n_steps))
    paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)], axis=1)

    fig, ax = plt.subplots()
    for i in range(n_paths):
        ax.plot(t_grid, paths[i], lw=1.4, alpha=0.9)
    ax.set_title(f"Simulated Brownian Motion Paths (T={T:g}, steps={n_steps}, paths={n_paths})")
    ax.set_xlabel("t")
    ax.set_ylabel("W(t)")
    ax.grid(True)
    st.pyplot(fig)

    if show_hist:
        fig_h, ax_h = plt.subplots()
        flat_inc = increments.ravel()
        ax_h.hist(flat_inc, bins=50, density=True, alpha=0.85)
        xg = np.linspace(flat_inc.min(), flat_inc.max(), 400)
        ax_h.plot(xg, norm.pdf(xg, 0, np.sqrt(dt)), ls="--", label="Normal density")
        ax_h.set_title("Distribution of simulated increments")
        ax_h.set_xlabel("Increment value")
        ax_h.set_ylabel("Density")
        ax_h.legend()
        ax_h.grid(True)
        st.pyplot(fig_h)

        st.caption(
            f"Sample increment mean = {flat_inc.mean():.4f}; sample variance = {flat_inc.var():.5f}. "
            f"The theoretical variance for Brownian increments is dt = {dt:.5f}."
        )
def _section_ito():
    st.header("Itô's Lemma")
    st.markdown('<div class="box remark">Itô process.</div>', unsafe_allow_html=True)
    st.latex(r"dX_t = a(X_t,t)\,dt + b(X_t,t)\,dW_t")
    st.info(r"**Itô's Lemma.** If $X_t$ is an Itô process and $f(x,t)$ is smooth:")
    st.latex(r"df(X_t,t) = \left(f_t + a f_x + \tfrac{1}{2} b^2 f_{xx}\right)dt + b f_x\, dW_t")

def _section_gbm_def():
    st.header("Geometric Brownian Motion (definition & solution)")
    st.latex(r"dS_t = \mu\,S_t\,dt + \sigma\,S_t\,dW_t, \qquad S_0 > 0")
    st.markdown("Applying Itô's Lemma to $f(S_t) = \\ln S_t$:")
    st.latex(r"d(\ln S_t) = \left(\mu - \tfrac{1}{2}\sigma^2\right)dt + \sigma\,dW_t")
    st.latex(r"S_t = S_0 \exp\!\left[\left(\mu - \tfrac{1}{2}\sigma^2\right)t + \sigma W_t\right]")
    st.markdown("Hence $\\ln S_t \\sim \\mathcal{N}(\\ln S_0 + (\\mu - \\frac{1}{2}\\sigma^2)t,\\; \\sigma^2 t)$ — **log-normal prices**.")
    st.caption("Under the risk-neutral measure Q: replace μ with r − q.")

def _section_gbm_var():
    st.header("GBM: Realisations with Different Variances")
    st.markdown("Set μ = 0.05 and plot one GBM path for each σ on the **same Brownian path** (NVDA-like parameters).")
    c1, c2, c3 = st.columns(3)
    with c1:
        S0 = st.number_input("S₀", value=183.0, step=10.0, key="rv_S0")
        mu = st.number_input("Drift μ", value=0.05, step=0.01, key="rv_mu")
        T = st.number_input("Horizon T", value=1.0, step=0.1, key="rv_T")
    with c2:
        N = st.slider("Steps N", 200, 4000, 2000, 100, key="rv_N")
        seed = st.number_input("Seed", min_value=0, value=42, step=1, key="rv_seed")
    with c3:
        s1 = st.number_input("σ₁ (low)", value=0.25, step=0.05, key="rv_s1")
        s2 = st.number_input("σ₂ (medium)", value=0.40, step=0.05, key="rv_s2")
        s3 = st.number_input("σ₃ (high)", value=0.60, step=0.05, key="rv_s3")

    rng = np.random.default_rng(int(seed))
    dt = T / N; t = np.linspace(0, T, N+1)
    dW = rng.normal(0, np.sqrt(dt), N)
    W = np.concatenate([[0.0], np.cumsum(dW)])

    def gbm(S0, mu, sig, t, W):
        return S0 * np.exp((mu - 0.5*sig**2)*t + sig*W)

    fig, ax = plt.subplots()
    ax.plot(t, gbm(S0, mu, s1, t, W), label=f"σ = {s1:.2f}")
    ax.plot(t, gbm(S0, mu, s2, t, W), label=f"σ = {s2:.2f}")
    ax.plot(t, gbm(S0, mu, s3, t, W), label=f"σ = {s3:.2f}")
    ax.set_title(f"NVDA-like GBM paths with different σ (μ = {mu:.2f})")
    ax.set_xlabel("t"); ax.set_ylabel("S(t)"); ax.grid(True); ax.legend()
    st.pyplot(fig)
    st.caption("Same Brownian path for all; only σ changes. Higher σ → wider dispersion.")

def _section_bs_theory():
    st.header("Black–Scholes (Theory, PDE & Solution)")
    st.markdown('<div class="box definition">Risk-neutral dynamics.</div>', unsafe_allow_html=True)
    st.latex(r"dS_t = (r-q)S_t\,dt + \sigma S_t\, dW_t^{\mathbb{Q}}")
    st.markdown('<div class="box theorem">Black–Scholes PDE.</div>', unsafe_allow_html=True)
    st.latex(r"V_t + (r-q)S V_S + \tfrac{1}{2}\sigma^2 S^2 V_{SS} - rV = 0")
    st.markdown('<div class="box theorem">European call price.</div>', unsafe_allow_html=True)
    st.latex(r"C = S_0 e^{-q\tau}\Phi(d_1) - Ke^{-r\tau}\Phi(d_2)")
    st.latex(r"d_1 = \frac{\ln(S_0/K) + (r - q + \frac{1}{2}\sigma^2)\tau}{\sigma\sqrt{\tau}}, \quad d_2 = d_1 - \sigma\sqrt{\tau}")

def _section_bs_calc():
    st.header("Black–Scholes Mini-Calculator")
    st.markdown("Use NVDA default parameters from the thesis.")
    S0 = st.number_input("Spot S₀", value=183.12, key="bc_S")
    K  = st.number_input("Strike K", value=185.0, key="bc_K")
    r  = st.number_input("Risk-free r", value=0.043, key="bc_r")
    q  = st.number_input("Dividend q", value=0.0, key="bc_q")
    vol= st.number_input("Vol σ", value=0.36, key="bc_v")
    tau= st.number_input("Time τ (yrs)", value=60/365, key="bc_t")

    if st.button("Compute BS Price"):
        from math import log, sqrt, exp
        d1 = (log(S0/K) + (r-q+0.5*vol**2)*tau) / (vol*sqrt(tau))
        d2 = d1 - vol*sqrt(tau)
        C = S0*exp(-q*tau)*norm.cdf(d1) - K*exp(-r*tau)*norm.cdf(d2)
        P = K*exp(-r*tau)*norm.cdf(-d2) - S0*exp(-q*tau)*norm.cdf(-d1)
        st.success(f"**Call** = ${C:.4f}  |  **Put** = ${P:.4f}  |  d₁ = {d1:.4f}, d₂ = {d2:.4f}")

def _section_limitations():
    st.header("Limitations & Volatility Smile — NVDA Evidence")
    st.markdown("""
    **Why Black–Scholes fails for NVDA:**
    - **Constant volatility** → flat IV surface; NVDA shows pronounced smile/skew
    - **Log-normal returns** → thin tails; NVDA returns exhibit fat tails and negative skew
    - **No volatility clustering** → BS has i.i.d. returns; NVDA shows clustering around earnings
    - **Term structure** → short maturities curve more; NVDA 30-day IV ≈ 36%, 150-day IV ≈ 44%
    """)
    m = np.linspace(0.6, 1.4, 161)
    logm = np.log(m)
    def smile(logm, base, beta, kappa):
        return base + beta*logm + kappa*(logm**2)
    iv_30  = smile(logm, 0.36, -0.20, 0.60)
    iv_60  = smile(logm, 0.42, -0.15, 0.45)
    iv_150 = smile(logm, 0.44, -0.08, 0.30)

    fig, ax = plt.subplots()
    ax.plot(m, iv_30,  label="30-day (NVDA)")
    ax.plot(m, iv_60,  label="60-day (NVDA)")
    ax.plot(m, iv_150, label="150-day (NVDA)")
    ax.axhline(0.36, ls="--", alpha=0.7, label="BS: flat σ (ATM 30d)")
    ax.set_xlabel("Moneyness K/F"); ax.set_ylabel("Implied Volatility")
    ax.set_title("NVDA Volatility Smile vs Black–Scholes Flat σ")
    ax.grid(True, alpha=0.3); ax.legend()
    st.pyplot(fig)
    st.caption("Shorter maturities show stronger curvature. BS cannot capture this structure with a single σ.")

def _section_meanrev():
    st.header("Mean-Reverting Process (Ornstein–Uhlenbeck)")
    st.latex(r"dX_t = \kappa(\theta - X_t)\,dt + \sigma\,dW_t, \qquad \kappa > 0")
    st.markdown("- κ: speed of reversion  |  θ: long-run level  |  σ: noise scale")
    st.markdown("**Link to Heston:** The CIR variance process is a mean-reverting square-root diffusion:")
    st.latex(r"dv_t = \kappa(\theta - v_t)\,dt + \sigma_v\sqrt{v_t}\,dW_t, \qquad v_t \ge 0")

def _section_ou_explorer():
    st.header("Ornstein–Uhlenbeck Explorer")
    c1, c2, c3 = st.columns(3)
    with c1:
        T = st.slider("Horizon T", 0.5, 10.0, 2.0, 0.5, key="ou_T")
        N = st.slider("Steps N", 100, 4000, 800, 100, key="ou_N")
        seed = st.number_input("Seed", min_value=0, value=1, step=1, key="ou_seed")
    with c2:
        kappa = st.slider("κ", 0.1, 8.0, 2.0, 0.1, key="ou_k")
        theta = st.slider("θ", -2.0, 2.0, 0.0, 0.1, key="ou_th")
        sigma = st.slider("σ", 0.05, 2.0, 0.6, 0.05, key="ou_s")
    with c3:
        x0 = st.slider("x₀", -3.0, 3.0, 1.5, 0.1, key="ou_x0")
        M = st.slider("Paths M", 1, 50, 8, 1, key="ou_M")

    rng = np.random.default_rng(int(seed))
    dt = T / N; t = np.linspace(0, T, N+1)
    Z = rng.normal(size=(M, N))
    X = np.empty((M, N+1)); X[:, 0] = x0
    for i in range(1, N+1):
        X[:, i] = X[:, i-1] + kappa*(theta - X[:, i-1])*dt + sigma*np.sqrt(dt)*Z[:, i-1]

    m_t = theta + (x0 - theta)*np.exp(-kappa*t)
    fig, ax = plt.subplots()
    ax.plot(t, X.T, lw=1.0, alpha=0.6)
    ax.plot(t, m_t, ls="--", lw=2.0, label="E[Xₜ]")
    ax.axhline(theta, ls=":", alpha=0.9, label="θ")
    ax.set_title(f"OU Paths (M={M})"); ax.set_xlabel("t"); ax.set_ylabel("X(t)")
    ax.grid(True); ax.legend()
    st.pyplot(fig)

def _section_heston():
    st.header("Heston Model — Dynamics & Pricing")
    st.markdown('<div class="box definition">Risk-neutral Heston dynamics.</div>', unsafe_allow_html=True)
    st.latex(r"""
    \begin{aligned}
    dS_t &= rS_t\,dt + \sqrt{v_t}\,S_t\,dW_t^{(1)} \\
    dv_t &= \kappa(\theta - v_t)\,dt + \sigma_v\sqrt{v_t}\,dW_t^{(2)}, \qquad
    dW^{(1)} \cdot dW^{(2)} = \rho\,dt
    \end{aligned}
    """)
    st.markdown("""
    **NVDA calibrated parameters (Table 4.4):**
    - v₀ = 0.0625 (25% vol) — current elevated variance
    - θ* = 0.0324 (18% vol) — long-run variance expectation
    - κ* = 3.2 — moderate mean reversion
    - σ = 0.52 — high vol-of-vol (pronounced smile)
    - ρ = −0.81 — strong leverage effect (downside skew)
    """)
    st.markdown('<div class="box theorem">European call via characteristic functions.</div>', unsafe_allow_html=True)
    st.latex(r"C = S_0 P_1 - Ke^{-rT} P_2, \qquad P_j = \tfrac{1}{2} + \tfrac{1}{\pi}\int_0^\infty \Re\!\left[\frac{e^{-i\varphi \ln K}\phi_j(\varphi)}{i\varphi}\right] d\varphi")

def _section_heston_calib():
    st.header("Heston Calibration")
    st.latex(r"\min_{\Theta}\; \frac{1}{2}\sum_{i=1}^N \left(P_i^{\text{mkt}} - P_i^{\text{model}}(\Theta)\right)^2")
    st.markdown("**Levenberg–Marquardt update:**")
    st.latex(r"\Delta\Theta = -(J^\top J + \mu I)^{-1} J^\top r")
    st.markdown("""
    **For NVDA (from thesis):**
    - High σ (0.52) → market priced in significant volatility risk
    - Strongly negative ρ (−0.81) → extreme downside skew
    - v₀ > θ* → market expected near-term volatility to decrease
    - RMSE ≈ $0.85 across 50 option contracts
    """)

def _section_glossary():
    st.header("Glossary")
    terms = {
        "implied volatility": "σ that makes BS price match the market price.",
        "historical volatility": "Annualised stdev of past log-returns.",
        "volatility smile/skew": "IV varies with strike; absent in BS, natural in Heston.",
        "leverage effect": "ρ < 0: prices fall → vol rises → left skew.",
        "mean reversion": "Tendency to revert to θ; OU/CIR processes.",
        "Feller condition": "2κθ ≥ σ² ensures v_t stays positive.",
        "characteristic function": "Fourier transform of distribution; core to Heston pricing.",
        "calibration": "Fitting model parameters to market option prices.",
        "vol-of-vol (σ_v)": "Controls curvature of the implied volatility smile.",
        "risk-neutral measure": "Q-measure under which discounted prices are martingales.",
    }
    for k, v in terms.items():
        st.markdown(f"**{k.capitalize()}** — {v}")

SECTION_RENDER = {
    "brownian": _section_brownian, "ito": _section_ito,
    "gbm_def": _section_gbm_def, "gbm_var": _section_gbm_var,
    "bs_theory": _section_bs_theory, "bs_calc": _section_bs_calc,
    "limitations": _section_limitations, "meanrev": _section_meanrev,
    "ou_explorer": _section_ou_explorer, "heston": _section_heston,
    "heston_calib": _section_heston_calib, "glossary": _section_glossary,
}

def page_theory():
    _inject_css()
    view = st.session_state.get("theory_view", None)
    if view is None:
        st.markdown("#### Jump to a section")
        rows = (len(THEORY_SECTIONS) + 2) // 3
        idx = 0
        for _ in range(rows):
            cols = st.columns(3)
            for col in cols:
                if idx >= len(THEORY_SECTIONS):
                    col.empty(); continue
                sid, title, desc, emoji = THEORY_SECTIONS[idx]
                with col:
                    st.markdown(f'<div class="card"><h4>{emoji} {title}</h4><p>{desc}</p>', unsafe_allow_html=True)
                    st.button("Open", key=f"o_{sid}", on_click=_set_theory_view, args=(sid,))
                    st.markdown("</div>", unsafe_allow_html=True)
                idx += 1
    else:
        st.button("⬅️ Back to Theory Home", on_click=_set_theory_view, args=(None,))
        st.markdown("---")
        SECTION_RENDER.get(view, lambda: st.info("Section not found."))()


# ══════════════════════════════════════════════════════════════════════
# PAGE: SDE VISUALISER
# ══════════════════════════════════════════════════════════════════════

def page_sde_visualiser():
    st.title("🧪 SDE Visualiser — Black–Scholes vs Heston (NVDA)")

    with st.sidebar:
        st.header("🧮 Global Parameters")
        S0 = st.number_input("Initial Price S₀ (NVDA)", value=183.12, key="S0")
        K  = st.number_input("Strike K", value=185.0, key="K")
        r  = st.slider("Risk-Free Rate r", 0.0, 0.1, 0.043, key="r_vis")
        T  = st.slider("Time Horizon (Years)", 0.5, 5.0, 1.0, key="T_vis")
        N  = st.slider("Time Steps", 100, 1000, 250, key="N_vis")
        M  = st.slider("Simulations (Monte Carlo)", 10, 2000, 250, key="M_vis")

        with st.expander("Black–Scholes Parameters"):
            mu    = st.slider("Drift μ", -0.1, 0.3, 0.05, key="mu_vis")
            sigma = st.slider("Volatility σ", 0.01, 1.0, 0.36, key="sigma_vis")

        with st.expander("Heston Parameters (NVDA defaults)"):
            v0      = st.slider("Initial Variance v₀", 0.01, 0.5, 0.0625, key="v0_vis")
            kappa   = st.slider("Mean Reversion κ", 0.1, 5.0, 3.2, key="kappa_vis")
            theta   = st.slider("Long-Term Variance θ", 0.01, 0.5, 0.0324, key="theta_vis")
            sigma_v = st.slider("Vol of Vol σ_v", 0.01, 1.0, 0.52, key="sigmav_vis")
            rho     = st.slider("Correlation ρ", -1.0, 1.0, -0.81, key="rho_vis")

    t_arr = np.linspace(0, T, N)
    gbm_paths = simulate_gbm_paths(S0, T, mu, sigma, M, N)
    heston_paths, heston_vars = simulate_heston_paths(S0, T, r, v0, kappa, theta, sigma_v, rho, M, N, return_variance=True)

    _eps = 1e-12
    gbm_norm = gbm_paths / max(S0, _eps)
    heston_norm = heston_paths / max(S0, _eps)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Simulated Price Paths (normalised to S₀)")
        logy = st.checkbox("Log y-scale", False)
        fig, ax = plt.subplots()
        show = min(12, M)
        idx = np.random.choice(M, show, replace=False)
        for i in idx:
            ax.plot(t_arr, gbm_norm[i], lw=1, alpha=0.7, color="steelblue")
        for i in idx:
            ax.plot(t_arr, heston_norm[i], lw=1, alpha=0.7, color="darkorange")
        ax.set_title(f"{show} sample paths — GBM (blue) vs Heston (orange)")
        ax.set_xlabel("Time"); ax.set_ylabel("S(t) / S₀")
        if logy: ax.set_yscale("log")
        ax.grid(True); ax.legend(["GBM", "Heston"])
        st.pyplot(fig)

    with col2:
        st.subheader("📊 Terminal Return Distribution")
        ret_gbm = np.log(np.maximum(gbm_paths[:, -1], _eps) / max(S0, _eps))
        ret_hes = np.log(np.maximum(heston_paths[:, -1], _eps) / max(S0, _eps))
        fig2, ax2 = plt.subplots()
        ax2.hist(ret_gbm, bins=40, alpha=0.5, density=True, label="GBM")
        ax2.hist(ret_hes, bins=40, alpha=0.5, density=True, label="Heston")
        x = np.linspace(min(ret_gbm.min(), ret_hes.min()), max(ret_gbm.max(), ret_hes.max()), 200)
        ax2.plot(x, norm.pdf(x, ret_gbm.mean(), ret_gbm.std()), ls="--", label="Normal (BS)")
        ax2.set_xlabel("Log Returns"); ax2.set_ylabel("Density")
        ax2.set_title("Terminal Log-Return Distributions")
        ax2.legend(); ax2.grid(True)
        st.pyplot(fig2)

        help_block("How to read this", (
            "- GBM → log-returns are Normal by construction\n"
            "- Heston → stochastic vol creates **fat tails** (more extreme returns)\n"
            "- For NVDA: high σ_v=0.52 and ρ=−0.81 produce left-skewed, heavy-tailed returns\n"
            "- Increase σ_v or |ρ| to see fatter tails"
        ), presets=[
            {"label": "Fat tails: σ_v=1.0, ρ=-0.9", "on_click": lambda: set_state(sigmav_vis=1.0, rho_vis=-0.9)},
            {"label": "Near-GBM: σ_v=0.05, ρ=0", "on_click": lambda: set_state(sigmav_vis=0.05, rho_vis=0.0)},
        ])

    # Variance dynamics
    st.subheader("🔄 Heston Variance Dynamics v(t)")
    fig3, ax3 = plt.subplots()
    show_v = min(8, M)
    for i in range(show_v):
        ax3.plot(t_arr, heston_vars[i], lw=1, alpha=0.6)
    ax3.axhline(theta, ls="--", color="red", lw=2, label=f"θ = {theta:.4f}")
    ax3.axhline(v0, ls=":", color="blue", lw=1.5, label=f"v₀ = {v0:.4f}")
    ax3.set_title("Heston Variance Process v(t)")
    ax3.set_xlabel("Time"); ax3.set_ylabel("v(t)")
    ax3.grid(True); ax3.legend()
    st.pyplot(fig3)
    st.caption("Variance mean-reverts to θ. For NVDA: v₀ > θ means near-term vol is elevated.")


# ══════════════════════════════════════════════════════════════════════
# PAGE: PERFORMANCE & BENCHMARK
# ══════════════════════════════════════════════════════════════════════

def page_performance():
    st.title("⚡ Performance & Benchmark")

    left, right = st.columns([1.2, 0.8])
    with left:
        st.subheader("Interactive Timing Test")
        bm_M = st.slider("Paths M", 10, 5000, 100, 10, key="bm_M")
        bm_N = st.slider("Steps N", 10, 5000, 100, 10, key="bm_N")

        if st.button("Run Performance Test"):
            start = time.perf_counter()
            _ = simulate_gbm_paths(183.12, 1.0, 0.05, 0.36, bm_M, bm_N)
            t_gbm = time.perf_counter() - start

            start = time.perf_counter()
            _ = simulate_heston_paths(183.12, 1.0, 0.043, 0.0625, 3.2, 0.0324, 0.52, -0.81, bm_M, bm_N)
            t_heston = time.perf_counter() - start

            st.success(f"GBM (M={bm_M}, N={bm_N}) → {t_gbm:.3f}s  |  Heston → {t_heston:.3f}s")

            fig, ax = plt.subplots()
            ax.bar(["GBM", "Heston"], [t_gbm, t_heston], color=["steelblue", "darkorange"])
            ax.set_ylabel("Time (seconds)"); ax.set_title("Runtime Comparison")
            ax.grid(True, axis='y')
            st.pyplot(fig)

    with right:
        st.subheader("Scaling Reference")
        Ns = np.array([50, 100, 200, 500, 1000, 2000])
        t_g = 0.002 * Ns / 100
        t_h = 0.008 * Ns / 100

        fig, ax = plt.subplots()
        ax.plot(Ns, t_g, 'o-', label="GBM (illustrative)")
        ax.plot(Ns, t_h, 's-', label="Heston (illustrative)")
        ax.set_xlabel("Time Steps N"); ax.set_ylabel("Runtime (s)")
        ax.set_title("Expected Scaling (M=100)")
        ax.grid(True); ax.legend()
        st.pyplot(fig)
        st.caption("Heston is ~3–5× slower per step due to correlated variance dynamics.")


# ══════════════════════════════════════════════════════════════════════
# PAGE: VOLATILITY SMILE EXPLORER
# ══════════════════════════════════════════════════════════════════════

def page_vol_smile():
    st.title("📊 Volatility Smile Explorer — NVDA Options")

    # --- Load data ---
    data_path = DEFAULT_DATA_DIR / DEFAULT_DATA_FILE
    if not data_path.exists():
        st.error(f"Data file not found: {data_path}")
        st.info("Run `python generate_nvda_data.py` to generate the dataset.")
        return

    df = load_clean(str(data_path))
    st.success(f"Loaded {len(df)} NVDA option records.")

    # --- Selectors ---
    dates = sorted(df["QUOTE_DATE"].dropna().dt.normalize().unique())
    sel_date = st.selectbox("Quote Date", dates, format_func=lambda x: pd.Timestamp(x).strftime("%Y-%m-%d"))

    df_date = df[df["QUOTE_DATE"].dt.normalize() == pd.Timestamp(sel_date).normalize()]
    expiries = sorted(df_date["EXPIRE_DATE"].dropna().dt.normalize().unique())
    sel_expiry = st.selectbox("Expiry Date", expiries, format_func=lambda x: pd.Timestamp(x).strftime("%Y-%m-%d"))

    sub = _subset(df, sel_date, sel_expiry)
    if sub.empty:
        st.warning("No data for this date/expiry combination."); return

    S0 = float(sub["UNDERLYING_LAST"].iloc[0])
    DTE = int(sub["DTE"].iloc[0])
    T = DTE / 365.0
    r = 0.043; q = 0.0

    st.markdown(f"**S₀ = {S0:.2f}  |  DTE = {DTE}  |  T = {T:.4f} years**")

    # --- Market IV ---
    iv_col = "C_IV"
    sub_iv = sub.loc[sub[iv_col].notna() & np.isfinite(sub[iv_col])].copy()
    if sub_iv.empty:
        st.warning("No valid IV data."); return

    # --- BS flat vol ---
    atm = _atm_iv(sub_iv, S0)
    st.markdown(f"**ATM implied volatility (BS):** σ = {atm:.4f} ({atm*100:.1f}%)")

    # --- Heston calibration ---
    st.subheader("Heston Calibration")
    with st.expander("Calibration Settings"):
        v0_init    = st.number_input("v₀ init", value=0.0625, format="%.4f", key="cal_v0")
        kappa_init = st.number_input("κ init", value=3.2, key="cal_k")
        theta_init = st.number_input("θ init", value=0.0324, format="%.4f", key="cal_th")
        sigma_init = st.number_input("σ_v init", value=0.52, key="cal_s")
        rho_init   = st.number_input("ρ init", value=-0.81, key="cal_rho")

    heston_params = None
    heston_ivs = []

    if st.button("Calibrate Heston Model"):
        strikes = sub_iv["STRIKE"].values
        mkt_prices = sub_iv["C_LAST"].values

        def residuals(params):
            v0_, kappa_, theta_, sigma_, rho_ = params
            v0_ = max(v0_, 1e-6); kappa_ = max(kappa_, 0.01)
            theta_ = max(theta_, 1e-6); sigma_ = max(sigma_, 0.01)
            rho_ = np.clip(rho_, -0.999, 0.999)
            res = []
            for K_i, mkt_p in zip(strikes, mkt_prices):
                try:
                    mdl_p = heston_call_price(S0, K_i, r, q, T, v0_, kappa_, theta_, sigma_, rho_)
                    res.append(mdl_p - mkt_p)
                except:
                    res.append(0.0)
            return np.array(res)

        x0 = [v0_init, kappa_init, theta_init, sigma_init, rho_init]
        bounds = ([1e-6, 0.01, 1e-6, 0.01, -0.999], [1.0, 10.0, 1.0, 3.0, 0.999])

        with st.spinner("Calibrating Heston model..."):
            result = least_squares(residuals, x0, bounds=bounds, method='trf', max_nfev=200)

        v0_c, kappa_c, theta_c, sigma_c, rho_c = result.x
        rmse = np.sqrt(np.mean(result.fun**2))

        st.session_state["heston_cal"] = {
            "v0": v0_c, "kappa": kappa_c, "theta": theta_c,
            "sigma": sigma_c, "rho": rho_c, "rmse": rmse,
            "strikes": strikes.tolist(), "T": T, "S0": S0,
        }

        st.success(f"Calibration complete! RMSE = ${rmse:.4f}")
        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | v₀ | {v0_c:.4f} ({np.sqrt(v0_c)*100:.1f}% vol) |
        | κ  | {kappa_c:.4f} |
        | θ  | {theta_c:.4f} ({np.sqrt(theta_c)*100:.1f}% vol) |
        | σ_v | {sigma_c:.4f} |
        | ρ  | {rho_c:.4f} |
        """)

    # --- Retrieve calibration ---
    cal = st.session_state.get("heston_cal", None)

    # --- Implied Volatility Plot ---
    st.subheader("Implied Volatility Comparison")

    fig, ax = plt.subplots(figsize=(10, 5))

    # Market IVs
    mkt_strikes = sub_iv["STRIKE"].values
    mkt_ivs = sub_iv[iv_col].values * 100
    ax.scatter(mkt_strikes, mkt_ivs, s=40, zorder=5, label="Market IV", color="dodgerblue")

    # BS flat line
    ax.axhline(atm * 100, ls="--", color="red", lw=2, label=f"BS flat σ = {atm*100:.1f}%")

    # Heston IV curve
    if cal is not None:
        K_range = np.linspace(mkt_strikes.min(), mkt_strikes.max(), 50)
        h_ivs = []
        for K_i in K_range:
            try:
                hp = heston_call_price(cal["S0"], K_i, r, q, cal["T"],
                                       cal["v0"], cal["kappa"], cal["theta"],
                                       cal["sigma"], cal["rho"])
                iv = bs_iv(hp, cal["S0"], K_i, r, q, cal["T"])
                h_ivs.append(iv * 100 if not np.isnan(iv) else np.nan)
            except:
                h_ivs.append(np.nan)
        ax.plot(K_range, h_ivs, color="darkorange", lw=2.5, label="Heston IV")

    ax.set_xlabel("Strike Price"); ax.set_ylabel("Implied Volatility (%)")
    ax.set_title(f"NVDA Implied Volatility — {DTE}-Day Options")
    ax.grid(True, alpha=0.3); ax.legend()
    st.pyplot(fig)

    # --- Price Comparison Plot ---
    st.subheader("Option Price Comparison")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    mkt_prices = sub_iv["C_LAST"].values
    ax2.scatter(mkt_strikes, mkt_prices, s=40, label="Market Prices", color="dodgerblue", zorder=5)

    # BS prices
    bs_prices = [bs_price(S0, K_i, r, q, T, atm) for K_i in mkt_strikes]
    ax2.plot(mkt_strikes, bs_prices, color="red", lw=2, ls="--", label=f"BS (σ={atm:.3f})")

    # Heston prices
    if cal is not None:
        h_prices = []
        for K_i in mkt_strikes:
            try:
                hp = heston_call_price(cal["S0"], K_i, r, q, cal["T"],
                                       cal["v0"], cal["kappa"], cal["theta"],
                                       cal["sigma"], cal["rho"])
                h_prices.append(hp)
            except:
                h_prices.append(np.nan)
        ax2.plot(mkt_strikes, h_prices, color="darkorange", lw=2.5, label="Heston")

    ax2.set_xlabel("Strike Price"); ax2.set_ylabel("Option Price ($)")
    ax2.set_title(f"NVDA Call Prices — {DTE}-Day Options")
    ax2.grid(True, alpha=0.3); ax2.legend()
    st.pyplot(fig2)

    # --- Residuals ---
    if cal is not None:
        st.subheader("Calibration Diagnostics")
        h_res = []
        for K_i, mkt_p in zip(mkt_strikes, sub_iv["C_LAST"].values):
            try:
                hp = heston_call_price(cal["S0"], K_i, r, q, cal["T"],
                                       cal["v0"], cal["kappa"], cal["theta"],
                                       cal["sigma"], cal["rho"])
                h_res.append(mkt_p - hp)
            except:
                h_res.append(np.nan)

        fig3, ax3 = plt.subplots()
        ax3.axhline(0, ls="--", alpha=0.7)
        ax3.scatter(mkt_strikes, h_res, s=25, color="darkorange")
        moneyness = mkt_strikes / S0
        ax3.set_xlabel("Strike"); ax3.set_ylabel("Market − Heston ($)")
        ax3.set_title("Heston Pricing Residuals")
        ax3.grid(True)
        st.pyplot(fig3)


# ══════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:
    choice = st.radio("Navigation", NAV_LABELS,
                      index=NAV_LABELS.index(st.session_state["nav_page"]),
                      key="nav_page")

if choice == NAV_LABELS[0]:
    page_home()
elif choice == NAV_LABELS[1]:
    page_theory()
elif choice == NAV_LABELS[2]:
    page_sde_visualiser()
elif choice == NAV_LABELS[3]:
    page_performance()
elif choice == NAV_LABELS[4]:
    page_vol_smile()
