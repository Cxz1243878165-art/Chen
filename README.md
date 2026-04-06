# 📈 NVDA Volatility & SDE Explorer — Black–Scholes vs Heston

An interactive Tool application to explore stochastic differential equations (SDEs) in finance.  
The tool demonstrates the Black–Scholes (GBM) and Heston models through simulations, performance benchmarks, and volatility smile analysis using NVIDIA (NVDA) option data from January 2026.

---

## 🚀 Features

- **SDE Visualiser**: Simulate asset price paths under GBM and Heston; compare terminal return distributions; visualise variance dynamics.
- **Performance & Benchmark**: Run timing tests for GBM vs Heston simulations; understand computational scaling.
- **Volatility Smile Explorer**: Load NVDA option data; compare market implied volatility against Black–Scholes Model and Heston Model; inspect calibration diagnostics.
- **Theory & Notes**: Interactive reference covering Brownian motion, Itô's Lemma, GBM, Black–Scholes, OU processes, Heston model, and calibration.

---

## 🛠 Installation & Running Locally

```bash
git clone https://github.com/YOUR_USERNAME/sde-visualizer-nvda.git
cd sde-visualizer-nvda

# Generate NVDA options data
python generate_nvda_data.py

# Run the app
streamlit run app_sde_visualizer_nvda.py
```

## 📦 Dependencies

- Python 3.9+
- streamlit, numpy, pandas, matplotlib, scipy

Install via: `pip install -r requirements.txt`

## 📊 NVDA Calibration Parameters (Table 4.4)

| Parameter | Description | Calibrated Value |
|-----------|-------------|-----------------|
| v₀ | Initial Variance | 0.0625 (25.0% vol) |
| θ* | Long-Run Variance | 0.0324 (18.0% vol) |
| κ* | Mean Reversion Speed | 3.2 |
| σ  | Vol of Vol | 0.52 |
| ρ  | Correlation | −0.81 |
