# Fixed Income Analytics

A self-contained Python library for fixed income analytics, built from mathematical first principles and validated against real U.S. Treasury market data from the Federal Reserve (FRED).
Each module corresponds to a formal mathematical derivation in the companion document [Fixed Income Analytics (Palma, 2026)](docs/fixed_income_analytics.pdf), a Jupyter notebook with interactive visualizations, and a Python class with full docstrings.
---

## Modules

| Module | Python | Notebook | Description |
|--------|--------|----------|-------------|
| 1. Bond Pricing | `src/bonds.py` | `01_bond_pricing.ipynb` | Pricing, YTM, Duration, Convexity, DV01 |
| 2. Yield Curve | `src/curves.py` | `02_yield_curve.ipynb` | Bootstrapping, Spline, NSS calibration |
| 3. Sensitivity | `src/sensitivity.py` | `03_sensitivity.ipynb` | Parallel DV01, Key Rate Durations, Portfolio |
| 4. PCA | `src/pca.py` | `04_pca.ipynb` | Level/Slope/Curvature, Factor risk, NSS alignment |
| 5. IRS Pricing | `src/swaps.py` | `05_irs.ipynb` | Swap rate K*, FRA decomposition, Curve trade |
---

## Mathematical Reference

All results are derived formally in:
📄 **[Fixed Income Analytics (Palma, 2026)](docs/fixed_income_analytics.pdf)**
Key results implemented:
- Bond price generalization for T compoundings per year
- YTM via Newton-Raphson with analytical derivative
- Bootstrapping formula: $r(n) = \frac{1}{n}\ln\left(\frac{CF_n}{P - \sum CF_i Z_i}\right)$
- Nelson-Siegel-Svensson calibration via L-BFGS-B
- KRD additivity: $\sum_k KRD_k = D_{mod}$
- Floating leg = par (telescoping proof)
- Swap rate: $K^* = \frac{1 - Z(t_n)}{\sum \delta_i Z(t_i)}$
- PCA P&L variance: $\text{Var}(\Delta V) \approx \sum_k \lambda_k (\mathbf{v}_k^\top \mathbf{d})^2$
---

## Project Structure

```text
fixed-income-analytics/
│
├── src/                    # Source modules
│   ├── bonds.py            # VanillaBond
│   ├── curves.py           # DiscountCurve, Bootstrapper,
│   │                       # CurveInterpolator (NSS)
│   ├── sensitivity.py      # BondSensitivity,
│   │                       # PortfolioSensitivity
│   ├── pca.py              # YieldCurvePCA, PCARiskManager
│   └── swaps.py            # VanillaIRS, SwapPortfolio
│
├── notebooks/              # Interactive notebooks
│   ├── 01_bond_pricing.ipynb
│   ├── 02_yield_curve.ipynb
│   ├── 03_sensitivity.ipynb
│   ├── 04_pca.ipynb
│   └── 05_irs.ipynb
│
├── docs/                   # Mathematical reference
│   ├── fixed_income_analytics.pdf
│   └── fixed_income_analytics.tex
│
├── figures/                # Generated plots
├── data/                   # Market data (FRED)
├── tests/                  # Validation tests
├── requirements.txt
├── .env.example            # API key template
└── README.md
```
---

## Quickstart

### Installation

```bash
git clone https://github.com/tu-usuario/fixed-income-analytics
cd fixed-income-analytics
pip install -r requirements.txt
```

### FRED API Key (optional, for real data)

```bash
cp .env.example .env
# Edit .env and add your key:
# FRED_API_KEY=your_key_here
```

Get a free key at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html).

### Basic Usage

```python
import numpy as np
from src.bonds  import VanillaBond
from src.curves import DiscountCurve, Bootstrapper, CurveInterpolator
from src.sensitivity import BondSensitivity
from src.swaps  import VanillaIRS

# 1. Build yield curve
maturities = np.array([1., 2., 3., 5., 7., 10.])
yields     = np.array([0.04, 0.042, 0.045,
                       0.048, 0.050, 0.052])
curve = Bootstrapper.from_yields(maturities, yields)

# 2. Price a bond
bond = VanillaBond(coupon_rate=0.05,
                   face_value=1000,
                   maturity=10,
                   frequency=1)
metrics = bond.risk_metrics(ytm=0.04)
print(f"Price: {metrics['Price']:.2f}")
print(f"DV01:  {metrics['DV01']:.4f}")

# 3. Key Rate Durations
sens = BondSensitivity(bond, curve)
krd  = sens.key_rate_durations()
sm   = sens.summary()
print(f"Additivity check: {sm['additivity_ok']}")

# 4. Fit NSS curve
interp = CurveInterpolator(curve)
beta   = interp.fit_nss()
print(interp.nss_summary(beta))

# 5. Price an IRS
swap   = VanillaIRS(1_000_000, 0.0, 10, 2,
                    curve, "long")
K_star = swap.swap_rate()
swap   = VanillaIRS(1_000_000, K_star, 10, 2,
                    curve, "long")
print(f"Swap rate K* = {K_star:.4%}")
print(f"Float = Par: {swap.summary()['float_eq_par']}")
```

---

## Key Visualizations

| Figure | Description |
|--------|-------------|
| Price-yield curve with premium/discount regions | Bond parity |
| Yield curve: linear vs spline vs NSS | Interpolation comparison |
| NSS historical curves: Pre-COVID → Fed hikes | Curve evolution |
| KRD profiles by bond type | Risk concentration |
| Eigenvectors vs maturities | Level, slope, curvature |
| PCA variance explained | Marginal gain per component |
| PCA vs NSS loading vectors | Statistical vs parametric |
| Swap curve vs spot curve | Averaging effect of K* |
| Curve trade P&L heatmap | Flattener/steepener scenarios |

All notebooks include **interactive widgets** (ipywidgets) 
for real-time exploration of parameters.

---

## Data Sources

| Data | Source | Access |
|------|--------|--------|
| U.S. Treasury yields | Federal Reserve (FRED) | `fredapi` |
| Historical yield curves | FRED: DGS1, DGS2, ..., DGS30 | Free API key |

All notebooks include a synthetic data fallback 
if no API key is provided.

---

## Requirements

- numpy>=1.24.0
- scipy>=1.10.0
- pandas>=2.0.0
- matplotlib>=3.7.0
- plotly>=5.14.0
- ipywidgets>=8.0.0
- jupyter>=1.0.0
- scikit-learn>=1.3.0
- fredapi>=0.5.0
- python-dotenv>=1.0.0

---

## How to Use This Library
The modules are designed to be used in sequence, 
but each class is independently importable.
A typical workflow follows the mathematical chain:
1. DiscountCurve / Bootstrapper
2. CurveInterpolator (NSS)
3. BondSensitivity / PortfolioSensitivity
4. YieldCurvePCA / PCARiskManager
5. VanillaIRS / SwapPortfolio
See the [Quickstart](#quickstart) above for 
a minimal working example, and each notebook for 
a full walkthrough with real market data.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Renato Antonio Palma Valencia**
Economist | MSc Economics & Quantitative Methods  
Universidad Adolfo Ibáñez, Chile

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/renato-antonio-p-a488191a7)
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black)](https://github.com/renatopalmavalencia)

---

## Acknowledgements

- Nelson & Siegel (1987) and Svensson (1994) for the parametric curve model
- Litterman & Scheinkman (1991) for the empirical three-factor result
- Ho (1992) for the Key Rate Duration framework
- Federal Reserve Bank of St. Louis (FRED) for open market data
