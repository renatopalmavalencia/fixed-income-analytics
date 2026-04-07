"""
sensitivity.py
==============
Interest rate sensitivity analysis: DV01, Key Rate
Durations and portfolio aggregation.

Mathematical reference:
    Palma, R. (2026). Fixed Income Analytics.
    Module 3: Sensitivity Analysis.
"""

import numpy as np
import pandas as pd
from src.bonds import VanillaBond
from src.curves import DiscountCurve, CurveInterpolator


# ------------------------- #
# CLASS 1: BOND SENSITIVITY #
#-------------------------- #

class BondSensitivity:
    """
    Computes DV01 and Key Rate Durations for a single bond
    under a full yield curve.

    Mathematical reference:
        DV01_parallel = D_mod * P * 0.0001
        KRD_k = [P(r(t_k)-eps) - P(r(t_k)+eps)] / (2*eps*P)
        sum(KRD_k) = D_mod  (additivity property)

    Parameters
    ----------
    bond  : VanillaBond    Bond to analyze
    curve : DiscountCurve  Current market curve
    """

    def __init__(self, bond: VanillaBond,
                 curve: DiscountCurve):
        self.bond  = bond
        self.curve = curve
        self._P    = self._price()

    def __repr__(self) -> str:
        return (f"BondSensitivity("
                f"bond={self.bond}, "
                f"price={self._P:.4f})")

    # ---- 1. PRICE UNDER FULL CURVE ---- #

    def _price(self, curve: DiscountCurve = None) -> float:
        """
        Price the bond under a given curve.
        Uses the curve stored in self if none provided.

        Formula:
            P = sum CF_i * Z(t_i) = sum CF_i * e^(-r(t_i)*t_i)

        Parameters
        ----------
        curve : DiscountCurve  Optional override curve

        Returns
        -------
        float : Bond price
        """
        if curve is None:
            curve = self.curve

        cf      = self.bond.cash_flows()
        periods = np.arange(1, self.bond.n + 1) / self.bond.freq

        pv = sum(
            cf[i] * np.exp(-curve.spot_rate(t) * t)
            for i, t in enumerate(periods)
        )
        return pv

    # ---- 2. PARALLEL SHIFT DV01 ---- #

    def dv01_parallel(self, eps: float = 0.0001) -> float:
        """
        DV01 under a parallel shift of the entire curve.

        Computed via central finite difference:
            DV01 = [P(r - eps) - P(r + eps)] / 2

        Parameters
        ----------
        eps : float  Shift size (default 1bp = 0.0001)

        Returns
        -------
        float : DV01 in currency units
        """
        curve_up   = DiscountCurve(
            self.curve.maturities,
            self.curve.spot_rates + eps
        )
        curve_down = DiscountCurve(
            self.curve.maturities,
            self.curve.spot_rates - eps
        )

        P_up   = self._price(curve_up)
        P_down = self._price(curve_down)

        return (P_down - P_up) / 2

    def modified_duration_curve(self) -> float:
        """
        Modified Duration implied by the full curve.

        Formula:
            D_mod = DV01_parallel / (P * 0.0001)

        Returns
        -------
        float : Modified Duration in years
        """
        return self.dv01_parallel() / (self._P * 0.0001)

    # ---- 3. KEY RATE DURATIONS ---- #

    def key_rate_durations(self,
                           eps: float = 0.0001) -> dict:
        """
        Key Rate Duration at each node of the curve.

        Formula:
            KRD_k = [P(r(t_k)-eps) - P(r(t_k)+eps)]
                    / (2 * eps * P)

        Only the rate at node t_k is shifted;
        all other rates remain fixed.

        Validation:
            sum(KRD_k) == D_mod  (additivity)

        Parameters
        ----------
        eps : float  Shift size (default 1bp)

        Returns
        -------
        dict : {maturity: KRD_k} for each node
        """
        krd    = {}
        nodes  = self.curve.maturities

        for k, t_k in enumerate(nodes):
            # Shift only node k
            rates_up   = self.curve.spot_rates.copy()
            rates_down = self.curve.spot_rates.copy()
            rates_up[k]   += eps
            rates_down[k] -= eps

            curve_up   = DiscountCurve(nodes, rates_up)
            curve_down = DiscountCurve(nodes, rates_down)

            P_up   = self._price(curve_up)
            P_down = self._price(curve_down)

            krd[float(t_k)] = (P_down - P_up) / (
                2 * eps * self._P)

        return krd

    def dv01_by_key_rate(self,
                          eps: float = 0.0001) -> dict:
        """
        DV01 at each key rate node.

        Formula:
            DV01_k = KRD_k * P * 0.0001

        Total DV01 = sum(DV01_k) = DV01_parallel

        Parameters
        ----------
        eps : float  Shift size

        Returns
        -------
        dict : {maturity: DV01_k}
        """
        krds = self.key_rate_durations(eps)
        return {
            t: krd * self._P * 0.0001
            for t, krd in krds.items()
        }

    # ---- 4. SUMMARY ---- #

    def summary(self, eps: float = 0.0001) -> dict:
        """
        Full sensitivity summary for the bond.

        Returns
        -------
        dict with:
            price, dv01_parallel, mod_duration,
            krd (dict), dv01_by_krd (dict),
            additivity_check (bool)
        """
        krds       = self.key_rate_durations(eps)
        dv01_krds  = self.dv01_by_key_rate(eps)
        dv01_par   = self.dv01_parallel(eps)
        d_mod      = self.modified_duration_curve()
        krd_sum    = sum(krds.values())
        additive   = np.isclose(krd_sum, d_mod, atol=1e-4)

        return {
            "price":           round(self._P, 6),
            "dv01_parallel":   round(dv01_par, 6),
            "mod_duration":    round(d_mod, 6),
            "krd_sum":         round(krd_sum, 6),
            "additivity_ok":   additive,
            "krd":             {
                round(t, 2): round(v, 6)
                for t, v in krds.items()
            },
            "dv01_by_krd":     {
                round(t, 2): round(v, 6)
                for t, v in dv01_krds.items()
            }
        }


# ------------------------------ #
# CLASS 2: PORTFOLIO SENSITIVITY #
# ------------------------------ #

class PortfolioSensitivity:
    """
    Aggregates DV01 and KRD vectors across a portfolio
    of bonds.

    Mathematical reference:
        KRD_k^portfolio = sum(w_j * P_j * KRD_k^j) / V
        DV01_k^portfolio = sum(w_j * DV01_k^j)

    Parameters
    ----------
    bonds    : list of VanillaBond
    curve    : DiscountCurve
    weights  : list of float  Number of units held per bond
                              (default: 1 unit each)
    labels   : list of str    Bond labels for reporting
    """

    def __init__(self, bonds: list,
                 curve: DiscountCurve,
                 weights: list = None,
                 labels: list  = None):

        self.bonds   = bonds
        self.curve   = curve
        self.weights = weights if weights else [1.0] * len(bonds)
        self.labels  = labels  if labels  else [
            f"Bond {i+1}" for i in range(len(bonds))
        ]

        assert len(self.bonds) == len(self.weights), \
            "bonds and weights must have the same length"

        # Pre-compute individual sensitivities
        self._sensitivities = [
            BondSensitivity(b, curve) for b in bonds
        ]

    def __repr__(self) -> str:
        return (f"PortfolioSensitivity("
                f"{len(self.bonds)} bonds, "
                f"curve nodes={len(self.curve.maturities)})")

    # ---- 1. PORTFOLIO VALUE ---- #

    def portfolio_value(self) -> float:
        """
        Total portfolio value.

        Formula:
            V = sum w_j * P_j

        Returns
        -------
        float : Portfolio value V
        """
        return sum(
            w * s._P
            for w, s in zip(self.weights,
                            self._sensitivities)
        )

    # ---- 2. PORTFOLIO DV01 ---- #

    def dv01_parallel(self) -> float:
        """
        Portfolio DV01 under parallel shift.

        Formula:
            DV01^portfolio = sum w_j * DV01_j

        Returns
        -------
        float : Total portfolio DV01
        """
        return sum(
            w * s.dv01_parallel()
            for w, s in zip(self.weights,
                            self._sensitivities)
        )

    # ---- 3. PORTFOLIO KRD ---- #

    def key_rate_durations(self) -> dict:
        """
        Portfolio KRD vector.

        Formula:
            KRD_k^p = sum(w_j * P_j * KRD_k^j) / V

        Returns
        -------
        dict : {maturity: portfolio KRD_k}
        """
        V    = self.portfolio_value()
        nodes = self.curve.maturities
        krd_portfolio = {float(t): 0.0 for t in nodes}

        for w, s in zip(self.weights,
                        self._sensitivities):
            krds = s.key_rate_durations()
            for t, krd_k in krds.items():
                krd_portfolio[t] += w * s._P * krd_k / V

        return krd_portfolio

    def dv01_by_key_rate(self) -> dict:
        """
        Portfolio DV01 at each key rate node.

        Formula:
            DV01_k^portfolio = sum w_j * DV01_k^j

        Returns
        -------
        dict : {maturity: portfolio DV01_k}
        """
        nodes        = self.curve.maturities
        dv01_portfolio = {float(t): 0.0 for t in nodes}

        for w, s in zip(self.weights,
                        self._sensitivities):
            dv01s = s.dv01_by_key_rate()
            for t, dv01_k in dv01s.items():
                dv01_portfolio[t] += w * dv01_k

        return dv01_portfolio

    # ---- 4. SCENARIO ANALYSIS ---- #

    def scenario_pnl(self,
                     curve_shift: np.ndarray) -> float:
        """
        Approximate P&L under a non-parallel curve shift.

        Formula:
            dV ≈ -DV01_vector · curve_shift / 0.0001

        where curve_shift is in rate units (e.g. 0.01 = 100bps)

        Parameters
        ----------
        curve_shift : np.ndarray  Rate shift at each node
                                  (same length as curve nodes)

        Returns
        -------
        float : Approximate P&L in currency units
        """
        assert len(curve_shift) == len(self.curve.maturities), \
            "curve_shift must match number of curve nodes"

        dv01s  = self.dv01_by_key_rate()
        dv01_v = np.array(list(dv01s.values()))

        # dV ≈ -DV01_k * (shift_k / 0.0001) per node
        return -np.dot(dv01_v, curve_shift / 0.0001)

    # ---- 5. SUMMARY TABLE ---- #

    def summary_table(self) -> pd.DataFrame:
        """
        Full sensitivity table: one row per bond
        plus a portfolio total row.

        Returns
        -------
        pd.DataFrame with price, DV01, KRD per node
        """
        nodes  = self.curve.maturities
        rows   = []

        for label, w, s in zip(self.labels,
                                self.weights,
                                self._sensitivities):
            krds  = s.key_rate_durations()
            row   = {
                "Bond":          label,
                "Units":         w,
                "Price":         round(s._P, 4),
                "MV":            round(w * s._P, 2),
                "DV01 (parallel)": round(
                    w * s.dv01_parallel(), 4),
            }
            for t in nodes:
                row[f"KRD {t}y"] = round(
                    krds.get(float(t), 0), 4)
            rows.append(row)

        # Portfolio total row
        total = {
            "Bond":  "PORTFOLIO",
            "Units": "",
            "Price": "",
            "MV":    round(self.portfolio_value(), 2),
            "DV01 (parallel)": round(
                self.dv01_parallel(), 4),
        }
        port_krds = self.key_rate_durations()
        for t in nodes:
            total[f"KRD {t}y"] = round(
                port_krds.get(float(t), 0), 4)
        rows.append(total)

        return pd.DataFrame(rows).set_index("Bond")

    # ---- 6. HEDGE RATIOS ---- #

    def hedge_ratios(self,
                     hedges: list,
                     hedge_labels: list = None) -> dict:
        """
        Compute hedge ratios to neutralize portfolio DV01
        using a set of hedging instruments.

        Solves: sum h_j * DV01_k^j = -DV01_k^portfolio
        via least squares if overdetermined.

        Parameters
        ----------
        hedges       : list of VanillaBond  Hedging instruments
        hedge_labels : list of str          Labels

        Returns
        -------
        dict : {label: hedge_ratio}
        """
        if hedge_labels is None:
            hedge_labels = [
                f"Hedge {i+1}" for i in range(len(hedges))
            ]

        nodes     = self.curve.maturities
        n_nodes   = len(nodes)
        n_hedges  = len(hedges)

        # Build DV01 matrix: rows=nodes, cols=hedges
        D = np.zeros((n_nodes, n_hedges))
        for j, hedge in enumerate(hedges):
            hs    = BondSensitivity(hedge, self.curve)
            dv01s = hs.dv01_by_key_rate()
            for k, t in enumerate(nodes):
                D[k, j] = dv01s.get(float(t), 0)

        # Target: negative of portfolio DV01 vector
        port_dv01 = np.array(
            list(self.dv01_by_key_rate().values()))
        target = -port_dv01

        # Solve via least squares
        h, _, _, _ = np.linalg.lstsq(D, target,
                                      rcond=None)

        return {label: round(float(ratio), 6)
                for label, ratio in zip(hedge_labels, h)}