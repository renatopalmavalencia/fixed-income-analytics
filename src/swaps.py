"""
swaps.py
========
Interest Rate Swap pricing, valuation and risk metrics.

Mathematical reference:
    Palma, R. (2026). Fixed Income Analytics.
    Module 5: IRS Pricing.

Key results:
    - Floating leg = par (telescoping proof)
    - Swap as portfolio of FRAs
    - Swap rate K* from discount factors
    - Swap value at t > 0
    - Swap DV01 and key rate sensitivity
"""

import numpy as np
import pandas as pd
from src.curves import DiscountCurve, CurveInterpolator


# -------------------- #
# CLASS 1: VANILLA IRS #
# -------------------- #

class VanillaIRS:
    """
    Plain vanilla Interest Rate Swap.

    One party pays fixed rate K and receives floating
    (SOFR/EURIBOR) on notional N at dates t_1,...,t_n.

    Mathematical reference:
        V_swap = V_float - V_fixed
        V_float = N  (always, at reset dates)
        K* = (1 - Z(t_n)) / sum(delta_i * Z(t_i))

    Parameters
    ----------
    notional    : float          Notional amount N
    fixed_rate  : float          Fixed rate K (annual)
    tenor       : float          Swap tenor in years
    frequency   : int            Payment frequency per year
                                 (1=annual, 2=semi, 4=quarterly)
    curve       : DiscountCurve  Discount curve at inception
    position    : str            'long' (pay fixed, recv float)
                                 'short' (recv fixed, pay float)
    """

    def __init__(self, notional: float,
                 fixed_rate: float,
                 tenor: float,
                 frequency: int,
                 curve: DiscountCurve,
                 position: str = "long"):

        assert notional > 0, \
            "Notional must be positive"
        assert fixed_rate >= 0, \
            "Fixed rate must be non-negative"
        assert tenor > 0, \
            "Tenor must be positive"
        assert frequency in [1, 2, 4, 12], \
            "Frequency must be 1, 2, 4 or 12"
        assert position in ["long", "short"], \
            "Position must be 'long' or 'short'"

        self.N         = notional
        self.K         = fixed_rate
        self.tenor     = tenor
        self.freq      = frequency
        self.curve     = curve
        self.position  = position
        self.sign      = 1.0 if position == "long" else -1.0

        # Payment schedule
        self.delta      = 1.0 / frequency
        self.n          = int(tenor * frequency)
        self.pay_dates  = np.array([
            (i + 1) * self.delta
            for i in range(self.n)
        ])

        # Discount factors at payment dates
        self.Z = np.array([
            curve.discount_factor(t)
            for t in self.pay_dates
        ])

    def __repr__(self) -> str:
        return (f"VanillaIRS("
                f"N={self.N:,.0f}, "
                f"K={self.K:.4%}, "
                f"tenor={self.tenor}y, "
                f"freq={self.freq}, "
                f"position={self.position})")

    # ---- 1. SWAP RATE ---- #

    def swap_rate(self) -> float:
        """
        Fair swap rate K* that makes V_swap = 0 at inception.

        Formula:
            K* = (1 - Z(t_n)) / sum(delta_i * Z(t_i))

        Numerator:   decay of discount factor to maturity
        Denominator: present value of annuity of delta_i

        Returns
        -------
        float : Swap rate K*
        """
        numerator   = 1.0 - self.Z[-1]
        denominator = np.sum(self.delta * self.Z)

        assert denominator > 0, \
            "Annuity factor must be positive"

        return numerator / denominator

    # ---- 2. LEG VALUES ---- #

    def fixed_leg_value(self,
                        curve: DiscountCurve = None
                        ) -> float:
        """
        Present value of the fixed leg.

        Formula:
            V_fixed = N * K * sum(delta_i * Z(t_i))
                    + N * Z(t_n)

        Parameters
        ----------
        curve : DiscountCurve  Optional override curve

        Returns
        -------
        float : PV of fixed leg
        """
        Z = self._get_Z(curve)
        coupon_pv  = self.N * self.K * np.sum(
            self.delta * Z)
        principal  = self.N * Z[-1]
        return coupon_pv + principal

    def floating_leg_value(self,
                           curve: DiscountCurve = None
                           ) -> float:
        """
        Present value of the floating leg.

        Proposition (Floating = Par):
            V_float = N * sum(Z(t_{i-1}) - Z(t_i))
                    + N * Z(t_n)
                    = N * Z(0) = N

        The telescoping sum always collapses to par.

        Parameters
        ----------
        curve : DiscountCurve  Optional override curve

        Returns
        -------
        float : PV of floating leg (always = N at reset)
        """
        Z       = self._get_Z(curve)
        Z_prev  = np.concatenate([[1.0], Z[:-1]])

        # Telescoping sum
        telescoping = np.sum(Z_prev - Z)
        return self.N * (telescoping + Z[-1])

    def _get_Z(self,
               curve: DiscountCurve = None
               ) -> np.ndarray:
        """
        Discount factors under given or stored curve.

        Parameters
        ----------
        curve : DiscountCurve  Optional override

        Returns
        -------
        np.ndarray : Z(t_i) for all payment dates
        """
        if curve is None:
            return self.Z
        return np.array([
            curve.discount_factor(t)
            for t in self.pay_dates
        ])

    # ---- 3. SWAP VALUE ---- #

    def value(self,
              curve: DiscountCurve = None) -> float:
        """
        Mark-to-market value of the swap.

        Formula (long position, pay fixed):
            V = V_float - V_fixed
              = N * [1 - Z(t_n) - K * sum(delta_i * Z(t_i))]

        Equivalently:
            V = N * (K*_t - K) * sum(delta_i * Z(t_i))

        Parameters
        ----------
        curve : DiscountCurve  Current market curve
                               (may differ from inception)

        Returns
        -------
        float : Swap MtM value (positive = asset)
        """
        V = (self.floating_leg_value(curve) -
             self.fixed_leg_value(curve))
        return self.sign * V

    def inception_value(self) -> float:
        """
        Swap value at inception with K = K*.

        Must equal zero by definition of K*.

        Returns
        -------
        float : Should be ~0.0
        """
        swap_at_par = VanillaIRS(
            self.N, self.swap_rate(),
            self.tenor, self.freq,
            self.curve, self.position
        )
        return swap_at_par.value()

    # ---- 4. FRA DECOMPOSITION ---- #

    def fra_values(self) -> pd.DataFrame:
        """
        Decompose swap into individual FRAs.

        Formula per FRA:
            V_FRA(t_i) = N * delta_i *
                         (f(t_{i-1}, t_i) - K) * Z(t_i)

        Returns
        -------
        pd.DataFrame : FRA values with forward rates
        """
        rows = []
        Z    = self.Z
        Z_prev = np.concatenate([[1.0], Z[:-1]])

        for i, (t, z, z_prev) in enumerate(
                zip(self.pay_dates, Z, Z_prev)):

            t_prev = self.pay_dates[i-1] if i > 0 else 0.0

            # Forward rate from discount factors
            # f(t_{i-1}, t_i) = [ln Z(t_{i-1}) - ln Z(t_i)]
            #                    / delta_i
            f_i = (z_prev / z - 1.0) / self.delta

            # FRA value
            v_fra = (self.N * self.delta *
                     (f_i - self.K) * z)

            rows.append({
                "Period":       i + 1,
                "t_start":      round(t_prev, 4),
                "t_end":        round(t, 4),
                "Z(t_i)":       round(z, 6),
                "Forward Rate": round(f_i, 6),
                "Fixed Rate K": round(self.K, 6),
                "Spread":       round(f_i - self.K, 6),
                "FRA Value":    round(self.sign * v_fra, 4)
            })

        return pd.DataFrame(rows)

    # ---- 5. DV01 ---- #

    def dv01(self, eps: float = 0.0001) -> float:
        """
        Swap DV01 under parallel shift.

        Formula:
            DV01 = [V(r - eps) - V(r + eps)] / 2

        Approximately equals DV01 of fixed leg since
        floating leg has near-zero duration.

        Parameters
        ----------
        eps : float  Shift size (default 1bp)

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

        return (self.value(curve_down) -
                self.value(curve_up)) / 2

    def key_rate_dv01(self,
                      eps: float = 0.0001) -> dict:
        """
        Swap DV01 at each key rate node.

        Formula:
            DV01_k = [V(r_k - eps) - V(r_k + eps)] / 2
            (only node k shifted, others fixed)

        Returns
        -------
        dict : {maturity: DV01_k}
        """
        nodes  = self.curve.maturities
        result = {}

        for k, t_k in enumerate(nodes):
            rates_up   = self.curve.spot_rates.copy()
            rates_down = self.curve.spot_rates.copy()
            rates_up[k]   += eps
            rates_down[k] -= eps

            c_up   = DiscountCurve(nodes, rates_up)
            c_down = DiscountCurve(nodes, rates_down)

            dv01_k = (self.value(c_down) -
                      self.value(c_up)) / 2
            result[float(t_k)] = dv01_k

        return result

    def modified_duration(self) -> float:
        """
        Modified duration of the swap.

        Formula:
            D_mod = DV01 / (V_fixed * 0.0001)

        Returns
        -------
        float : Modified duration in years
        """
        V_fixed = self.fixed_leg_value()
        if np.isclose(V_fixed, 0):
            return 0.0
        return self.dv01() / (V_fixed * 0.0001)

    # ---- 6. SENSITIVITY TO RATE CHANGES ---- #

    def value_vs_rate(self,
                      rate_range: np.ndarray
                      ) -> pd.DataFrame:
        """
        Swap value as a function of parallel rate shift.

        Useful for visualizing P&L profile.

        Parameters
        ----------
        rate_range : np.ndarray  Rate shifts to evaluate
                                 (e.g. np.linspace(-0.02, 0.02))

        Returns
        -------
        pd.DataFrame : rate_shift, swap_value, fixed_leg,
                       float_leg
        """
        rows = []
        for dr in rate_range:
            c_shifted = DiscountCurve(
                self.curve.maturities,
                self.curve.spot_rates + dr
            )
            rows.append({
                "Rate Shift":  round(dr, 6),
                "Swap Value":  round(self.value(c_shifted), 4),
                "Fixed Leg":   round(
                    self.fixed_leg_value(c_shifted), 4),
                "Float Leg":   round(
                    self.floating_leg_value(c_shifted), 4),
            })

        return pd.DataFrame(rows)

    # ---- 7. SUMMARY ---- #

    def summary(self) -> dict:
        """
        Full swap summary at current market curve.

        Returns
        -------
        dict with swap rate, leg values, DV01,
        duration and FRA decomposition
        """
        K_star  = self.swap_rate()
        V_fixed = self.fixed_leg_value()
        V_float = self.floating_leg_value()
        V_swap  = self.value()
        dv01    = self.dv01()
        d_mod   = self.modified_duration()

        return {
            "notional":       self.N,
            "fixed_rate_K":   round(self.K, 6),
            "swap_rate_K*":   round(K_star, 6),
            "spread_K_Kstar": round(self.K - K_star, 6),
            "fixed_leg":      round(V_fixed, 4),
            "float_leg":      round(V_float, 4),
            "float_eq_par":   np.isclose(
                V_float, self.N, rtol=1e-4),
            "swap_value":     round(V_swap, 4),
            "dv01":           round(dv01, 6),
            "mod_duration":   round(d_mod, 6),
            "position":       self.position,
        }


# ----------------------- #
# CLASS 2: SWAP PORTFOLIO #
# ----------------------- #

class SwapPortfolio:
    """
    Portfolio of vanilla IRS contracts.

    Aggregates values, DV01s and key rate sensitivities
    across multiple swaps.

    Parameters
    ----------
    swaps  : list of VanillaIRS
    labels : list of str         Labels for reporting
    """

    def __init__(self, swaps: list,
                 labels: list = None):

        self.swaps  = swaps
        self.labels = labels or [
            f"Swap {i+1}" for i in range(len(swaps))
        ]
        assert len(swaps) == len(self.labels), \
            "swaps and labels must match in length"

    def __repr__(self) -> str:
        return (f"SwapPortfolio("
                f"{len(self.swaps)} swaps, "
                f"total_value="
                f"{self.total_value():,.2f})")

    # ---- 1. PORTFOLIO VALUE ---- #

    def total_value(self,
                    curve: DiscountCurve = None
                    ) -> float:
        """
        Total portfolio MtM value.

        Parameters
        ----------
        curve : DiscountCurve  Optional override

        Returns
        -------
        float : Sum of swap values
        """
        return sum(s.value(curve) for s in self.swaps)

    # ---- 2. PORTFOLIO DV01 ---- #

    def total_dv01(self) -> float:
        """
        Portfolio DV01 under parallel shift.

        Returns
        -------
        float
        """
        return sum(s.dv01() for s in self.swaps)

    def key_rate_dv01(self) -> dict:
        """
        Aggregated KRD DV01 vector.

        Returns
        -------
        dict : {maturity: total DV01_k}
        """
        # Use maturities from first swap's curve
        nodes  = self.swaps[0].curve.maturities
        result = {float(t): 0.0 for t in nodes}

        for swap in self.swaps:
            krd = swap.key_rate_dv01()
            for t, dv01_k in krd.items():
                if t in result:
                    result[t] += dv01_k

        return result

    # ---- 3. SCENARIO PNL ---- #

    def scenario_pnl(self,
                     curve_shift: np.ndarray
                     ) -> dict:
        """
        Approximate P&L under a non-parallel curve shift.

        Formula:
            dV ≈ -DV01_vec . (curve_shift / 0.0001)

        Parameters
        ----------
        curve_shift : np.ndarray  Rate shift at each node

        Returns
        -------
        dict : P&L per swap and total
        """
        result = {}
        total  = 0.0

        for label, swap in zip(self.labels, self.swaps):
            krd   = swap.key_rate_dv01()
            dv01v = np.array(list(krd.values()))
            pnl   = -np.dot(dv01v,
                            curve_shift / 0.0001)
            result[label] = round(pnl, 4)
            total += pnl

        result["Total"] = round(total, 4)
        return result

    # ---- 4. SUMMARY TABLE ---- #

    def summary_table(self) -> pd.DataFrame:
        """
        Summary table: one row per swap plus total.

        Returns
        -------
        pd.DataFrame
        """
        rows = []
        for label, swap in zip(self.labels, self.swaps):
            sm = swap.summary()
            rows.append({
                "Swap":        label,
                "Notional":    f"{sm['notional']:,.0f}",
                "K":           f"{sm['fixed_rate_K']:.4%}",
                "K*":          f"{sm['swap_rate_K*']:.4%}",
                "Spread":      f"{sm['spread_K_Kstar']:.4%}",
                "Value":       f"{sm['swap_value']:,.2f}",
                "DV01":        f"{sm['dv01']:,.4f}",
                "D_mod":       f"{sm['mod_duration']:.4f}",
                "Position":    sm["position"],
                "Float=Par":   "✓" if sm["float_eq_par"]
                               else "✗"
            })

        # Total row
        rows.append({
            "Swap":     "TOTAL",
            "Notional": "",
            "K":        "",
            "K*":       "",
            "Spread":   "",
            "Value":    f"{self.total_value():,.2f}",
            "DV01":     f"{self.total_dv01():,.4f}",
            "D_mod":    "",
            "Position": "",
            "Float=Par": ""
        })

        return pd.DataFrame(rows).set_index("Swap")

    # ---- 5. CURVE TRADE ---- #

    @staticmethod
    def curve_trade(notional: float,
                    tenor_short: float,
                    tenor_long: float,
                    frequency: int,
                    curve: DiscountCurve) -> 'SwapPortfolio':
        """
        Classic yield curve trade:
            - Receive fixed in long-dated swap (long)
            - Pay fixed in short-dated swap (short)

        Profits from curve flattening (long rates fall
        relative to short rates) regardless of parallel
        level shifts.

        Parameters
        ----------
        notional    : float          Notional amount
        tenor_short : float          Short swap tenor
        tenor_long  : float          Long swap tenor
        frequency   : int            Payment frequency
        curve       : DiscountCurve  Current curve

        Returns
        -------
        SwapPortfolio : Two-legged curve trade
        """
        # Short swap: pay fixed (standard long position)
        swap_short = VanillaIRS(
            notional, 0.0,   # K set to K* below
            tenor_short, frequency, curve, "long"
        )
        K_short = swap_short.swap_rate()
        swap_short = VanillaIRS(
            notional, K_short,
            tenor_short, frequency, curve, "long"
        )

        # Long swap: receive fixed (short position)
        swap_long = VanillaIRS(
            notional, 0.0,
            tenor_long, frequency, curve, "short"
        )
        K_long = swap_long.swap_rate()
        swap_long = VanillaIRS(
            notional, K_long,
            tenor_long, frequency, curve, "short"
        )

        return SwapPortfolio(
            [swap_short, swap_long],
            [f"Pay Fixed {tenor_short}y (K={K_short:.4%})",
             f"Recv Fixed {tenor_long}y (K={K_long:.4%})"]
        )