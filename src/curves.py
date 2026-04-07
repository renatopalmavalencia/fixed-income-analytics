"""
curves.py
=========
Yield curve construction, interpolation and parametric models.

Mathematical reference:
    Palma, R. (2026). Fixed Income Analytics.
    Module 2: Yield Curve.
"""

import numpy as np
from scipy import optimize, interpolate


# ----------------------- #
# CLASS 1: DISCOUNT CURVE #
# ----------------------- #

class DiscountCurve:
    """
    Represents a yield curve as a set of (maturity, spot rate)
    pairs extracted via bootstrapping.

    Attributes
    ----------
    maturities : np.ndarray  Observed maturities in years
    spot_rates : np.ndarray  Continuously compounded spot rates
    """

    def __init__(self, maturities: np.ndarray,
                 spot_rates: np.ndarray):
        """
        Parameters
        ----------
        maturities : array-like  Maturities in years
        spot_rates : array-like  Continuously compounded spot rates
        """
        self.maturities = np.asarray(maturities, dtype=float)
        self.spot_rates = np.asarray(spot_rates, dtype=float)

        assert len(self.maturities) == len(self.spot_rates), \
            "maturities and spot_rates must have the same length"
        assert np.all(np.diff(self.maturities) > 0), \
            "maturities must be strictly increasing"
        assert np.all(self.spot_rates > 0), \
            "spot rates must be positive"

    def __repr__(self) -> str:
        return (f"DiscountCurve("
                f"maturities={self.maturities}, "
                f"spot_rates={np.round(self.spot_rates, 4)})")

    # ---- 1. DISCOUNT FACTORS ---- #

    def discount_factor(self, t: float) -> float:
        """
        Continuous discount factor at maturity t.

        Formula:
            Z(t) = e^(-r(t) * t)

        Parameters
        ----------
        t : float  Maturity in years

        Returns
        -------
        float : Discount factor Z(t)
        """
        r = self.spot_rate(t)
        return np.exp(-r * t)

    def discount_factors(self) -> np.ndarray:
        """
        Discount factors at all observed maturities.

        Returns
        -------
        np.ndarray : Z(t_i) for all observed maturities
        """
        return np.exp(-self.spot_rates * self.maturities)

    # ---- 2. SPOT RATE (from interpolation) ---- #

    def spot_rate(self, t: float) -> float:
        """
        Spot rate at arbitrary maturity t via linear
        interpolation on zero rates.

        Parameters
        ----------
        t : float  Maturity in years

        Returns
        -------
        float : Spot rate r(t)
        """
        return float(np.interp(t, self.maturities,
                               self.spot_rates))

    # ---- 3. FORWARD RATES ---- #

    def forward_rate(self, t1: float, t2: float) -> float:
        """
        Forward rate between t1 and t2.

        Formula:
            f(t1, t2) = [r(t2)*t2 - r(t1)*t1] / (t2 - t1)

        Equivalently in terms of discount factors:
            f(t1, t2) = [ln Z(t1) - ln Z(t2)] / (t2 - t1)

        Parameters
        ----------
        t1 : float  Start of forward period
        t2 : float  End of forward period

        Returns
        -------
        float : Forward rate f(t1, t2)
        """
        assert t2 > t1 > 0, "Must have 0 < t1 < t2"
        r1 = self.spot_rate(t1)
        r2 = self.spot_rate(t2)
        return (r2 * t2 - r1 * t1) / (t2 - t1)

    def instantaneous_forward(self, t: float,
                               dt: float = 1e-5) -> float:
        """
        Instantaneous forward rate at maturity t.

        Formula:
            f(t) = r(t) + t * dr(t)/dt
                 = d[r(t)*t] / dt

        Computed numerically as the limit of f(t, t+dt).

        Parameters
        ----------
        t  : float  Maturity in years
        dt : float  Step size for numerical derivative

        Returns
        -------
        float : Instantaneous forward rate f(t)
        """
        t = max(t, dt)
        r_t    = self.spot_rate(t)
        r_tdt  = self.spot_rate(t + dt)
        # d[r(t)*t]/dt via finite difference
        return ((r_tdt * (t + dt)) - (r_t * t)) / dt

    def forward_curve(self, t_grid: np.ndarray) -> np.ndarray:
        """
        Instantaneous forward curve over a grid of maturities.

        Parameters
        ----------
        t_grid : np.ndarray  Grid of maturities

        Returns
        -------
        np.ndarray : f(t) for each t in t_grid
        """
        return np.array([self.instantaneous_forward(t)
                         for t in t_grid])


# --------------------- #
# CLASS 2: BOOTSTRAPPER #
# --------------------- #

class Bootstrapper:
    """
    Extracts spot rates from observed bond prices
    via sequential bootstrapping.

    Mathematical reference:
        r(n) = (1/n) * ln(CF_n / (P - sum CF_i * Z(i)))
    """

    @staticmethod
    def bootstrap(prices: np.ndarray,
                  cash_flows: list,
                  maturities: np.ndarray) -> DiscountCurve:
        """
        Bootstrap spot rates from observed bond prices.

        Parameters
        ----------
        prices     : np.ndarray  Observed bond prices
        cash_flows : list        List of CF arrays, one per bond
                                 Each array has length = maturity
        maturities : np.ndarray  Bond maturities in years
                                 (must be integers for now)

        Returns
        -------
        DiscountCurve with bootstrapped spot rates
        """
        n_bonds   = len(prices)
        spot_rates = np.zeros(n_bonds)
        Z          = np.zeros(n_bonds)   # discount factors

        for i in range(n_bonds):
            P   = prices[i]
            cf  = cash_flows[i]
            n   = maturities[i]

            # Sum of already-known PV of intermediate flows
            pv_known = sum(cf[j] * Z[j]
                           for j in range(i)
                           if Z[j] > 0)

            # Last cash flow (coupon + principal)
            cf_n = cf[i]

            # Bootstrapping formula
            # r(n) = (1/n) * ln(CF_n / (P - pv_known))
            denom = P - pv_known
            assert denom > 0, \
                f"Negative denominator at maturity {n}. " \
                f"Check bond {i} price or cash flows."

            r_n = (1 / n) * np.log(cf_n / denom)
            spot_rates[i] = r_n
            Z[i]          = np.exp(-r_n * n)

        return DiscountCurve(maturities, spot_rates)

    @staticmethod
    def from_yields(maturities: np.ndarray,
                    yields: np.ndarray) -> 'DiscountCurve':
        """
        Build a DiscountCurve directly from observed yields
        (e.g. from FRED Treasury data).

        Converts from annually compounded yields to
        continuously compounded rates:
            r_c = ln(1 + r_d)

        Parameters
        ----------
        maturities : np.ndarray  Maturities in years
        yields     : np.ndarray  Annually compounded yields

        Returns
        -------
        DiscountCurve
        """
        r_continuous = np.log(1 + yields)
        return DiscountCurve(maturities, r_continuous)


# --------------------------- #
# CLASS 3: CURVE INTERPOLATOR #
# --------------------------- #

class CurveInterpolator:
    """
    Three interpolation methods for yield curves:
        1. Linear on zero rates
        2. Cubic spline
        3. Nelson-Siegel-Svensson (NSS)

    All methods take a DiscountCurve as input and return
    spot rates and forward rates at arbitrary maturities.
    """

    def __init__(self, curve: DiscountCurve):
        """
        Parameters
        ----------
        curve : DiscountCurve  Bootstrapped discount curve
        """
        self.curve = curve
        self.t_obs = curve.maturities
        self.r_obs = curve.spot_rates

        # Pre-fit cubic spline on initialization
        self._spline = interpolate.CubicSpline(
            self.t_obs, self.r_obs,
            bc_type="not-a-knot"
        )

    # ---- 1. LINEAR INTERPOLATION ---- #

    def linear(self, t: np.ndarray) -> np.ndarray:
        """
        Linear interpolation on zero rates.

        Formula:
            r(t) = r(t_i) + [r(t_{i+1}) - r(t_i)] /
                   [t_{i+1} - t_i] * (t - t_i)

        Implies piecewise constant forward rates with
        discontinuous jumps at nodes.

        Parameters
        ----------
        t : np.ndarray  Maturities to interpolate

        Returns
        -------
        np.ndarray : Interpolated spot rates
        """
        return np.interp(t, self.t_obs, self.r_obs)

    # ---- 2. CUBIC SPLINE ---- #

    def cubic_spline(self, t: np.ndarray) -> np.ndarray:
        """
        Cubic spline interpolation on zero rates.

        Fits a piecewise cubic polynomial S(t) such that
        S(t), S'(t), S''(t) are continuous at every node.
        Uses not-a-knot boundary conditions.

        Implies continuous forward rates everywhere since
        f(t) = r(t) + t*r'(t) and r'(t) is continuous.

        Parameters
        ----------
        t : np.ndarray  Maturities to interpolate

        Returns
        -------
        np.ndarray : Interpolated spot rates
        """
        return self._spline(t)

    def cubic_spline_forward(self, t: np.ndarray) -> np.ndarray:
        """
        Instantaneous forward rate from cubic spline.

        Formula:
            f(t) = r(t) + t * r'(t)

        Uses analytical derivative of the cubic spline.

        Parameters
        ----------
        t : np.ndarray  Maturities

        Returns
        -------
        np.ndarray : Forward rates f(t)
        """
        r      = self._spline(t)
        r_prime = self._spline(t, 1)   # first derivative
        return r + t * r_prime

    # ---- 3. NELSON-SIEGEL-SVENSSON ---- #

    @staticmethod
    def nss_spot(t: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """
        NSS spot rate curve.

        Formula:
            r(t) = b0
                 + b1 * [(1 - e^(-t/l1)) / (t/l1)]
                 + b2 * [(1 - e^(-t/l1)) / (t/l1)
                         - e^(-t/l1)]
                 + b3 * [(1 - e^(-t/l2)) / (t/l2)
                         - e^(-t/l2)]

        Parameters
        ----------
        t    : np.ndarray  Maturities
        beta : np.ndarray  [b0, b1, b2, b3, l1, l2]

        Returns
        -------
        np.ndarray : NSS spot rates
        """
        b0, b1, b2, b3, l1, l2 = beta
        t = np.asarray(t, dtype=float)

        e1 = np.exp(-t / l1)
        e2 = np.exp(-t / l2)

        # Loading factors
        f1 = (1 - e1) / (t / l1)       # slope factor
        f2 = f1 - e1                   # curvature factor 1
        f3 = (1 - e2) / (t / l2) - e2  # curvature factor 2

        return b0 + b1 * f1 + b2 * f2 + b3 * f3

    @staticmethod
    def nss_forward(t: np.ndarray,
                    beta: np.ndarray) -> np.ndarray:
        """
        NSS instantaneous forward rate curve.

        Formula:
            f(t) = b0
                 + b1 * e^(-t/l1)
                 + b2 * (t/l1) * e^(-t/l1)
                 + b3 * (t/l2) * e^(-t/l2)

        Parameters
        ----------
        t    : np.ndarray  Maturities
        beta : np.ndarray  [b0, b1, b2, b3, l1, l2]

        Returns
        -------
        np.ndarray : NSS instantaneous forward rates
        """
        b0, b1, b2, b3, l1, l2 = beta
        t = np.asarray(t, dtype=float)

        e1 = np.exp(-t / l1)
        e2 = np.exp(-t / l2)

        return b0 + b1*e1 + b2*(t/l1)*e1 + b3*(t/l2)*e2

    def fit_nss(self,
                beta0: np.ndarray = None) -> np.ndarray:
        """
        Calibrate NSS parameters by minimizing squared
        errors between model and observed spot rates.

        Objective:
            min_beta sum [r_obs(t_i) - r_NSS(t_i; beta)]^2

        Subject to: b0 > 0, l1 > 0, l2 > 0

        Uses L-BFGS-B optimizer with bounds.

        Parameters
        ----------
        beta0 : np.ndarray  Initial guess [b0,b1,b2,b3,l1,l2]
                            Default: [0.03,-0.02,0.01,0.01,1,5]

        Returns
        -------
        np.ndarray : Calibrated beta [b0,b1,b2,b3,l1,l2]
        """
        if beta0 is None:
            beta0 = np.array([0.03, -0.02, 0.01,
                              0.01, 1.0, 5.0])

        def objective(beta):
            r_model = self.nss_spot(self.t_obs, beta)
            return np.sum((self.r_obs - r_model) ** 2)

        # Bounds: b0>0, b1 free, b2 free, b3 free,
        #         l1>0.01, l2>0.01
        bounds = [
            (1e-4, None),   # b0 > 0
            (None, None),   # b1 free
            (None, None),   # b2 free
            (None, None),   # b3 free
            (0.01, None),   # l1 > 0
            (0.01, None),   # l2 > 0
        ]

        result = optimize.minimize(
            objective, beta0,
            method  = "L-BFGS-B",
            bounds  = bounds,
            options = {"ftol": 1e-12, "maxiter": 10000}
        )

        if not result.success:
            print(f"Warning: NSS calibration did not "
                  f"converge. Message: {result.message}")

        return result.x

    def nss_summary(self, beta: np.ndarray) -> dict:
        """
        NSS parameter summary with economic interpretation.

        Parameters
        ----------
        beta : np.ndarray  Calibrated [b0,b1,b2,b3,l1,l2]

        Returns
        -------
        dict : Parameters with labels and limits
        """
        b0, b1, b2, b3, l1, l2 = beta
        return {
            "Long-run level  (b0)":    round(b0, 6),
            "Short-run adj   (b1)":    round(b1, 6),
            "Curvature 1     (b2)":    round(b2, 6),
            "Curvature 2     (b3)":    round(b3, 6),
            "Decay 1         (l1)":    round(l1, 6),
            "Decay 2         (l2)":    round(l2, 6),
            "Short rate lim  (b0+b1)": round(b0+b1, 6),
            "Long rate lim   (b0)":    round(b0, 6),
        }

    # ---- 4. COMPARISON ---- #

    def compare(self, t_grid: np.ndarray,
                beta: np.ndarray) -> dict:
        """
        Compare all three methods on a maturity grid.

        Parameters
        ----------
        t_grid : np.ndarray  Fine maturity grid
        beta   : np.ndarray  Calibrated NSS parameters

        Returns
        -------
        dict with spot rates from all three methods
        """
        return {
            "maturities":   t_grid,
            "linear":       self.linear(t_grid),
            "cubic_spline": self.cubic_spline(t_grid),
            "nss":          self.nss_spot(t_grid, beta),
            "forward_linear":  np.array([
                self.curve.instantaneous_forward(t)
                for t in t_grid]),
            "forward_spline":  self.cubic_spline_forward(
                t_grid),
            "forward_nss":  self.nss_forward(t_grid, beta),
        }