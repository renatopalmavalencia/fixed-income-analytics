"""
bonds.py
========
Bond pricing, yield-to-maturity and risk metrics.

Mathematical reference:
    Palma, R. (2026). Fixed Income Analytics.
"""

import numpy as np
from scipy import optimize

class VanillaBond:
    """
    It represents a standard bullet bond (vanilla bond).
    Encapsules pricing, YTM calculation and risk measures.
    """

    def __init__(self, coupon_rate: float, face_value: float, maturity: int, frequency: int = 1):
        """
        Bond attributes.

        Parameters
        ----------
        coupon_rate: Annual coupon rate
        face_value: Par value or face value
        maturity: Years to maturity
        frequency: Coupon payments per year (1 = annual, 2 = semi-annual, 4 = quarterly)
        """
        assert coupon_rate >= 0, "Coupon rate must be non-negative"
        assert face_value > 0,   "Face value must be positive"
        assert maturity > 0,     "Maturity must be positive"
        assert frequency in [1, 2, 4, 12], "Frequency must be 1, 2, 4 or 12"
        
        self.C_annual = coupon_rate * face_value
        self.F = face_value
        self.T = maturity
        self.freq = frequency
        self.c = self.C_annual / frequency
        self.n = maturity * frequency

    def __repr__(self) -> str:
        return (f"VanillaBond(coupon_rate={self.C_annual/self.F:.2%}, "
                f"face_value={self.F}, maturity={self.T}, "
                f"frequency={self.freq})")

    # ---- 2. BOND PRICING ---- #

    def bond_price(self, ytm: float) -> float:
         """
        Bond price given a yield (YTM).

        Formula:
            P = c * [1 - (1+y)^(-n)] / y + F * (1+y)^(-n)
            where y = ytm / freq (periodic yield)

        Parameters
        ----------
        ytm : float  Annual yield to maturity

        Returns
        -------
        float : Bond price P
        """
        ytm = np.asarray(ytm)
        y = ytm / self.freq
        
        if np.any(y == 0):
            # Caso tasa cero: suma simple de flujos
            return self.c * self.n + self.F
            
        discount = (1 + y) ** (-self.n)
        price = self.c * (1 - discount) / y + self.F * discount
        return price

    def bond_price_continuous(self, r: float) -> float:
        """
        Bond price under continuous compounding.

        Formula (limit freq -> infinity):
            P = C * [1 - e^(-rT)] / [e^r - 1] + F * e^(-rT)

        Parameters
        ----------
        r : float  Continuously compounded rate

        Returns
        -------
        float : Bond price P
        """
        discount = np.exp(-r * self.T)
        denom = np.exp(r) - 1
        if np.isclose(denom, 0):
            return self.C_annual * self.T + self.F
        return self.C_annual * (1 - discount) / denom + self.F * discount


    # ---- 3. YIELD TO MATURITY (YTM) ---- #
    
    def ytm(self, market_price: float, guess: float = 0.05) -> float:
        """
        Yield to Maturity via Newton-Raphson.

        Finds y such that bond_price(y) == market_price.
        Has no closed-form solution -> solved numerically.

        Parameters
        ----------
        market_price : float  Observed market price
        guess        : float  Initial guess (default 5%)

        Returns
        -------
        float : YTM (annual)
        """
        f = lambda y: self.bond_price(y) - market_price
        
        try:
            # optimize.newton es eficiente para encontrar raíces en finanzas
            implied_ytm = optimize.newton(f, guess, tol=1e-8, maxiter=100)
            return implied_ytm
        except RuntimeError:
            print(f"Error: Newton did not converge for price = {market_price}")
            return np.nan


    # ---- 4. RISK METRICS ---- #
    
    def risk_metrics(self, ytm: float) -> dict:
        """
        Compute all risk metrics given a YTM.

        Metrics computed:
            - Price
            - Macaulay Duration
            - Modified Duration
            - Convexity
            - DV01

        Formula references (Palma, 2026):
            D_mac = (1/P) * sum i * CF_i / (1+y)^(iT)
            D_mod = D_mac / (1 + y/T)
            Cx    = (1/P) * sum [i(iT+1)/T] * CF_i / (1+y)^(iT+2)
            DV01  = D_mod * P * 0.0001

        Parameters
        ----------
        ytm : float  Annual yield to maturity

        Returns
        -------
        dict with all metrics
        """
        y = ytm / self.freq
        periods = np.arange(1, self.n + 1)
        
        # Cash Flows
        cash_flows = np.full(self.n, self.c)
        cash_flows[-1] += self.F
        
        # Present Value of each flow
        pv_flows = cash_flows / (1 + y)**periods
        total_price = np.sum(pv_flows)
        if total_price <= 0:
            raise ValueError(f"Invalid price {total_price:.4f}. "
                             f"Check ytm={ytm:.4f} is reasonable.")
        
        # Macaulay Duration
        mac_dur = np.sum(pv_flows * periods) / (total_price * self.freq)
        
        # Modified Duration
        mod_dur = mac_dur / (1 + y)
        
        # DV01 (Dollar Value of a basis point)
        dv01 = mod_dur * total_price * 0.0001
        
        # Convexity (approximation for Bullet bonds)
        weights = periods * (periods * self.freq + 1) / self.freq
        convexity = np.sum(weights * cash_flows / (1 + y)**(periods + 2)) / total_price
        
        return {
            "Price": total_price,
            "Mac_Duration": mac_dur,
            "Mod_Duration": mod_dur,
            "Convexity": convexity,
            "DV01": dv01
        }

    # ---- 5. PRICE APPROXIMATION ---- #
    
    def price_change(self, ytm: float, dy: float) -> dict:
        """
        Second-order Taylor approximation of price change.

        Formula:
            dP ≈ -D_mod * P * dy + (1/2) * Cx * P * dy^2

        Parameters
        ----------
        ytm : float  Current YTM
        dy  : float  Yield change (e.g. 0.01 = +100bps)

        Returns
        -------
        dict with exact and approximate price changes
        """
        metrics   = self.risk_metrics(ytm)
        P         = metrics["Price"]
        D_mod     = metrics["Mod_Duration"]
        Cx        = metrics["Convexity"]

        dP_approx = -D_mod * P * dy + 0.5 * Cx * P * dy ** 2
        P_exact   = self.bond_price(ytm + dy)
        dP_exact  = P_exact - P

        return {
            "P_initial":    round(P, 6),
            "P_exact":      round(P_exact, 6),
            "dP_exact":     round(dP_exact, 6),
            "dP_approx":    round(dP_approx, 6),
            "error":        round(abs(dP_exact - dP_approx), 8)
        }

    # ---- 6. PARITY ---- #
    
    def parity(self, ytm: float) -> str:
        """
        Bond parity given coupon rate and YTM.

        Result:
            y = C/F  ->  P = F  (at par)
            y > C/F  ->  P < F  (discount bond)
            y < C/F  ->  P > F  (premium bond)

        Parameters
        ----------
        ytm : float  Annual yield

        Returns
        -------
        str : Parity description
        """
        coupon_rate = self.C_annual / self.F
        P = self.bond_price(ytm)
        if np.isclose(ytm, coupon_rate, atol=1e-6):
            return f"At Par: P = F = {self.F:.2f}"
        elif ytm > coupon_rate:
            return f"Discount Bond: P = {P:.4f} < F = {self.F:.2f}"
        else:
            return f"Premium Bond:  P = {P:.4f} > F = {self.F:.2f}"

    # ---- 7. VALIDATION ---- #

    def validate_roundtrip(self, r: float) -> dict:
        """
        Roundtrip validation: price -> YTM -> reprice.

        Must satisfy: bond_price(ytm(P)) == P

        Parameters
        ----------
        r : float  Discount rate used to price the bond

        Returns
        -------
        dict with prices, recovered YTM and error
        """
        P_original  = self.bond_price(r)
        y_recovered = self.ytm(P_original)
        P_recovered = self.bond_price(y_recovered)
        error       = abs(P_original - P_recovered)

        return {
            "original_price":  round(P_original, 8),
            "recovered_ytm":   round(y_recovered, 8),
            "recovered_price": round(P_recovered, 8),
            "abs_error":       round(error, 10),
            "passed":          error < 1e-6
        }