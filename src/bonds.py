"""
bonds.py
========
Bond pricing, yield-to-maturity and risk metrics.

Mathematical reference:
    Palma, R. (2026). Fixed Income Analytics.
"""

import numpy as np
from scipy import optimize
from scipy.stats import norm


# ---- 1. BOND PRICING --- #

def bond_price(C: float, F: float, r: float, n: int) -> float:
    """
    Price of a bond with annual coupon payments.

    Formula (closed form):
        P = C * [1 - (1 + r)^(-n)] / r + F * (1 + r)^(-n)

    Parameters
    ----------
    C : float  Coupon payment per period
    F : float  Face value (principal)
    r : float  Discount rate per period
    n : int    Number of periods (years)

    Returns
    -------
    float : Bond price P
    """
    if r == 0:
        return C * n + F
    discount = (1 + r) ** (-n)
    return C * (1 - discount) / r + F * discount


def bond_price_T_compoundings(C: float, F: float, r: float,
                               n: int, T: int) -> float:
    """
    Price of a bond where r compounds T times per year.

    Formula:
        P = C * [1 - (1 + r/T)^(-nT)] / [(1 + r/T)^T - 1]
          + F * (1 + r/T)^(-nT)

    Parameters
    ----------
    C : float  Annual coupon payment
    F : float  Face value
    r : float  Annual interest rate
    n : int    Maturity in years
    T : int    Compoundings per year

    Returns
    -------
    float : Bond price P
    """
    factor = (1 + r / T) ** (n * T)
    denominator = (1 + r / T) ** T - 1
    if denominator == 0:
        return C * n + F
    return C * (1 - 1 / factor) / denominator + F / factor


def bond_price_continuous(C: float, F: float, r: float, n: int) -> float:
    """
    Price of a bond under continuous compounding.

    Formula (limit T -> infinity):
        P = C * [1 - e^(-rn)] / [e^r - 1] + F * e^(-rn)

    Parameters
    ----------
    C : float  Annual coupon
    F : float  Face value
    r : float  Continuously compounded rate
    n : int    Maturity in years

    Returns
    -------
    float : Bond price P
    """
    discount = np.exp(-r * n)
    denominator = np.exp(r) - 1
    if denominator == 0:
        return C * n + F
    return C * (1 - discount) / denominator + F * discount


def cash_flows(C: float, F: float, n: int) -> np.ndarray:
    """
    Generate the cash flow vector CF_i for a bond.

    CF_i = C        for i = 1, ..., n-1
    CF_n = C + F    at maturity

    Parameters
    ----------
    C : float  Coupon payment
    F : float  Face value
    n : int    Number of periods

    Returns
    -------
    np.ndarray : Array of cash flows of length n
    """
    cf = np.full(n, C)
    cf[-1] += F
    return cf


# ---- 2. YIELD-TO-MATURITY (YTM) ---- #

def ytm(price: float, C: float, F: float, n: int,
        guess: float = 0.05) -> float:
    """
    Yield to Maturity via Newton-Raphson.

    Finds y such that:
        P = sum_{i = 1}^{n} C/(1 + y)^i + F/(1 + y)^n

    Has no closed-form solution -> solved numerically.

    Parameters
    ----------
    price : float  Observed market price
    C     : float  Coupon payment
    F     : float  Face value
    n     : int    Maturity in years
    guess : float  Initial guess (default 5%)

    Returns
    -------
    float : YTM y
    """
    cf = cash_flows(C, F, n)
    periods = np.arange(1, n + 1)

    def objective(y):
        return np.sum(cf / (1 + y) ** periods) - price

    def derivative(y):
        return -np.sum(periods * cf / (1 + y) ** (periods + 1))

    result = optimize.newton(objective, guess, fprime=derivative,
                             tol=1e-10, maxiter=1000)
    return result


def ytm_continuous(price: float, C: float, F: float, n: int,
                   guess: float = 0.05) -> float:
    """
    YTM under continuous compounding.

    Finds y such that:
        P = sum CF_i * e^(-y*i)

    Parameters
    ----------
    price : float  Market price
    C     : float  Coupon
    F     : float  Face value
    n     : int    Maturity
    guess : float  Initial guess

    Returns
    -------
    float : Continuously compounded YTM
    """
    cf = cash_flows(C, F, n)
    periods = np.arange(1, n + 1)

    def objective(y):
        return np.sum(cf * np.exp(-y * periods)) - price

    def derivative(y):
        return -np.sum(periods * cf * np.exp(-y * periods))

    return optimize.newton(objective, guess, fprime = derivative,
                           tol = 1e-10, maxiter = 1000)


# ---- 3. DURATION & CONVEXITY ---- #

def macaulay_duration(C: float, F: float, y: float,
                      n: int, T: int = 1) -> float:
    """
    Macaulay Duration: weighted average time of cash flows.

    Formula:
        D_mac = (1/P) * sum_{i = 1}^{n} i * CF_i / (1 + y/T)^(iT)

    Parameters
    ----------
    C : float  Coupon payment
    F : float  Face value
    y : float  Yield (YTM)
    n : int    Maturity in years
    T : int    Compoundings per year (default 1)

    Returns
    -------
    float : Macaulay Duration in years
    """
    cf = cash_flows(C, F, n)
    periods = np.arange(1, n + 1)
    discount = (1 + y / T) ** (periods * T)
    P = np.sum(cf / discount)
    return np.sum(periods * cf / discount) / P


def modified_duration(C: float, F: float, y: float,
                      n: int, T: int = 1) -> float:
    """
    Modified Duration: price sensitivity to yield changes.

    Formula:
        D_mod = D_mac / (1 + y/T)

    Interpretation:
        dP/dy = -D_mod * P

    Parameters
    ----------
    C : float  Coupon
    F : float  Face value
    y : float  YTM
    n : int    Maturity
    T : int    Compoundings per year

    Returns
    -------
    float : Modified Duration
    """
    D_mac = macaulay_duration(C, F, y, n, T)
    return D_mac / (1 + y / T)


def convexity(C: float, F: float, y: float,
              n: int, T: int = 1) -> float:
    """
    Bond Convexity: second-order price sensitivity.

    Formula:
        Cx = (1/P) * sum_{i = 1}^{n} [i(iT + 1)/T] * CF_i / (1 + y/T)^(iT + 2)

    Parameters
    ----------
    C : float  Coupon
    F : float  Face value
    y : float  YTM
    n : int    Maturity
    T : int    Compoundings per year

    Returns
    -------
    float : Convexity
    """
    cf = cash_flows(C, F, n)
    periods = np.arange(1, n + 1)
    u = 1 + y / T
    P = np.sum(cf / u ** (periods * T))
    weights = periods * (periods * T + 1) / T
    return np.sum(weights * cf / u ** (periods * T + 2)) / P


def dv01(C: float, F: float, y: float, n: int, T: int = 1) -> float:
    """
    DV01: Dollar Value of a Basis Point.

    Price change for a 1bp (0.01%) move in yield:
        DV01 = D_mod * P * 0.0001

    Parameters
    ----------
    C : float  Coupon
    F : float  Face value
    y : float  YTM
    n : int    Maturity
    T : int    Compoundings per year

    Returns
    -------
    float : DV01 in currency units
    """
    cf = cash_flows(C, F, n)
    periods = np.arange(1, n + 1)
    P = np.sum(cf / (1 + y / T) ** (periods * T))
    D_mod = modified_duration(C, F, y, n, T)
    return D_mod * P * 0.0001


# ---- 4. PRICE APPROXIMATION ---- #

def price_change_approximation(P: float, D_mod: float,
                                Cx: float, dy: float) -> float:
    """
    Second-order Taylor approximation of price change.

    Formula:
        dP ≈ -D_mod * P * dy + (1/2) * Cx * P * dy^2

    Parameters
    ----------
    P     : float  Current bond price
    D_mod : float  Modified Duration
    Cx    : float  Convexity
    dy    : float  Yield change (e.g. 0.01 for 100bps)

    Returns
    -------
    float : Approximate price change dP
    """
    return -D_mod * P * dy + 0.5 * Cx * P * dy ** 2


def price_parity(C: float, F: float, y: float, n: int) -> str:
    """
    Determine bond parity given coupon rate and yield.

    Result:
        y = C/F  ->  P = F  (at par)
        y > C/F  ->  P < F  (discount bond)
        y < C/F  ->  P > F  (premium bond)

    Parameters
    ----------
    C : float  Coupon
    F : float  Face value
    y : float  YTM

    Returns
    -------
    str : Parity description
    """
    coupon_rate = C / F
    P = bond_price(C, F, y, n)
    if np.isclose(y, coupon_rate, atol = 1e-6):
        return f"At Par: P = F = {F:.2f}"
    elif y > coupon_rate:
        return f"Discount Bond: P = {P:.4f} < F = {F:.2f}"
    else:
        return f"Premium Bond: P = {P:.4f} > F = {F:.2f}"


# ---- 5. VALIDATION ---- #

def validate_ytm_roundtrip(C: float, F: float, r: float, n: int,
                            T: int = 1) -> dict:
    """
    Validation: price a bond, extract YTM, reprice and verify.

    A correct implementation must satisfy:
        bond_price(C, F, ytm(P, C, F, n), n) == P

    Parameters
    ----------
    C : float  Coupon
    F : float  Face value
    r : float  Discount rate
    n : int    Maturity
    T : int    Compoundings

    Returns
    -------
    dict with original price, recovered price and error
    """
    P_original = bond_price_T_compoundings(C, F, r, n, T)
    y_recovered = ytm(P_original, C, F, n)
    P_recovered = bond_price(C, F, y_recovered, n)
    error = abs(P_original - P_recovered)
    return {
        "original_price":  round(P_original, 8),
        "recovered_price": round(P_recovered, 8),
        "ytm_recovered":   round(y_recovered, 8),
        "abs_error":       round(error, 10),
        "passed":          error < 1e-6
    }