"""
pca.py
======
Principal Component Analysis on the yield curve
term structure.

Mathematical reference:
    Palma, R. (2026). Fixed Income Analytics.
    Module 4: PCA on Term Structure.

Key results:
    - Spectral decomposition of covariance matrix
    - Three factors: Level, Slope, Curvature
    - Variance explained by each component
    - PCA reconstruction of curve moves
    - Factor DV01 and P&L variance
    - Connection with NSS loading vectors
"""

import numpy as np
import pandas as pd
from scipy import linalg
from src.curves import DiscountCurve, CurveInterpolator


# ------------------------ #
# CLASS 1: YIELD CURVE PCA #
# ------------------------ #

class YieldCurvePCA:
    """
    PCA on historical yield curve changes.

    Decomposes the covariance matrix of daily rate changes
    into eigenvectors (factor loadings) and eigenvalues
    (variance explained).

    Parameters
    ----------
    df_yields  : pd.DataFrame  Historical yields
                               rows = dates, cols = maturities
    n_factors  : int           Number of factors to retain
                               (default 3)
    """

    FACTOR_NAMES = {
        0: "PC1 — Level",
        1: "PC2 — Slope",
        2: "PC3 — Curvature"
    }

    def __init__(self, df_yields: pd.DataFrame,
                 n_factors: int = 3):

        assert isinstance(df_yields, pd.DataFrame), \
            "df_yields must be a pandas DataFrame"
        assert n_factors >= 1, \
            "n_factors must be at least 1"
        assert n_factors <= df_yields.shape[1], \
            "n_factors cannot exceed number of maturities"

        self.df_yields  = df_yields
        self.maturities = np.array(df_yields.columns,
                                   dtype=float)
        self.n_factors  = n_factors

        # Compute changes and fit PCA
        self._fit()

    def __repr__(self) -> str:
        return (f"YieldCurvePCA("
                f"maturities={list(self.maturities)}, "
                f"n_factors={self.n_factors}, "
                f"T={self.T} days, "
                f"R2={self.variance_explained(self.n_factors):.2%})")

    # ---- 1. FIT ---- #

    def _fit(self):
        """
        Fit PCA on daily yield curve changes.

        Steps:
            1. Compute changes: X = diff(yields)
            2. Center: X -= mean(X)
            3. Covariance: Sigma = X.T @ X / (T-1)
            4. Spectral decomposition: Sigma = V @ L @ V.T
            5. Sort by decreasing eigenvalue
        """
        # Daily changes matrix X: shape (T, m)
        self.X = self.df_yields.diff().dropna()
        self.T = len(self.X)
        self.m = len(self.maturities)

        # Center
        self.X_centered = self.X - self.X.mean()

        # Covariance matrix: shape (m, m)
        X_arr        = self.X_centered.values
        self.Sigma   = (X_arr.T @ X_arr) / (self.T - 1)

        # Spectral decomposition
        # linalg.eigh is faster and more stable for
        # symmetric matrices
        eigenvalues, eigenvectors = linalg.eigh(self.Sigma)

        # Sort descending
        idx              = np.argsort(eigenvalues)[::-1]
        self.eigenvalues  = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]

        # Ensure consistent sign convention:
        # first element of each eigenvector is positive
        for k in range(self.m):
            if self.eigenvectors[0, k] < 0:
                self.eigenvectors[:, k] *= -1

    # ---- 2. VARIANCE EXPLAINED ---- #

    def variance_explained(self, K: int = None) -> float:
        """
        Fraction of total variance explained by first K PCs.

        Formula:
            R2_K = sum(lambda_1..K) / sum(lambda_1..m)

        Parameters
        ----------
        K : int  Number of components (default: n_factors)

        Returns
        -------
        float : R2_K in [0, 1]
        """
        if K is None:
            K = self.n_factors
        return (self.eigenvalues[:K].sum() /
                self.eigenvalues.sum())

    def variance_table(self) -> pd.DataFrame:
        """
        Variance explained table for all components.

        Returns
        -------
        pd.DataFrame with eigenvalue, individual and
        cumulative variance explained per component
        """
        total    = self.eigenvalues.sum()
        ind_var  = self.eigenvalues / total
        cum_var  = np.cumsum(ind_var)

        rows = []
        for k in range(self.m):
            name = self.FACTOR_NAMES.get(k, f"PC{k+1}")
            rows.append({
                "Component":    name,
                "Eigenvalue":   round(self.eigenvalues[k], 8),
                "Var Explained": f"{ind_var[k]:.4%}",
                "Cumulative":    f"{cum_var[k]:.4%}"
            })

        return pd.DataFrame(rows)

    # ---- 3. FACTOR LOADINGS ---- #

    def loadings(self, K: int = None) -> pd.DataFrame:
        """
        Factor loading matrix: eigenvectors as columns.

        Each column v_k describes how each maturity
        moves under factor k.

        Parameters
        ----------
        K : int  Number of factors (default: n_factors)

        Returns
        -------
        pd.DataFrame : shape (m, K)
                       rows = maturities, cols = factors
        """
        if K is None:
            K = self.n_factors

        cols = [self.FACTOR_NAMES.get(k, f"PC{k+1}")
                for k in range(K)]

        return pd.DataFrame(
            self.eigenvectors[:, :K],
            index   = self.maturities,
            columns = cols
        )

    # ---- 4. FACTOR SCORES ---- #

    def scores(self, K: int = None) -> pd.DataFrame:
        """
        Factor scores: projection of data onto PCs.

        Formula:
            F = X_centered @ V_K
            shape (T, K)

        Parameters
        ----------
        K : int  Number of factors

        Returns
        -------
        pd.DataFrame : shape (T, K)
                       rows = dates, cols = factors
        """
        if K is None:
            K = self.n_factors

        V_K   = self.eigenvectors[:, :K]
        F     = self.X_centered.values @ V_K
        cols  = [self.FACTOR_NAMES.get(k, f"PC{k+1}")
                 for k in range(K)]

        return pd.DataFrame(F,
                            index   = self.X_centered.index,
                            columns = cols)

    # ---- 5. RECONSTRUCTION ---- #

    def reconstruct(self, K: int = None) -> pd.DataFrame:
        """
        Reconstruct yield curve changes using K factors.

        Formula:
            X_hat = F_K @ V_K.T = X_centered @ V_K @ V_K.T

        Parameters
        ----------
        K : int  Number of factors

        Returns
        -------
        pd.DataFrame : Reconstructed changes, shape (T, m)
        """
        if K is None:
            K = self.n_factors

        V_K   = self.eigenvectors[:, :K]
        F_K   = self.X_centered.values @ V_K
        X_hat = F_K @ V_K.T

        return pd.DataFrame(
            X_hat,
            index   = self.X_centered.index,
            columns = self.maturities
        )

    def reconstruction_error(self, K: int = None) -> dict:
        """
        Reconstruction error using K factors.

        Analytical formula:
            MSE = sum(lambda_{K+1}...lambda_m)

        Also computed empirically as validation.

        Parameters
        ----------
        K : int  Number of factors

        Returns
        -------
        dict with analytical and empirical MSE
        """
        if K is None:
            K = self.n_factors

        # Analytical
        mse_analytical = self.eigenvalues[K:].sum()

        # Empirical
        X_hat        = self.reconstruct(K).values
        X_true       = self.X_centered.values
        residuals    = X_true - X_hat
        mse_empirical = (residuals ** 2).mean()

        return {
            "K":               K,
            "mse_analytical":  round(mse_analytical, 10),
            "mse_empirical":   round(mse_empirical, 10),
            "match":           np.isclose(
                mse_analytical, mse_empirical, rtol=1e-3)
        }

    # ---- 6. FACTOR SHOCK ---- #

    def factor_shock(self, k: int,
                     sigma_multiple: float = 1.0
                     ) -> np.ndarray:
        """
        Yield curve change from a shock to factor k.

        A shock of sigma_multiple standard deviations
        along PC k produces:
            delta_r = sigma_multiple * sqrt(lambda_k) * v_k

        Parameters
        ----------
        k              : int    Factor index (0-based)
        sigma_multiple : float  Size of shock in std devs
                                (default: 1 sigma)

        Returns
        -------
        np.ndarray : Rate change at each maturity (bps)
        """
        assert 0 <= k < self.m, \
            f"Factor index must be in [0, {self.m-1}]"

        shock = (sigma_multiple
                 * np.sqrt(self.eigenvalues[k])
                 * self.eigenvectors[:, k])

        return shock * 10000   # convert to bps

    # ---- 7. NSS CONNECTION ---- #

    def nss_alignment(self,
                      lambda_nss: float = 1.5) -> pd.DataFrame:
        """
        Measure alignment between PCA eigenvectors and
        NSS loading vectors.

        NSS loadings evaluated at observed maturities:
            phi_0 = 1  (level)
            phi_1 = (1 - e^(-t/l)) / (t/l)  (slope)
            phi_2 = phi_1 - e^(-t/l)  (curvature)

        Computes R^2 between each PC and its NSS counterpart.

        Parameters
        ----------
        lambda_nss : float  NSS decay parameter lambda
                            (default 1.5, typical value)

        Returns
        -------
        pd.DataFrame : R^2 between PCs and NSS loadings
        """
        t  = self.maturities
        e  = np.exp(-t / lambda_nss)
        f1 = (1 - e) / (t / lambda_nss)
        f2 = f1 - e

        nss_loadings = np.column_stack([
            np.ones(len(t)),   # phi_0: level
            f1,                # phi_1: slope
            f2                 # phi_2: curvature
        ])

        rows = []
        for k in range(min(3, self.n_factors)):
            pc_vec    = self.eigenvectors[:, k]
            nss_vec   = nss_loadings[:, k]

            # Normalize both
            pc_norm   = pc_vec  / np.linalg.norm(pc_vec)
            nss_norm  = nss_vec / np.linalg.norm(nss_vec)

            # R^2 as squared cosine similarity
            r2 = np.dot(pc_norm, nss_norm) ** 2

            rows.append({
                "PC":          self.FACTOR_NAMES.get(
                    k, f"PC{k+1}"),
                "NSS Loading": ["φ₀ (Level)",
                                "φ₁ (Slope)",
                                "φ₂ (Curvature)"][k],
                "R² alignment": f"{r2:.4f}",
                "Aligned":     r2 > 0.90
            })

        return pd.DataFrame(rows)


# ------------------------- #
# CLASS 2: PCA RISK MANAGER #
# ------------------------- #

class PCARiskManager:
    """
    Factor-based risk management using PCA decomposition.

    Connects KRD vectors from Module 3 with PCA factors
    to compute factor DV01s and P&L variance.

    Mathematical reference:
        DV01_PCk = v_k.T @ d
        Var(dV)  = d.T @ Sigma @ d
                 = sum_k lambda_k * (v_k.T @ d)^2

    Parameters
    ----------
    pca     : YieldCurvePCA   Fitted PCA object
    dv01_vec : np.ndarray      KRD DV01 vector from Module 3
    """

    def __init__(self, pca: YieldCurvePCA,
                 dv01_vec: np.ndarray):

        assert len(dv01_vec) == pca.m, \
            "dv01_vec must match number of PCA maturities"

        self.pca      = pca
        self.dv01_vec = np.asarray(dv01_vec)

    def __repr__(self) -> str:
        return (f"PCARiskManager("
                f"total_dv01=${self.total_dv01():.4f})")

    # ---- 1. FACTOR DV01 ---- #

    def factor_dv01(self) -> dict:
        """
        DV01 along each principal component.

        Formula:
            DV01_PCk = v_k.T @ d

        where d is the KRD DV01 vector.

        Returns
        -------
        dict : {factor_name: DV01}
        """
        result = {}
        for k in range(self.pca.n_factors):
            v_k   = self.pca.eigenvectors[:, k]
            name  = self.pca.FACTOR_NAMES.get(k, f"PC{k+1}")
            result[name] = float(np.dot(v_k, self.dv01_vec))

        return result

    def total_dv01(self) -> float:
        """
        Total parallel DV01 (sum of KRD DV01s).

        Returns
        -------
        float
        """
        return float(self.dv01_vec.sum())

    # ---- 2. PNL VARIANCE ---- #

    def pnl_variance(self, K: int = None) -> dict:
        """
        Portfolio P&L variance under factor model.

        Formula:
            Var(dV) = d.T @ Sigma @ d
                    ≈ sum_{k=1}^K lambda_k * (v_k.T @ d)^2

        Parameters
        ----------
        K : int  Number of factors (default: n_factors)

        Returns
        -------
        dict with exact and approximate variance,
        and contribution per factor
        """
        if K is None:
            K = self.pca.n_factors

        # Exact
        var_exact = float(
            self.dv01_vec @ self.pca.Sigma @ self.dv01_vec)

        # Approximate via K factors
        var_approx    = 0.0
        contributions = {}
        for k in range(K):
            v_k    = self.pca.eigenvectors[:, k]
            lam_k  = self.pca.eigenvalues[k]
            proj   = float(np.dot(v_k, self.dv01_vec))
            contrib = lam_k * proj ** 2
            var_approx += contrib
            name   = self.pca.FACTOR_NAMES.get(k, f"PC{k+1}")
            contributions[name] = round(contrib, 10)

        pnl_vol_exact  = np.sqrt(var_exact)
        pnl_vol_approx = np.sqrt(var_approx)

        return {
            "var_exact":       round(var_exact, 10),
            "var_approx":      round(var_approx, 10),
            "pnl_vol_exact":   round(pnl_vol_exact, 6),
            "pnl_vol_approx":  round(pnl_vol_approx, 6),
            "contributions":   contributions,
            "K":               K
        }

    # ---- 3. IMMUNIZATION CHECK ---- #

    def immunization_check(self,
                           tol: float = 1e-4) -> dict:
        """
        Check immunization conditions.

        Full immunization requires DV01 = 0 along ALL
        factors. Level immunization (most common) requires
        only DV01_PC1 = 0.

        As noted in Palma (2026): when DV01_PC1 = 0 the
        portfolio is immunized against parallel shifts —
        the dominant source of yield curve risk (PC1
        explains ~80% of total variance).

        Parameters
        ----------
        tol : float  Tolerance for near-zero check

        Returns
        -------
        dict : immunization status per factor
        """
        fdv01  = self.factor_dv01()
        result = {}

        for k, (name, dv01_k) in enumerate(fdv01.items()):
            immunized = abs(dv01_k) < tol
            result[name] = {
                "DV01":       round(dv01_k, 6),
                "immunized":  immunized,
                "pct_var":    f"{self.pca.eigenvalues[k] / self.pca.eigenvalues.sum():.2%}"
            }

        # Overall status
        all_immunized    = all(
            v["immunized"] for v in result.values())
        level_immunized  = result[
            self.pca.FACTOR_NAMES[0]]["immunized"]

        result["summary"] = {
            "level_immunized":  level_immunized,
            "fully_immunized":  all_immunized,
            "var_covered":      f"{self.pca.variance_explained(1):.2%}"
        }

        return result

    # ---- 4. SCENARIO PNL ---- #

    def scenario_pnl(self,
                     factor_shocks: dict) -> dict:
        """
        P&L decomposition by factor shock.

        Formula:
            dV_k = -DV01_PCk * (shock_k / 0.0001)

        Parameters
        ----------
        factor_shocks : dict  {factor_index: shock_in_bps}

        Returns
        -------
        dict : P&L per factor and total
        """
        fdv01  = self.factor_dv01()
        result = {}
        total  = 0.0

        for k, shock_bps in factor_shocks.items():
            name   = self.pca.FACTOR_NAMES.get(k, f"PC{k+1}")
            dv01_k = list(fdv01.values())[k]
            pnl_k  = -dv01_k * shock_bps
            result[name] = round(pnl_k, 4)
            total += pnl_k

        result["Total P&L"] = round(total, 4)
        return result

    # ---- 5. SUMMARY ---- #

    def summary(self) -> dict:
        """
        Full risk summary combining DV01 and PCA.

        Returns
        -------
        dict with factor DV01s, P&L variance,
        immunization status
        """
        fdv01   = self.factor_dv01()
        pnl_var = self.pnl_variance()
        immun   = self.immunization_check()

        return {
            "total_dv01":    round(self.total_dv01(), 6),
            "factor_dv01":   {k: round(v, 6)
                              for k, v in fdv01.items()},
            "pnl_vol":       pnl_var["pnl_vol_exact"],
            "var_explained": f"{self.pca.variance_explained():.2%}",
            "immunization":  immun["summary"]
        }