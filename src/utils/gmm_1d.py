"""
Shared 1D Gaussian Mixture Model utility.

§3a.3 / §3a.6 Staff Refinement: Both bar_associator.py and histogram_extractor.py
use 1D GMM EM. This shared utility avoids code duplication.

Exposes fit_gmm_1d(data, max_k=2) → GMMResult with best_k, params, responsibilities, bic_scores.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GMMComponent:
    """Parameters of a single Gaussian component."""
    weight: float   # π_k mixing weight
    mean: float     # μ_k
    variance: float  # σ²_k

    @property
    def std(self) -> float:
        return np.sqrt(max(self.variance, 1e-12))


@dataclass
class GMMResult:
    """Result of 1D GMM fitting."""
    best_k: int                          # Selected number of components (1 or 2)
    components: List[GMMComponent]       # Parameters per component
    responsibilities: np.ndarray         # (N, K) matrix of γ_ik
    bic_scores: dict                     # {k: BIC(k)} for each K tried
    log_likelihood: float                # Log-likelihood of best model
    converged: bool                      # Whether EM converged
    group_assignments: Optional[np.ndarray] = None  # Per-gap component assignment (argmax γ)


def _gaussian_pdf(x: np.ndarray, mu: float, var: float) -> np.ndarray:
    """Evaluate univariate Gaussian density."""
    var = max(var, 1e-12)  # Prevent division by zero
    return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2.0 * np.pi * var)


def _em_1d(data: np.ndarray, k: int, max_iter: int = 50, tol: float = 1e-6):
    """
    Run EM for a 1D GMM with K components.

    Returns (components, responsibilities, log_likelihood, converged).
    """
    n = len(data)
    if n == 0:
        return [], np.empty((0, k)), -np.inf, False

    # ── Initialization ──
    if k == 1:
        components = [GMMComponent(
            weight=1.0,
            mean=float(np.mean(data)),
            variance=float(np.var(data)) + 1e-6
        )]
    else:
        # K-means initialization for K=2
        sorted_data = np.sort(data)
        mid = n // 2
        cluster1 = sorted_data[:mid]
        cluster2 = sorted_data[mid:]
        components = [
            GMMComponent(
                weight=len(cluster1) / n,
                mean=float(np.mean(cluster1)),
                variance=float(np.var(cluster1)) + 1e-6
            ),
            GMMComponent(
                weight=len(cluster2) / n,
                mean=float(np.mean(cluster2)),
                variance=float(np.var(cluster2)) + 1e-6
            ),
        ]

    prev_ll = -np.inf
    gamma = np.zeros((n, k))
    converged = False

    for iteration in range(max_iter):
        # ── E-step: Compute responsibilities γ_ik ──
        for j, comp in enumerate(components):
            gamma[:, j] = comp.weight * _gaussian_pdf(data, comp.mean, comp.variance)

        row_sums = gamma.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-300)  # Prevent division by zero
        gamma /= row_sums

        # ── Log-likelihood ──
        ll = 0.0
        for j, comp in enumerate(components):
            ll += np.sum(np.log(np.maximum(
                comp.weight * _gaussian_pdf(data, comp.mean, comp.variance),
                1e-300
            )))
        # Correct log-likelihood using log-sum-exp
        log_mix = np.zeros(n)
        for j, comp in enumerate(components):
            log_mix += comp.weight * _gaussian_pdf(data, comp.mean, comp.variance)
        ll = float(np.sum(np.log(np.maximum(log_mix, 1e-300))))

        # ── Convergence check ──
        if abs(ll - prev_ll) < tol:
            converged = True
            break
        prev_ll = ll

        # ── M-step ──
        for j in range(k):
            N_k = np.sum(gamma[:, j])
            if N_k < 1e-6:
                continue
            components[j].weight = float(N_k / n)
            components[j].mean = float(np.sum(gamma[:, j] * data) / N_k)
            diff = data - components[j].mean
            components[j].variance = float(np.sum(gamma[:, j] * diff ** 2) / N_k) + 1e-6

    return components, gamma, ll, converged


def _bic(log_likelihood: float, n_params: int, n_samples: int) -> float:
    """
    Bayesian Information Criterion.

    BIC(K) = -2·ℓ_K + p_K·log(N)
    Parameter count for 1D GMM: p_K = 3K - 1 (weights: K-1, means: K, variances: K).
    """
    return -2.0 * log_likelihood + n_params * np.log(max(n_samples, 1))


def fit_gmm_1d(
    data: np.ndarray,
    max_k: int = 2,
    bic_delta: float = 3.0,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> GMMResult:
    """
    Fit a 1D Gaussian Mixture with K ∈ {1, 2, ..., max_k} components.

    Uses BIC model selection: accept K=2 only if BIC(2) + δ < BIC(1).

    Args:
        data: 1D array of observations (e.g., normalized inter-bar gaps).
        max_k: Maximum number of components (default 2).
        bic_delta: Safety margin for BIC comparison (§3a.3.3: δ ≈ 2-5).
        max_iter: Maximum EM iterations.
        tol: Log-likelihood convergence tolerance.

    Returns:
        GMMResult with best model parameters.
    """
    data = np.asarray(data, dtype=float)
    n = len(data)

    # Edge case: insufficient data
    if n < 3:
        return GMMResult(
            best_k=1,
            components=[GMMComponent(
                weight=1.0,
                mean=float(np.mean(data)) if n > 0 else 0.0,
                variance=float(np.var(data)) if n > 1 else 1.0
            )],
            responsibilities=np.ones((n, 1)) if n > 0 else np.empty((0, 1)),
            bic_scores={1: 0.0},
            log_likelihood=0.0,
            converged=True
        )

    # Fit K=1 and K=2
    results = {}
    bic_scores = {}

    for k in range(1, max_k + 1):
        components, gamma, ll, conv = _em_1d(data, k, max_iter, tol)
        n_params = 3 * k - 1
        bic_val = _bic(ll, n_params, n)
        results[k] = (components, gamma, ll, conv)
        bic_scores[k] = bic_val

    # BIC model selection
    best_k = 1
    if max_k >= 2 and 2 in bic_scores and 1 in bic_scores:
        if bic_scores[2] + bic_delta < bic_scores[1]:
            best_k = 2

    components, gamma, ll, conv = results[best_k]

    # Compute per-observation component assignments
    assignments = np.argmax(gamma, axis=1) if gamma.size > 0 else np.zeros(n, dtype=int)

    return GMMResult(
        best_k=best_k,
        components=components,
        responsibilities=gamma,
        bic_scores=bic_scores,
        log_likelihood=ll,
        converged=conv,
        group_assignments=assignments
    )


def find_group_separators(
    data: np.ndarray,
    gmm_result: GMMResult,
) -> List[int]:
    """
    Given sorted inter-element gaps and a K=2 GMM result, return the indices
    of gaps that are group separators (belong to the large-gap component).

    A group separator at index i means: start a new group after element i+1.
    Returns empty list if K=1 (no grouping detected).
    """
    if gmm_result.best_k < 2 or len(gmm_result.components) < 2:
        return []

    # Identify the large-gap component (higher mean)
    means = [c.mean for c in gmm_result.components]
    large_comp_idx = int(np.argmax(means))

    # Gaps assigned to the large component are group separators
    separators = []
    assignments = gmm_result.group_assignments
    if assignments is not None:
        for i, assignment in enumerate(assignments):
            if assignment == large_comp_idx:
                separators.append(i)

    return separators
