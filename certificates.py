"""Sufficient certificates for affine-constrained variance optimization."""

import numpy as np

C_0 = np.log(2.0) - 0.5
S_0 = 0.1854712448


def projection_statistic(A, b, r):
    """Compute ``S = d.T @ (A D_r^2 A.T)^(-1) @ d``."""

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    r = np.asarray(r, dtype=float)
    d = A @ r - b
    C = A * r
    middle = C @ C.T
    return float(d @ np.linalg.solve(middle, d))


def s_certificate(A, b, r):
    """Return whether the matrix-only certificate ``S < S_0`` passes."""

    return bool(projection_statistic(A, b, r) < S_0)


def likelihood_gap(x):
    """Return ``Psi(x) = sum(log(x_i) + 1/x_i - 1)``."""

    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or np.any(x <= 0) or not np.all(np.isfinite(x)):
        raise ValueError("x must contain finite, strictly positive values")
    return float(np.sum(np.log(x) + 1.0 / x - 1.0))


def projection_likelihood_gap(A, b, r):
    """Return ``T_proj = Psi(x_bar)`` for the relative projection.

    If the relative projection has a nonpositive coordinate, it is not a
    feasible variance vector and ``np.inf`` is returned.
    """

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    r = np.asarray(r, dtype=float)
    C = A * r
    d = A @ r - b
    correction = np.linalg.solve(C @ C.T, d)
    projection = np.ones(r.size) - C.T @ correction

    if np.any(projection <= 0):
        return np.inf
    return likelihood_gap(projection)


def projection_certificate(A, b, r):
    """Return whether the projection certificate ``T_proj < c_0`` passes."""

    return bool(projection_likelihood_gap(A, b, r) < C_0)


def nonnegative_dual_certificate(A, lagrange_multipliers, tolerance=1e-9):
    """Evaluate the nonnegative-dual certificate from Section 7.3.

    The caller must supply the equality-constraint multipliers of a positive
    feasible KKT point using the convention

        gradient(F)(omega) + A.T @ lagrange_multipliers = 0.

    The certificate passes when ``A.T @ lagrange_multipliers`` is
    coordinatewise nonnegative.  ``tolerance`` permits small negative values
    caused by numerical optimization error.
    """

    A = np.asarray(A, dtype=float)
    multipliers = np.asarray(lagrange_multipliers, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be a two-dimensional matrix")
    if multipliers.ndim == 0:
        multipliers = multipliers.reshape(1)
    elif multipliers.ndim == 2 and 1 in multipliers.shape:
        multipliers = multipliers.reshape(-1)
    elif multipliers.ndim != 1:
        raise ValueError("lagrange_multipliers must be a vector")
    if multipliers.size != A.shape[0]:
        raise ValueError(
            "A and lagrange_multipliers must have the same number of rows"
        )
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(multipliers)):
        raise ValueError("A and lagrange_multipliers must contain finite values")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")

    return bool(np.all(A.T @ multipliers >= -tolerance))


def rank_one_discrepancy_certificate(A, b, r):
    """Evaluate the rank-one discrepancy certificate.

    The condition is

        abs(a.T @ r - b) < min(abs(a_i) * r_i for a_i != 0).

    The certificate does not directly apply to a constraint matrix with more
    than one independent row, so this function rejects such input.
    """

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    r = np.asarray(r, dtype=float)
    if A.ndim != 2 or A.shape[0] != 1:
        raise ValueError(
            "The rank-one discrepancy certificate requires exactly one "
            "constraint row"
        )
    if r.ndim != 1 or A.shape[1] != r.size:
        raise ValueError("A and r have incompatible dimensions")
    if b.shape != (1,):
        raise ValueError("b must contain exactly one value")
    if np.any(r <= 0) or not np.all(np.isfinite(r)):
        raise ValueError("r must contain finite, strictly positive values")

    a = A[0]
    active = a != 0
    if not np.any(active):
        raise ValueError("The constraint row must contain a nonzero coefficient")
    discrepancy = abs(a @ r - b[0])
    threshold = np.min(np.abs(a[active]) * r[active])
    return bool(discrepancy < threshold)


def positive_rank_one_certificate(A, b, r):
    """Evaluate the positive rank-one certificate.

    After a possible sign change, all active entries of the single row must be
    positive.  For ``s >= 2`` active entries, the sufficient condition is

        a.T @ r > sqrt(s) / (sqrt(s) + 1) * b.

    When ``s == 1``, the constraint fixes the only active variance directly,
    so the rank-one problem is immediate and the function returns ``True``.
    """

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    r = np.asarray(r, dtype=float)
    if A.ndim != 2 or A.shape[0] != 1:
        raise ValueError("The positive rank-one certificate requires one row")
    if r.ndim != 1 or A.shape[1] != r.size:
        raise ValueError("A and r have incompatible dimensions")
    if b.shape != (1,):
        raise ValueError("b must contain exactly one value")
    if np.any(r <= 0) or not np.all(np.isfinite(r)):
        raise ValueError("r must contain finite, strictly positive values")

    a = A[0].copy()
    active = a != 0
    if not np.any(active):
        raise ValueError("The constraint row must contain a nonzero coefficient")
    if np.all(a[active] < 0):
        a = -a
        b_value = -b[0]
    elif np.all(a[active] > 0):
        b_value = b[0]
    else:
        raise ValueError("All active coefficients must have the same sign")
    if not np.isfinite(b_value) or b_value <= 0:
        raise ValueError("The sign-normalized right-hand side must be positive")

    s = int(np.count_nonzero(active))
    if s == 1:
        return True
    root_s = np.sqrt(s)
    threshold = root_s / (root_s + 1.0) * b_value
    return bool(a @ r > threshold)
