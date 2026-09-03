import time
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.linalg import qr
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize

from constraint_components import (
    ConstraintComponent,
    decompose_affine_constraints,
    unconstrained_columns,
)
from utils import is_DAG


@dataclass(frozen=True)
class SolverProblem:
    """Validated inputs and constants shared by all solver backends."""

    X: np.ndarray
    G: np.ndarray
    A: np.ndarray
    b: np.ndarray
    n: int
    p: int
    S: np.ndarray


@dataclass(frozen=True)
class ComponentSolution:
    """Result and certification metadata for one constraint component."""

    columns: Tuple[int, ...]
    omega: np.ndarray
    objective: float
    method: str
    certified_global: bool
    candidates_checked: int


@dataclass(frozen=True)
class ConstrainedMLEResult:
    """Complete constrained fit for a fixed DAG."""

    lambda_hat: np.ndarray
    omega_hat: np.ndarray
    log_likelihood: float
    certified_global: bool
    components: Tuple[ComponentSolution, ...]


@dataclass(frozen=True)
class AllNearFeasibilityResult:
    """Solution of the strict all-near feasibility LP from Section 7.1.

    ``status`` is one of ``"strict_feasible"``, ``"boundary_only"``, or
    ``"infeasible"``.  In the infeasible case, ``x``, ``omega``, and
    ``margin`` are ``None``.
    """

    x: Optional[np.ndarray]
    omega: Optional[np.ndarray]
    margin: Optional[float]
    status: str
    solver_status: int
    message: str

    @property
    def closed_box_feasible(self):
        return self.status != "infeasible"

    @property
    def strict_feasible(self):
        return self.status == "strict_feasible"


@dataclass(frozen=True)
class AllNearRootResult:
    """Optimizer of the likelihood over the convex all-near region."""

    residual_variances: np.ndarray
    x: np.ndarray
    omega: np.ndarray
    lagrange_multipliers: np.ndarray
    objective_gap: float
    certified_global: bool
    iterations: int
    constraint_violation: float
    solver_status: int
    message: str


def _positive_feasibility_margin(A, b):
    """Maximize the smallest variance subject to ``A @ omega = b``.

    Bounding the auxiliary margin by one prevents an unbounded LP when the
    affine set contains a strictly positive recession direction.  A positive
    optimum is equivalent to the existence of some ``omega > 0``.
    """

    number_of_variables = A.shape[1]
    objective = np.zeros(number_of_variables + 1)
    objective[-1] = -1.0
    equality_matrix = np.column_stack((A, np.zeros(A.shape[0])))
    inequality_matrix = np.column_stack(
        (-np.eye(number_of_variables), np.ones(number_of_variables))
    )
    result = linprog(
        objective,
        A_ub=inequality_matrix,
        b_ub=np.zeros(number_of_variables),
        A_eq=equality_matrix,
        b_eq=b,
        bounds=[(None, None)] * number_of_variables + [(0.0, 1.0)],
        method="highs",
    )
    if result.status == 2:
        return 0.0
    if not result.success:
        raise RuntimeError(f"Positive-feasibility LP failed: {result.message}")
    return result.x[-1]


def solve_all_near_feasibility_lp(A, b, r, tolerance=1e-9):
    """Run the margin LP in Section 7.1 of the report.

    For positive unconstrained residual variances ``r``, define
    ``D_r = diag(r)``, ``x = D_r^{-1} omega``, and ``C = A D_r``.  This
    function solves

        maximize    t
        subject to  C x = b,
                    t <= x_i <= 2 - t,
                    t >= 0.

    A positive optimal margin means that ``A omega = b`` has a point with
    every normalized variance strictly between zero and two.  A zero margin
    means that the closed box is feasible only on its boundary.  If the
    closed box is infeasible, the returned solution fields are ``None``.
    """

    A = np.asarray(A, dtype=float)
    r = np.asarray(r, dtype=float)
    b = np.asarray(b, dtype=float)

    if A.ndim != 2:
        raise ValueError("A must be a two-dimensional matrix")
    number_of_constraints, number_of_variables = A.shape
    if r.ndim != 1 or r.shape[0] != number_of_variables:
        raise ValueError(f"r must be a vector of length {number_of_variables}")
    if b.ndim == 0:
        b = b.reshape(1)
    elif b.ndim == 2 and 1 in b.shape:
        b = b.reshape(-1)
    elif b.ndim != 1:
        raise ValueError("b must be a vector or a one-column matrix")
    if b.shape[0] != number_of_constraints:
        raise ValueError("A and b must have the same number of rows")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise ValueError("A and b must contain only finite values")
    if not np.all(np.isfinite(r)) or np.any(r <= 0):
        raise ValueError("r must contain only finite, strictly positive values")
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")

    # A @ diag(r), computed without constructing the diagonal matrix.
    C = A * r
    objective = np.zeros(number_of_variables + 1)
    objective[-1] = -1.0

    # -x_i + t <= 0 and x_i + t <= 2 encode
    # t <= x_i <= 2 - t.
    A_ub = np.vstack(
        (
            np.column_stack((-np.eye(number_of_variables), np.ones(number_of_variables))),
            np.column_stack((np.eye(number_of_variables), np.ones(number_of_variables))),
        )
    )
    b_ub = np.concatenate(
        (np.zeros(number_of_variables), 2.0 * np.ones(number_of_variables))
    )
    A_eq = (
        np.column_stack((C, np.zeros(number_of_constraints)))
        if number_of_constraints
        else None
    )
    b_eq = b if number_of_constraints else None

    optimization = linprog(
        objective,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[(None, None)] * number_of_variables + [(0.0, None)],
        method="highs",
    )

    if optimization.status == 2:
        return AllNearFeasibilityResult(
            x=None,
            omega=None,
            margin=None,
            status="infeasible",
            solver_status=optimization.status,
            message=optimization.message,
        )
    if not optimization.success:
        raise RuntimeError(f"All-near feasibility LP failed: {optimization.message}")

    x = optimization.x[:-1]
    margin = max(0.0, float(optimization.x[-1]))
    status = "strict_feasible" if margin > tolerance else "boundary_only"
    return AllNearFeasibilityResult(
        x=x,
        omega=r * x,
        margin=margin,
        status=status,
        solver_status=optimization.status,
        message=optimization.message,
    )


def prepare_solver_problem(X, G, A, b, feasibility_tolerance=1e-10):
    """Validate inputs and compute the sample covariance.

    The data convention is ``X.shape == (n, p)``: rows are centered samples
    and columns are variables.  ``G[j, i] != 0`` represents the edge ``j -> i``.
    Constraints on only a subset of nodes must still be supplied with exactly
    ``p`` columns; unused nodes are represented by zero columns.
    """

    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a nonempty two-dimensional array")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values")
    n, p = X.shape

    G = np.asarray(G, dtype=float)
    if G.shape != (p, p):
        hint = " Did you forget to transpose X?" if G.shape == (n, n) else ""
        raise ValueError(f"G must have shape ({p}, {p}).{hint}")
    if not np.all(np.isfinite(G)):
        raise ValueError("G must contain only finite values")
    if not np.all((G == 0) | (G == 1)):
        raise ValueError("G must be a binary adjacency matrix")
    if np.any(np.diag(G) != 0):
        raise ValueError("G must not contain self-loops")
    if not is_DAG(G):
        raise ValueError("G must represent a directed acyclic graph")

    A = np.asarray(A, dtype=float)
    if A.size == 0:
        A = np.empty((0, p), dtype=float)
    if A.ndim != 2 or A.shape[1] != p:
        raise ValueError(f"A must be a two-dimensional matrix with {p} columns")
    if not np.all(np.isfinite(A)):
        raise ValueError("A must contain only finite values")

    b = np.asarray(b, dtype=float)
    if b.ndim == 0:
        b = b.reshape(1)
    elif b.ndim == 2 and 1 in b.shape:
        b = b.reshape(-1)
    elif b.ndim != 1:
        raise ValueError("b must be a vector or a one-column matrix")
    if b.shape[0] != A.shape[0]:
        raise ValueError("A and b must have the same number of rows")
    if not np.all(np.isfinite(b)):
        raise ValueError("b must contain only finite values")

    if A.shape[0] > 0:
        augmented = np.column_stack((A, b))
        if np.linalg.matrix_rank(A) < np.linalg.matrix_rank(augmented):
            raise ValueError("The affine system A omega = b is inconsistent")
        margin = _positive_feasibility_margin(A, b)
        if margin <= feasibility_tolerance:
            raise ValueError("The affine system is not positively feasible")

    S = X.T @ X / n
    return SolverProblem(X=X, G=G, A=A, b=b, n=n, p=p, S=S)


def compute_ols_estimates(X, G):
    """Compute the profiled OLS estimates for a fixed DAG.

    Rows of ``X`` are centered observations and columns are variables.
    The adjacency convention is ``G[j, i] == 1`` for an edge ``j -> i``.
    The returned matrix therefore stores the coefficient of ``j -> i`` at
    ``lambda_hat[j, i]``.  The residual variances use the Gaussian MLE
    divisor ``n``, rather than the degrees-of-freedom correction used for an
    unbiased variance estimate.

    Raises
    ------
    ValueError
        If a node's parent design matrix is rank deficient, in which case
        its OLS coefficient vector is not unique.
    """

    problem = prepare_solver_problem(X, G, [], [])
    X, G = problem.X, problem.G
    n, p = problem.n, problem.p

    lambda_hat = np.zeros((p, p), dtype=float)
    r = np.empty(p, dtype=float)

    for node in range(p):
        parents = np.flatnonzero(G[:, node])
        response = X[:, node]

        if parents.size:
            design = X[:, parents]
            coefficients, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
            if rank < parents.size:
                raise ValueError(
                    f"The parent design matrix for node {node} is rank "
                    "deficient; its OLS coefficients are not unique"
                )
            lambda_hat[parents, node] = coefficients
            residuals = response - design @ coefficients
        else:
            residuals = response

        r[node] = residuals @ residuals / n

    return lambda_hat, r


def solve_all_near_root(
    r,
    A,
    b,
    solver_tolerance=1e-10,
    maximum_iterations=1000,
):
    """Compute the unique likelihood optimizer in the all-near region.

    The residual-variance vector ``r`` is assumed to have already been
    profiled from the data and fitted DAG.  With ``D_r = diag(r)`` and
    ``x = D_r^{-1} omega``, this function minimizes

        Psi(x) = sum_i (log(x_i) + 1/x_i - 1)

    subject to ``A D_r x = b`` and ``0 < x_i <= 2``.  This problem is convex.
    SciPy's SLSQP implementation is supplied with the exact gradient and a
    strictly feasible starting point from :func:`solve_all_near_feasibility_lp`.

    The returned point is the all-near optimizer whether or not
    ``certified_global`` is true.  The latter is true precisely when its gap is
    below the universal far barrier ``log(2) - 1/2``.  The returned Lagrange
    multipliers correspond to the original rows of ``A`` and use the sign
    convention ``gradient(Psi)(x) + (A * r).T @ multipliers = 0`` at an
    interior optimum.  Multipliers for redundant rows removed internally are
    set to zero.
    """

    if solver_tolerance <= 0:
        raise ValueError("solver_tolerance must be strictly positive")
    if maximum_iterations <= 0:
        raise ValueError("maximum_iterations must be strictly positive")

    r = np.asarray(r, dtype=float)
    if r.ndim != 1:
        raise ValueError("r must be a one-dimensional vector")
    if np.any(r <= 0) or not np.all(np.isfinite(r)):
        raise ValueError(
            "The all-near problem requires finite, strictly positive residual "
            "variances"
        )
    p = r.size

    A = np.asarray(A, dtype=float)
    if A.size == 0:
        A = np.empty((0, p), dtype=float)
    if A.ndim != 2 or A.shape[1] != p:
        raise ValueError(f"A must be a two-dimensional matrix with {p} columns")
    if not np.all(np.isfinite(A)):
        raise ValueError("A must contain only finite values")

    b = np.asarray(b, dtype=float)
    if b.ndim == 0:
        b = b.reshape(1)
    elif b.ndim == 2 and 1 in b.shape:
        b = b.reshape(-1)
    elif b.ndim != 1:
        raise ValueError("b must be a vector or a one-column matrix")
    if b.shape[0] != A.shape[0]:
        raise ValueError("A and b must have the same number of rows")
    if not np.all(np.isfinite(b)):
        raise ValueError("b must contain only finite values")

    feasibility = solve_all_near_feasibility_lp(
        A,
        b,
        r,
        tolerance=solver_tolerance,
    )
    if not feasibility.strict_feasible:
        raise ValueError(
            "The normalized affine constraints have no strictly all-near "
            f"point; LP status is '{feasibility.status}'"
        )

    full_C = A * r
    C = full_C
    equality_rhs = b
    independent_rows = np.arange(C.shape[0])

    # SLSQP is more reliable with redundant equality rows removed.  Since r is
    # strictly positive, C and A have the same row rank.
    if C.shape[0]:
        rank = np.linalg.matrix_rank(C)
        if rank:
            _, _, pivots = qr(C.T, mode="economic", pivoting=True)
            independent_rows = np.sort(pivots[:rank])
            C = C[independent_rows]
            equality_rhs = equality_rhs[independent_rows]
        else:
            C = np.empty((0, p))
            equality_rhs = np.empty(0)

    if C.shape[0] == 0:
        x = np.ones(p)
        return AllNearRootResult(
            residual_variances=r,
            x=x,
            omega=r.copy(),
            lagrange_multipliers=np.zeros(A.shape[0]),
            objective_gap=0.0,
            certified_global=True,
            iterations=0,
            constraint_violation=0.0,
            solver_status=0,
            message="No effective affine constraints; the residual variances apply",
        )

    row_norms = np.linalg.norm(C, axis=1)
    C = C / row_norms[:, None]
    equality_rhs = equality_rhs / row_norms

    def objective(x):
        return float(np.sum(np.log(x) + 1.0 / x - 1.0))

    def gradient(x):
        return 1.0 / x - 1.0 / x**2

    def hessian(x):
        return np.diag((2.0 - x) / x**3)

    equality = LinearConstraint(C, equality_rhs, equality_rhs)
    lower_bound = min(1e-8, solver_tolerance)
    bounds = Bounds(
        np.full(p, lower_bound),
        np.full(p, 2.0),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Values in x were outside bounds during a minimize step",
            category=RuntimeWarning,
            module="scipy.optimize._slsqp_py",
        )
        optimization = minimize(
            objective,
            feasibility.x,
            method="SLSQP",
            jac=gradient,
            bounds=bounds,
            constraints=(equality,),
            options={
                "ftol": solver_tolerance,
                "maxiter": maximum_iterations,
                "disp": False,
            },
        )
    if not optimization.success:
        optimization = minimize(
            objective,
            feasibility.x,
            method="trust-constr",
            jac=gradient,
            hess=hessian,
            bounds=bounds,
            constraints=(equality,),
            options={
                "gtol": solver_tolerance,
                "xtol": solver_tolerance,
                "barrier_tol": solver_tolerance,
                "maxiter": maximum_iterations,
                "verbose": 0,
            },
        )
        if not optimization.success:
            raise RuntimeError(
                "All-near convex optimization failed: "
                f"{optimization.message}"
            )

    x = optimization.x
    constraint_violation = float(
        np.max(np.abs(full_C @ x - b))
    )
    allowed_violation = max(1e-7, 100.0 * solver_tolerance)
    if constraint_violation > allowed_violation:
        raise RuntimeError(
            "All-near optimizer violates the affine constraints by "
            f"{constraint_violation:.3e}"
        )
    if np.any(x <= 0) or np.any(x > 2.0 + allowed_violation):
        raise RuntimeError("All-near optimizer violates its box constraints")

    # Recover multipliers from stationarity on the independent, unnormalized
    # constraint rows. A zero entry is used for every redundant input row,
    # yielding a multiplier vector aligned with the original A and b.
    independent_C = full_C[independent_rows]
    gradient_at_solution = gradient(x)
    reduced_multipliers, _, _, _ = np.linalg.lstsq(
        independent_C.T,
        -gradient_at_solution,
        rcond=None,
    )
    lagrange_multipliers = np.zeros(A.shape[0])
    lagrange_multipliers[independent_rows] = reduced_multipliers

    gap = objective(x)
    return AllNearRootResult(
        residual_variances=r,
        x=x,
        omega=r * x,
        lagrange_multipliers=lagrange_multipliers,
        objective_gap=gap,
        certified_global=gap < np.log(2.0) - 0.5,
        iterations=int(optimization.nit),
        constraint_violation=constraint_violation,
        solver_status=int(optimization.status),
        message=str(optimization.message),
    )


def is_positive_ray(component: ConstraintComponent):
    """Recognize a homogeneous component with one-dimensional kernel.

    Positive feasibility is validated before components reach the constrained
    solver.  Under that assumption, homogeneity and nullity one imply that the
    component's positive feasible cone is a single ray.
    """

    return (
        len(component.columns) - component.rank == 1
        and all(value == 0 for value in component.c)
    )


def compute_constrained_MLE(X, G, A, b):
    problem = prepare_solver_problem(X, G, A, b)
    X, G, A, b = problem.X, problem.G, problem.A, problem.b
    n, p, S = problem.n, problem.p, problem.S

    lambda_hat, r_hat = compute_ols_estimates(X, G)
    #omega_hat = np.zeros(p)

    # Get components of (A,b)
    components: Tuple[ConstraintComponent, ...] = tuple(
        decompose_affine_constraints(A, b)
    )
    unconstrained_cols = unconstrained_columns(p, components)

    for component in components:
        q = len(component.columns)

        if q == component.rank:
            pass

        elif is_positive_ray(component):
            # Apply the closed-form positive-ray solution from Proposition 3.
            pass

        elif component.rank == 1:
            # if componen.R is all positive: Do test: if pass: return all near sol
            # else: do general test: if pass: return all near sol
            # compute one far sols
            pass

        elif q - component.rank == 1:
            pass

        else:
            # do general test: if pass: return all near sol
            # Otherwise, use component size, rank, and feasible dimension to
            # decide whether exhaustive far-pattern enumeration is practical.
            # Never silently fall back to all-near without certification.
            pass
            
        # Solve
    

    for node in unconstrained_cols:
        # omega_hat[node] = r_hat[node]
        pass
    
    #log_lik = utils.get_log_lik(X, lambda_hat, omega_hat)

    return # omega_hat, Lambda_hat, log_lik


def main():
    p = 20
    rng = np.random.default_rng(123)
    r = rng.uniform(0.2, 2.0, size=p)

    # Construct the constraint through a known point in the strict all-near
    # region.  This benchmarks the near-root optimization itself without
    # generating a large DAG, sample, or set of nodewise OLS regressions.
    feasible_x = rng.uniform(0.75, 1.25, size=p)
    feasible_omega = r * feasible_x
    active = rng.random(p) < 0.5
    A = np.zeros((1, p))
    A[0, active] = rng.uniform(-2.0, 2.0, size=np.count_nonzero(active))
    b = A @ feasible_omega

    t = time.perf_counter()
    result = solve_all_near_root(r, A, b)
    print("took", time.perf_counter()-t)

    x_near = result.x
    omega_near = result.omega


if __name__ == "__main__":
    main()
