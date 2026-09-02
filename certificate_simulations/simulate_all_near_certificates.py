"""Monte Carlo study of the Section 6.1 all-near feasibility LP.

For each trial, this script

1. draws a Gaussian DAG model using ``utils``;
2. draws affine constraints whose right-hand side is chosen so that the true
   error variances satisfy them;
3. generates one sample and uses nested prefixes for the requested sample
   sizes;
4. computes the DAG-wise OLS residual variances;
5. runs the strict all-near feasibility LP; and
6. solves the convex all-near problem and evaluates ``T_N < c_0``.

Run the default experiment with

    python3 -m certificate_simulations.simulate_all_near_certificates

Use ``--help`` to see the configurable simulation parameters.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

# These experiments live one directory below the solver modules.  Add that
# directory explicitly so the file works both as a package module and as a
# directly executed script.
SOLVER_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SOLVER_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOLVER_DIRECTORY))

import utils
from solver import (
    compute_ols_estimates,
    solve_all_near_feasibility_lp,
    solve_all_near_root,
)


C_0 = np.log(2.0) - 0.5
S_0 = 0.1854712448


@dataclass(frozen=True)
class SimulationConfig:
    p_values: tuple[int, ...] = (5, 20)
    sample_sizes: tuple[int, ...] = (25, 40, 60, 100, 160, 250, 400, 650, 1000)
    trials: int = 500
    number_of_constraints: int = 1
    coordinate_probability: float = 0.25
    coefficient_low: float = -2.0
    coefficient_high: float = 2.0
    edge_probability: float = 0.5
    fitted_graph: str = "true"
    seed: int = 20260902


@dataclass(frozen=True)
class SimulationResult:
    fitted_graph: str
    p: int
    n: int
    trials: int
    ols_successes: int
    strict_lp_successes: int
    tn_certificate_successes: int
    proposition_4_successes: Optional[int]
    either_certificate_successes: int
    certificate_disagreements: Optional[int]

    @property
    def strict_lp_rate(self):
        return self.strict_lp_successes / self.trials

    @property
    def tn_certificate_rate(self):
        return self.tn_certificate_successes / self.trials

    @property
    def proposition_4_rate(self):
        if self.proposition_4_successes is None:
            return np.nan
        return self.proposition_4_successes / self.trials

    @property
    def either_certificate_rate(self):
        return self.either_certificate_successes / self.trials

    @property
    def certificate_disagreement_rate(self):
        if self.certificate_disagreements is None:
            return np.nan
        return self.certificate_disagreements / self.trials


def generate_true_affine_constraints(
    omega: Sequence[float],
    number_of_constraints: int,
    coordinate_probability: float,
    coefficient_low: float,
    coefficient_high: float,
    rng: np.random.Generator,
):
    """Draw a full-row-rank ``A`` and set ``b = A @ omega``.

    Each coordinate is included independently with the requested probability.
    Empty rows are repaired by selecting one coordinate uniformly.  Rows that
    do not increase the rank are redrawn so that the matrix-only statistic is
    well-defined without a pseudoinverse.
    """

    omega = np.asarray(omega, dtype=float)
    p = omega.size
    if not 1 <= number_of_constraints <= p:
        raise ValueError("number_of_constraints must lie between 1 and p")
    if not 0 < coordinate_probability <= 1:
        raise ValueError("coordinate_probability must lie in (0, 1]")
    if coefficient_low >= coefficient_high:
        raise ValueError("coefficient_low must be smaller than coefficient_high")

    rows = []
    for row_number in range(number_of_constraints):
        for _ in range(10_000):
            selected = rng.random(p) < coordinate_probability
            if not np.any(selected):
                selected[rng.integers(p)] = True

            row = np.zeros(p)
            row[selected] = rng.uniform(
                coefficient_low, coefficient_high, size=np.count_nonzero(selected)
            )
            candidate = np.vstack((*rows, row)) if rows else row[None, :]
            if np.linalg.matrix_rank(candidate) == row_number + 1:
                rows.append(row)
                break
        else:
            raise RuntimeError("Could not draw linearly independent constraints")

    A = np.vstack(rows)
    b = A @ omega
    return A, b


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


def proposition_4_certificate(A, b, r):
    """Evaluate Proposition 4 for a single rank-one constraint.

    The condition is

        abs(a.T @ r - b) < min(abs(a_i) * r_i for a_i != 0).

    Proposition 4 does not directly apply to a constraint matrix with more
    than one independent row, so this function rejects such input.
    """

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    r = np.asarray(r, dtype=float)
    if A.ndim != 2 or A.shape[0] != 1:
        raise ValueError("Proposition 4 requires exactly one constraint row")
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


def wilson_interval(successes, trials, z=1.959963984540054):
    """Return a two-sided Wilson binomial confidence interval."""

    probability = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (probability + z**2 / (2.0 * trials)) / denominator
    half_width = (
        z
        * np.sqrt(
            probability * (1.0 - probability) / trials
            + z**2 / (4.0 * trials**2)
        )
        / denominator
    )
    return center - half_width, center + half_width


def run_simulation(config: SimulationConfig):
    """Run the Monte Carlo experiment and return one row per ``(p, n)``."""

    if config.fitted_graph not in {"true", "random"}:
        raise ValueError("fitted_graph must be either 'true' or 'random'")
    random.seed(config.seed)
    np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    sample_sizes = tuple(sorted(set(config.sample_sizes)))
    maximum_sample_size = max(sample_sizes)
    results = []

    for p in config.p_values:
        counts = {
            n: {
                "ols": 0,
                "lp": 0,
                "tn": 0,
                "prop4": 0,
                "either": 0,
                "disagree": 0,
            }
            for n in sample_sizes
        }

        for _ in range(config.trials):
            G, lambda_true, omega_true = utils.generate_DAG_with_params(
                p, config.edge_probability
            )
            G = np.asarray(G)
            if config.fitted_graph == "random":
                for _ in range(10_000):
                    fitted_G = np.asarray(
                        utils.get_random_DAG(p, config.edge_probability)
                    )
                    if not np.array_equal(fitted_G, G):
                        break
                else:
                    raise RuntimeError("Could not draw a DAG distinct from the true DAG")
            else:
                fitted_G = G
            omega_true = np.asarray(omega_true, dtype=float)
            A, b = generate_true_affine_constraints(
                omega_true,
                config.number_of_constraints,
                config.coordinate_probability,
                config.coefficient_low,
                config.coefficient_high,
                rng,
            )

            # ``utils.generate_sample`` returns variables by observations.
            full_sample = utils.generate_sample(
                maximum_sample_size, lambda_true, omega_true
            ).T

            for n in sample_sizes:
                X = full_sample[:n].copy()
                X -= X.mean(axis=0, keepdims=True)
                try:
                    _, r = compute_ols_estimates(X, fitted_G)
                except ValueError:
                    # A non-unique OLS fit means neither test is available.
                    continue

                if np.any(r <= 0):
                    continue
                counts[n]["ols"] += 1

                lp_result = solve_all_near_feasibility_lp(A, b, r)
                if lp_result.strict_feasible:
                    counts[n]["lp"] += 1

                tn_passes = False
                if lp_result.strict_feasible:
                    tn_result = solve_all_near_root(r, A, b)
                    tn_passes = tn_result.certified_global
                if tn_passes:
                    counts[n]["tn"] += 1

                # Proposition 4 applies directly only to the rank-one case.
                if config.number_of_constraints == 1:
                    proposition_4_passes = proposition_4_certificate(A, b, r)
                    if proposition_4_passes:
                        counts[n]["prop4"] += 1
                    if tn_passes or proposition_4_passes:
                        counts[n]["either"] += 1
                    if tn_passes != proposition_4_passes:
                        counts[n]["disagree"] += 1
                elif tn_passes:
                    # The union reduces to the optimized all-near test when
                    # Proposition 4 is not applicable.
                    counts[n]["either"] += 1

        for n in sample_sizes:
            results.append(
                SimulationResult(
                    fitted_graph=config.fitted_graph,
                    p=p,
                    n=n,
                    trials=config.trials,
                    ols_successes=counts[n]["ols"],
                    strict_lp_successes=counts[n]["lp"],
                    tn_certificate_successes=counts[n]["tn"],
                    proposition_4_successes=(
                        counts[n]["prop4"]
                        if config.number_of_constraints == 1
                        else None
                    ),
                    either_certificate_successes=counts[n]["either"],
                    certificate_disagreements=(
                        counts[n]["disagree"]
                        if config.number_of_constraints == 1
                        else None
                    ),
                )
            )

    return results


def save_results_csv(results_by_mode, output_path):
    """Save all fitted-graph modes in one machine-readable table."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "fitted_graph",
                "p",
                "n",
                "trials",
                "ols_successes",
                "strict_lp_successes",
                "strict_lp_percent",
                "tn_certificate_successes",
                "tn_certificate_percent",
                "proposition_4_successes",
                "proposition_4_percent",
                "either_certificate_successes",
                "either_certificate_percent",
                "certificate_disagreements",
                "certificate_disagreement_percent",
            )
        )
        for mode in ("true", "random"):
            for result in results_by_mode.get(mode, ()):
                writer.writerow(
                    (
                        result.fitted_graph,
                        result.p,
                        result.n,
                        result.trials,
                        result.ols_successes,
                        result.strict_lp_successes,
                        f"{100.0 * result.strict_lp_rate:.6f}",
                        result.tn_certificate_successes,
                        f"{100.0 * result.tn_certificate_rate:.6f}",
                        result.proposition_4_successes,
                        f"{100.0 * result.proposition_4_rate:.6f}",
                        result.either_certificate_successes,
                        f"{100.0 * result.either_certificate_rate:.6f}",
                        result.certificate_disagreements,
                        f"{100.0 * result.certificate_disagreement_rate:.6f}",
                    )
                )


def plot_results(results_by_mode, output_path):
    """Render true- and random-DAG mixed-sign results in one figure."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    modes = [mode for mode in ("true", "random") if mode in results_by_mode]
    p_values = sorted(
        {result.p for mode in modes for result in results_by_mode[mode]}
    )
    figure, axes = plt.subplots(
        len(modes),
        len(p_values),
        figsize=(6.2 * len(p_values), 4.3 * len(modes)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    series = (
        ("Strict LP feasibility", "#2878B5", "o", "strict_lp_successes", "-"),
        (
            r"Optimized $T_N<c_0$",
            "#D95319",
            "s",
            "tn_certificate_successes",
            "--",
        ),
        (
            "Proposition 4",
            "#77AC30",
            "^",
            "proposition_4_successes",
            ":",
        ),
        (
            "Either certificate",
            "#7E2F8E",
            "D",
            "either_certificate_successes",
            "-",
        ),
    )

    for row_index, mode in enumerate(modes):
        for column_index, p in enumerate(p_values):
            axis = axes[row_index, column_index]
            rows = sorted(
                (result for result in results_by_mode[mode] if result.p == p),
                key=lambda result: result.n,
            )
            sample_sizes = np.array([result.n for result in rows])
            for label, color, marker, field, line_style in series:
                if getattr(rows[0], field) is None:
                    continue
                counts = np.array([getattr(result, field) for result in rows])
                rates = 100.0 * counts / rows[0].trials
                intervals = np.array(
                    [wilson_interval(count, rows[0].trials) for count in counts]
                )
                axis.plot(
                    sample_sizes,
                    rates,
                    color=color,
                    marker=marker,
                    linestyle=line_style,
                    label=label,
                )
                axis.fill_between(
                    sample_sizes,
                    100.0 * intervals[:, 0],
                    100.0 * intervals[:, 1],
                    color=color,
                    alpha=0.08,
                    linewidth=0,
                )

            mode_label = "True fitted DAG" if mode == "true" else "Random fitted DAG"
            axis.set_title(f"{mode_label}, p = {p}")
            axis.set_xscale("log")
            axis.set_xticks(sample_sizes)
            axis.set_xticklabels([str(n) for n in sample_sizes], rotation=40)
            axis.set_ylim(-2, 102)
            axis.grid(alpha=0.25)
            axis.set_xlabel("Sample size n")
            if column_index == 0:
                axis.set_ylabel("Successful trials (%)")
            legend_location = "center right" if mode == "random" else "lower right"
            axis.legend(loc=legend_location)

    figure.suptitle("Mixed-sign all-near feasibility and optimality certificates")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--constraints", type=int, default=1)
    parser.add_argument("--coordinate-probability", type=float, default=0.25)
    parser.add_argument("--edge-probability", type=float, default=0.5)
    parser.add_argument(
        "--fitted-graph",
        choices=("true", "random", "both"),
        default="both",
        help="Fit the data-generating DAG, an independent DAG, or both",
    )
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("simulation_results"),
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    base_config = SimulationConfig(
        trials=arguments.trials,
        number_of_constraints=arguments.constraints,
        coordinate_probability=arguments.coordinate_probability,
        edge_probability=arguments.edge_probability,
        fitted_graph="true",
        seed=arguments.seed,
    )
    modes = (
        ("true", "random")
        if arguments.fitted_graph == "both"
        else (arguments.fitted_graph,)
    )
    results_by_mode = {}
    for mode in modes:
        config = replace(base_config, fitted_graph=mode)
        results = run_simulation(config)
        results_by_mode[mode] = results

        for result in results:
            print(
                f"graph={mode:>6}, p={result.p:>2}, n={result.n:>4}: "
                f"LP={100 * result.strict_lp_rate:6.2f}%, "
                f"T_N<c0={100 * result.tn_certificate_rate:6.2f}%, "
                f"Prop4={100 * result.proposition_4_rate:6.2f}%, "
                f"either={100 * result.either_certificate_rate:6.2f}%, "
                f"disagree="
                f"{100 * result.certificate_disagreement_rate:6.2f}%"
            )

    suffix = "_both_graphs" if len(modes) == 2 else f"_{modes[0]}_dag"
    save_results_csv(
        results_by_mode,
        arguments.output_dir
        / f"all_near_certificate_comparison{suffix}.csv",
    )
    plot_results(
        results_by_mode,
        arguments.output_dir
        / f"all_near_certificate_comparison{suffix}.png",
    )


if __name__ == "__main__":
    main()
