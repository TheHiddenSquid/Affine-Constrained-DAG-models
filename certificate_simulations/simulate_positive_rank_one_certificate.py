"""Monte Carlo study of the positive rank-one certificate.

The experiment draws one constraint ``a.T @ omega = b`` per data-generating
model.  Every active coefficient is positive and every row has at least two
active coordinates.  It compares

* strict all-near LP feasibility;
* the optimized likelihood-gap certificate ``T_N < c_0``;
* the positive rank-one certificate; and
* the logical union of the two optimality certificates.

Proposition 4 is retained in the CSV output as a diagnostic.  Its success set
must be contained in that of the positive certificate.

Run the default experiment from the solver directory with

    python3 -m certificate_simulations.simulate_positive_rank_one_certificate
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import utils
from .simulate_all_near_certificates import (
    positive_rank_one_certificate,
    proposition_4_certificate,
    wilson_interval,
)
from solver import (
    compute_ols_estimates,
    solve_all_near_feasibility_lp,
    solve_all_near_root,
)


@dataclass(frozen=True)
class PositiveSimulationConfig:
    p_values: tuple[int, ...] = (5, 20)
    sample_sizes: tuple[int, ...] = (25, 40, 60, 100, 160, 250, 400, 650, 1000)
    trials: int = 500
    coordinate_probability: float = 0.25
    coefficient_low: float = 0.0
    coefficient_high: float = 2.0
    minimum_active_coordinates: int = 2
    edge_probability: float = 0.5
    fitted_graph: str = "true"
    seed: int = 20260902


@dataclass(frozen=True)
class PositiveSimulationResult:
    fitted_graph: str
    p: int
    n: int
    trials: int
    ols_successes: int
    strict_lp_successes: int
    tn_certificate_successes: int
    positive_certificate_successes: int
    either_certificate_successes: int
    proposition_4_successes: int
    dominance_violations: int

    def rate(self, attribute):
        return getattr(self, attribute) / self.trials


def generate_positive_rank_one_constraint(
    omega,
    coordinate_probability,
    coefficient_low,
    coefficient_high,
    minimum_active_coordinates,
    rng,
):
    """Draw one positive rank-one constraint satisfied by ``omega``.

    Coordinates are first selected independently.  If fewer than the requested
    minimum are selected, uniformly chosen inactive coordinates are added.  In
    particular, the default keeps the original Bernoulli design while avoiding
    the trivial one-coordinate case.
    """

    omega = np.asarray(omega, dtype=float)
    if omega.ndim != 1 or np.any(omega <= 0):
        raise ValueError("omega must be a strictly positive vector")
    p = omega.size
    if not 0 < coordinate_probability <= 1:
        raise ValueError("coordinate_probability must lie in (0, 1]")
    if not 2 <= minimum_active_coordinates <= p:
        raise ValueError("minimum_active_coordinates must lie between 2 and p")
    if coefficient_low < 0 or coefficient_low >= coefficient_high:
        raise ValueError(
            "Positive coefficients require 0 <= coefficient_low < coefficient_high"
        )

    selected = rng.random(p) < coordinate_probability
    missing = minimum_active_coordinates - int(np.count_nonzero(selected))
    if missing > 0:
        inactive = np.flatnonzero(~selected)
        selected[rng.choice(inactive, size=missing, replace=False)] = True

    row = np.zeros(p)
    number_selected = int(np.count_nonzero(selected))
    coefficients = rng.uniform(
        coefficient_low, coefficient_high, size=number_selected
    )
    # A continuous U(0, high) draw is positive almost surely.  Guard against
    # an exact zero from a custom random-number implementation or rounding.
    if np.any(coefficients <= 0):
        coefficients = np.maximum(coefficients, np.nextafter(0.0, 1.0))
    row[selected] = coefficients
    A = row[None, :]
    b = A @ omega
    return A, b


def run_positive_simulation(config: PositiveSimulationConfig):
    """Run the positive-coefficient experiment."""

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
                "positive": 0,
                "either": 0,
                "prop4": 0,
                "dominance_violations": 0,
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
            A, b = generate_positive_rank_one_constraint(
                omega_true,
                config.coordinate_probability,
                config.coefficient_low,
                config.coefficient_high,
                config.minimum_active_coordinates,
                rng,
            )
            full_sample = utils.generate_sample(
                maximum_sample_size, lambda_true, omega_true
            ).T

            for n in sample_sizes:
                X = full_sample[:n].copy()
                X -= X.mean(axis=0, keepdims=True)
                try:
                    _, r = compute_ols_estimates(X, fitted_G)
                except ValueError:
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
                positive_passes = positive_rank_one_certificate(A, b, r)
                proposition_4_passes = proposition_4_certificate(A, b, r)

                if tn_passes:
                    counts[n]["tn"] += 1
                if positive_passes:
                    counts[n]["positive"] += 1
                if tn_passes or positive_passes:
                    counts[n]["either"] += 1
                if proposition_4_passes:
                    counts[n]["prop4"] += 1
                if proposition_4_passes and not positive_passes:
                    counts[n]["dominance_violations"] += 1

        for n in sample_sizes:
            row = counts[n]
            results.append(
                PositiveSimulationResult(
                    fitted_graph=config.fitted_graph,
                    p=p,
                    n=n,
                    trials=config.trials,
                    ols_successes=row["ols"],
                    strict_lp_successes=row["lp"],
                    tn_certificate_successes=row["tn"],
                    positive_certificate_successes=row["positive"],
                    either_certificate_successes=row["either"],
                    proposition_4_successes=row["prop4"],
                    dominance_violations=row["dominance_violations"],
                )
            )

    return results


def save_positive_results_csv(results_by_mode, output_path):
    """Save counts and rates for both fitted-graph modes."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "ols_successes",
        "strict_lp_successes",
        "tn_certificate_successes",
        "positive_certificate_successes",
        "either_certificate_successes",
        "proposition_4_successes",
        "dominance_violations",
    )
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        header = ["fitted_graph", "p", "n", "trials"]
        for field in fields:
            rate_field = (
                field.replace("_successes", "_percent")
                if field.endswith("_successes")
                else f"{field}_percent"
            )
            header.extend((field, rate_field))
        writer.writerow(header)
        for mode in ("true", "random"):
            for result in results_by_mode.get(mode, ()):
                row = [mode, result.p, result.n, result.trials]
                for field in fields:
                    count = getattr(result, field)
                    row.extend((count, f"{100.0 * count / result.trials:.6f}"))
                writer.writerow(row)


def plot_positive_results(results_by_mode, output_path):
    """Plot the positive-coefficient certificate comparison."""

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
        (r"Optimized $T_N<c_0$", "#D95319", "s", "tn_certificate_successes", "--"),
        (
            "Positive rank-one",
            "#77AC30",
            "^",
            "positive_certificate_successes",
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
            axis.legend(loc="lower right" if mode == "true" else "center right")

    figure.suptitle("Positive rank-one all-near certificates")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--coordinate-probability", type=float, default=0.25)
    parser.add_argument("--coefficient-low", type=float, default=0.0)
    parser.add_argument("--coefficient-high", type=float, default=2.0)
    parser.add_argument("--minimum-active", type=int, default=2)
    parser.add_argument("--edge-probability", type=float, default=0.5)
    parser.add_argument(
        "--fitted-graph", choices=("true", "random", "both"), default="both"
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
    base_config = PositiveSimulationConfig(
        trials=arguments.trials,
        coordinate_probability=arguments.coordinate_probability,
        coefficient_low=arguments.coefficient_low,
        coefficient_high=arguments.coefficient_high,
        minimum_active_coordinates=arguments.minimum_active,
        edge_probability=arguments.edge_probability,
        seed=arguments.seed,
    )
    modes = (
        ("true", "random")
        if arguments.fitted_graph == "both"
        else (arguments.fitted_graph,)
    )
    results_by_mode = {}
    for mode in modes:
        results = run_positive_simulation(replace(base_config, fitted_graph=mode))
        results_by_mode[mode] = results
        for result in results:
            print(
                f"graph={mode:>6}, p={result.p:>2}, n={result.n:>4}: "
                f"LP={100 * result.rate('strict_lp_successes'):6.2f}%, "
                f"T_N<c0={100 * result.rate('tn_certificate_successes'):6.2f}%, "
                f"positive={100 * result.rate('positive_certificate_successes'):6.2f}%, "
                f"either={100 * result.rate('either_certificate_successes'):6.2f}%, "
                f"Prop4={100 * result.rate('proposition_4_successes'):6.2f}%, "
                f"dominance violations={result.dominance_violations}"
            )

    suffix = "_both_graphs" if len(modes) == 2 else f"_{modes[0]}_dag"
    save_positive_results_csv(
        results_by_mode,
        arguments.output_dir / f"positive_rank_one_certificates{suffix}.csv",
    )
    plot_positive_results(
        results_by_mode,
        arguments.output_dir / f"positive_rank_one_certificates{suffix}.png",
    )


if __name__ == "__main__":
    main()
