"""Benchmark the maintained SymPy/SciPy constraint decomposition."""

import argparse
import random
import statistics
import timeit

import numpy as np

from constraint_components import decompose_affine_constraints


def _hidden_block_system(number_of_variables, rank, number_of_blocks, seed):
    """Create an integer system with known blocks hidden by row operations."""

    if not 1 <= number_of_blocks <= rank <= number_of_variables:
        raise ValueError("Require 1 <= number_of_blocks <= rank <= number_of_variables")

    generator = random.Random(seed)
    row_sizes = [rank // number_of_blocks] * number_of_blocks
    for index in range(rank % number_of_blocks):
        row_sizes[index] += 1
    column_sizes = [number_of_variables // number_of_blocks] * number_of_blocks
    for index in range(number_of_variables % number_of_blocks):
        column_sizes[index] += 1
    if any(columns < rows for columns, rows in zip(column_sizes, row_sizes)):
        raise ValueError("Every block must have at least as many columns as rows")

    reduced = np.zeros((rank, number_of_variables), dtype=np.int64)
    row_start = 0
    column_start = 0
    for block_rows, block_columns in zip(row_sizes, column_sizes):
        for local_row in range(block_rows):
            reduced[row_start + local_row][column_start + local_row] = 1

        if block_columns > block_rows:
            bridge_column = column_start + block_rows
            for local_row in range(block_rows):
                reduced[row_start + local_row][bridge_column] = 1
            for local_column in range(block_rows + 1, block_columns):
                for local_row in range(block_rows):
                    reduced[row_start + local_row][column_start + local_column] = (
                        generator.randint(-2, 2)
                    )

        row_start += block_rows
        column_start += block_columns

    mixing = np.eye(rank, dtype=np.int64)
    for _ in range(3 * rank):
        source, target = generator.sample(range(rank), 2)
        multiplier = generator.choice((-1, 1))
        mixing[target] += multiplier * mixing[source]

    block_rhs = np.array(
        [generator.randint(1, 10) for _ in range(rank)], dtype=np.int64
    )
    return mixing @ reduced, mixing @ block_rhs


def _median_runtime(function, A, b, repeats):
    return statistics.median(
        timeit.repeat(lambda: function(A, b), repeat=repeats, number=1)
    )


def run_benchmark(repeats=5):
    cases = [
        (20, 8, 4),
        (60, 24, 6),
        (120, 40, 8),
    ]
    print("p    rank blocks  median_ms")
    for case_number, (variables, rank, blocks) in enumerate(cases):
        A, b = _hidden_block_system(variables, rank, blocks, seed=9000 + case_number)
        result = decompose_affine_constraints(A, b)
        if len(result) != blocks:
            raise AssertionError("The generated block structure was not recovered")
        runtime = _median_runtime(decompose_affine_constraints, A, b, repeats)
        print(
            f"{variables:<4} {rank:<4} {blocks:<7} "
            f"{1000 * runtime:>9.3f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    arguments = parser.parse_args()
    run_benchmark(repeats=arguments.repeats)


if __name__ == "__main__":
    main()
