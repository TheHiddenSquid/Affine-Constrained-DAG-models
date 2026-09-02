import random
import unittest

import numpy as np
import sympy as sp

from constraint_components import (
    decompose_affine_constraints,
    unconstrained_columns,
)


class ConstraintComponentTests(unittest.TestCase):
    def test_visible_disjoint_sums(self):
        components = decompose_affine_constraints(
            [[1, 1, 0, 0], [0, 0, 1, 1]], [2, 4]
        )

        self.assertEqual([item.columns for item in components], [(0, 1), (2, 3)])
        self.assertEqual(components[0].R, sp.ImmutableMatrix([[1, 1]]))
        self.assertEqual(components[0].c, sp.ImmutableMatrix([2]))
        self.assertEqual(components[1].c, sp.ImmutableMatrix([4]))

    def test_hidden_blocks_are_revealed_by_rref(self):
        components = decompose_affine_constraints(
            [[1, 1, 0, 0], [1, 1, 1, 1]], [2, 6]
        )

        self.assertEqual([item.columns for item in components], [(0, 1), (2, 3)])
        self.assertEqual(
            [item.c for item in components],
            [sp.ImmutableMatrix([2]), sp.ImmutableMatrix([4])],
        )

    def test_connected_circuit_is_not_split(self):
        components = decompose_affine_constraints(
            [[1, 0, 1], [0, 1, 1]], [3, 4]
        )

        self.assertEqual(len(components), 1)
        self.assertEqual(components[0].columns, (0, 1, 2))
        self.assertEqual(components[0].rank, 2)

    def test_singletons_and_unconstrained_column(self):
        components = decompose_affine_constraints(
            np.array([[1, 0, 0], [0, 1, 0]]), np.array([2, 3])
        )

        self.assertEqual([item.columns for item in components], [(0,), (1,)])
        self.assertEqual(unconstrained_columns(3, components), (2,))

    def test_redundant_consistent_row_is_removed(self):
        components = decompose_affine_constraints([[1, 1], [2, 2]], [3, 6])

        self.assertEqual(len(components), 1)
        self.assertEqual(components[0].R, sp.ImmutableMatrix([[1, 1]]))
        self.assertEqual(components[0].c, sp.ImmutableMatrix([3]))

    def test_inconsistent_system_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            decompose_affine_constraints([[1, 1], [2, 2]], [3, 7])

    def test_random_feasible_integer_systems(self):
        generator = random.Random(20260827)
        for _ in range(25):
            rows = generator.randint(1, 7)
            columns = generator.randint(rows, rows + 8)
            A = [
                [generator.randint(-3, 3) for _ in range(columns)]
                for _ in range(rows)
            ]
            omega = [generator.randint(1, 5) for _ in range(columns)]
            b = [sum(A[i][j] * omega[j] for j in range(columns)) for i in range(rows)]
            components = decompose_affine_constraints(A, b)

            covered = [column for item in components for column in item.columns]
            self.assertEqual(len(covered), len(set(covered)))
            for item in components:
                local_omega = sp.ImmutableMatrix([omega[j] for j in item.columns])
                self.assertEqual(item.R * local_omega, item.c)


if __name__ == "__main__":
    unittest.main()
