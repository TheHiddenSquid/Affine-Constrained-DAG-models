import unittest

import numpy as np

from certificate_simulations.simulate_positive_rank_one_certificate import (
    generate_positive_rank_one_constraint,
)


class PositiveConstraintGenerationTests(unittest.TestCase):
    def test_generates_a_positive_constraint_with_two_active_coordinates(self):
        omega = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        rng = np.random.default_rng(7)

        A, b = generate_positive_rank_one_constraint(
            omega=omega,
            coordinate_probability=0.01,
            coefficient_low=0.0,
            coefficient_high=2.0,
            minimum_active_coordinates=2,
            rng=rng,
        )

        self.assertEqual(A.shape, (1, omega.size))
        self.assertGreaterEqual(np.count_nonzero(A), 2)
        self.assertTrue(np.all(A[A != 0] > 0))
        np.testing.assert_allclose(b, A @ omega)

    def test_rejects_a_trivial_minimum_support(self):
        with self.assertRaisesRegex(ValueError, "between 2 and p"):
            generate_positive_rank_one_constraint(
                omega=[1.0, 1.0],
                coordinate_probability=0.5,
                coefficient_low=0.0,
                coefficient_high=2.0,
                minimum_active_coordinates=1,
                rng=np.random.default_rng(1),
            )


if __name__ == "__main__":
    unittest.main()
