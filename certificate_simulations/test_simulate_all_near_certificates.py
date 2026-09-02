import unittest

import numpy as np

from certificate_simulations.simulate_all_near_certificates import (
    C_0,
    feasible_reference_gap,
    likelihood_gap,
    positive_rank_one_certificate,
    projection_statistic,
    proposition_4_certificate,
)
from solver import solve_all_near_feasibility_lp


class Proposition4CertificateTests(unittest.TestCase):
    def test_accepts_when_the_discrepancy_is_below_the_active_minimum(self):
        self.assertTrue(
            proposition_4_certificate([[1, -2]], [-1], [1.1, 0.9])
        )

    def test_rejects_when_the_discrepancy_is_too_large(self):
        self.assertFalse(proposition_4_certificate([[1]], [2], [0.5]))

    def test_rejects_more_than_one_constraint_row(self):
        with self.assertRaisesRegex(ValueError, "exactly one constraint"):
            proposition_4_certificate(np.eye(2), [1, 1], [1, 1])


class PositiveRankOneCertificateTests(unittest.TestCase):
    def test_uses_the_sqrt_s_over_sqrt_s_plus_one_threshold(self):
        # For s=2 the threshold is approximately 0.5858.  This example also
        # witnesses strict improvement over Proposition 4.
        A = np.array([[1.0, 1.0]])
        b = np.array([1.0])
        r = np.array([0.315, 0.315])

        self.assertTrue(positive_rank_one_certificate(A, b, r))
        self.assertFalse(proposition_4_certificate(A, b, r))

    def test_rejects_below_the_positive_threshold(self):
        self.assertFalse(
            positive_rank_one_certificate([[1.0, 1.0]], [1.0], [0.2, 0.2])
        )

    def test_accepts_an_all_negative_row_after_sign_normalization(self):
        self.assertTrue(
            positive_rank_one_certificate([[-1.0, -1.0]], [-1.0], [0.4, 0.4])
        )

    def test_rejects_mixed_signs(self):
        with self.assertRaisesRegex(ValueError, "same sign"):
            positive_rank_one_certificate([[1.0, -1.0]], [1.0], [1.0, 1.0])


class ProjectionStatisticTests(unittest.TestCase):
    def test_is_zero_when_residual_variances_satisfy_the_constraint(self):
        A = np.array([[1.0, -2.0]])
        r = np.array([2.0, 0.5])
        b = A @ r

        self.assertAlmostEqual(projection_statistic(A, b, r), 0.0)


class FeasibleReferenceGapTests(unittest.TestCase):
    def test_accepts_an_unconstrained_variance_point(self):
        A = np.array([[1.0]])
        b = np.array([1.0])
        r = np.array([1.0])
        lp_result = solve_all_near_feasibility_lp(A, b, r)

        self.assertLess(feasible_reference_gap(A, b, r, lp_result), C_0)

    def test_uses_projection_when_the_lp_point_has_too_large_a_gap(self):
        A = np.array([[1.6650341074, 0.0063458350, -0.4378336764]])
        b = np.array([0.5366836913])
        r = np.ones(3)
        lp_result = solve_all_near_feasibility_lp(A, b, r)

        self.assertGreater(likelihood_gap(lp_result.x), C_0)
        self.assertLess(feasible_reference_gap(A, b, r, lp_result), C_0)

    def test_rejects_when_no_reference_has_gap_below_the_barrier(self):
        A = np.array([[1.0]])
        b = np.array([3.0])
        r = np.array([1.0])
        lp_result = solve_all_near_feasibility_lp(A, b, r)

        self.assertGreaterEqual(feasible_reference_gap(A, b, r, lp_result), C_0)


if __name__ == "__main__":
    unittest.main()
