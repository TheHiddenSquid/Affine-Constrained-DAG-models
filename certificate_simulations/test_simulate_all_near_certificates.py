import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Keep the test runnable both from the solver root and from this subdirectory.
SOLVER_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SOLVER_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOLVER_DIRECTORY))

from certificates import (
    C_0,
    S_0,
    nonnegative_dual_certificate,
    positive_rank_one_certificate,
    projection_certificate,
    projection_likelihood_gap,
    projection_statistic,
    rank_one_discrepancy_certificate,
    s_certificate,
)

if __package__:
    from .simulate_all_near_certificates import (
        save_results_csv,
        SimulationResult,
    )
else:
    from simulate_all_near_certificates import (
        save_results_csv,
        SimulationResult,
    )


class RankOneDiscrepancyCertificateTests(unittest.TestCase):
    def test_accepts_when_the_discrepancy_is_below_the_active_minimum(self):
        self.assertTrue(
            rank_one_discrepancy_certificate([[1, -2]], [-1], [1.1, 0.9])
        )

    def test_rejects_when_the_discrepancy_is_too_large(self):
        self.assertFalse(rank_one_discrepancy_certificate([[1]], [2], [0.5]))

    def test_rejects_more_than_one_constraint_row(self):
        with self.assertRaisesRegex(ValueError, "exactly one constraint"):
            rank_one_discrepancy_certificate(np.eye(2), [1, 1], [1, 1])


class NonnegativeDualCertificateTests(unittest.TestCase):
    def test_accepts_a_coordinatewise_nonnegative_dual_image(self):
        A = np.array([[1.0, 0.0], [0.0, 2.0]])

        self.assertTrue(nonnegative_dual_certificate(A, [0.5, 0.25]))

    def test_rejects_a_negative_coordinate_in_the_dual_image(self):
        self.assertFalse(nonnegative_dual_certificate([[1.0, -1.0]], [0.5]))

    def test_allows_small_numerical_error(self):
        self.assertTrue(
            nonnegative_dual_certificate([[1.0]], [-1e-10], tolerance=1e-9)
        )

    def test_rejects_an_incompatible_multiplier_vector(self):
        with self.assertRaisesRegex(ValueError, "same number of rows"):
            nonnegative_dual_certificate(np.eye(2), [1.0])


class PositiveRankOneCertificateTests(unittest.TestCase):
    def test_uses_the_sqrt_s_over_sqrt_s_plus_one_threshold(self):
        # For s=2 the threshold is approximately 0.5858.  This example also
        # witnesses strict improvement over the discrepancy certificate.
        A = np.array([[1.0, 1.0]])
        b = np.array([1.0])
        r = np.array([0.315, 0.315])

        self.assertTrue(positive_rank_one_certificate(A, b, r))
        self.assertFalse(rank_one_discrepancy_certificate(A, b, r))

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

    def test_s_certificate_uses_the_sharp_threshold(self):
        A = np.array([[1.0]])
        r = np.array([1.0])

        self.assertAlmostEqual(projection_statistic(A, [0.9], r), 0.01)
        self.assertLess(projection_statistic(A, [0.9], r), S_0)
        self.assertTrue(s_certificate(A, [0.9], r))
        self.assertFalse(s_certificate(A, [2.0], r))


class ProjectionCertificateTests(unittest.TestCase):
    def test_accepts_when_the_projection_is_the_unconstrained_point(self):
        A = np.array([[1.0]])
        b = np.array([1.0])
        r = np.array([1.0])

        self.assertAlmostEqual(projection_likelihood_gap(A, b, r), 0.0)
        self.assertTrue(projection_certificate(A, b, r))

    def test_rejects_when_the_projection_gap_exceeds_the_barrier(self):
        A = np.array([[1.0]])
        b = np.array([3.0])
        r = np.array([1.0])

        self.assertGreaterEqual(projection_likelihood_gap(A, b, r), C_0)
        self.assertFalse(projection_certificate(A, b, r))

    def test_rejects_a_nonpositive_projection(self):
        A = np.array([[1.0]])
        b = np.array([0.0])
        r = np.array([1.0])

        self.assertEqual(projection_likelihood_gap(A, b, r), np.inf)
        self.assertFalse(projection_certificate(A, b, r))


class ResultsCsvTests(unittest.TestCase):
    @staticmethod
    def result(mode):
        return SimulationResult(
            fitted_graph=mode,
            p=5,
            n=25,
            trials=10,
            ols_successes=10,
            strict_lp_successes=9,
            tn_certificate_successes=8,
            rank_one_discrepancy_successes=7,
            either_certificate_successes=9,
            certificate_disagreements=1,
        )

    def test_one_csv_contains_fitted_and_unfitted_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "results.csv"
            save_results_csv(
                {
                    "true": [self.result("true")],
                    "random": [self.result("random")],
                },
                output_path,
            )

            with output_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual([row["fitted_graph"] for row in rows], ["true", "random"])
        self.assertNotIn("old_reference_gap_successes", rows[0])


if __name__ == "__main__":
    unittest.main()
