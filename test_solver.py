import unittest

import numpy as np

from solver import (
    AllNearFeasibilityResult,
    AllNearRootResult,
    ComponentSolution,
    ConstrainedMLEResult,
    compute_ols_estimates,
    prepare_solver_problem,
    solve_all_near_feasibility_lp,
    solve_all_near_root,
)


class PrepareSolverProblemTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array(
            [
                [-1.0, -2.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        self.G = np.array([[0, 1], [0, 0]])

    def test_extracts_dimensions_and_sample_covariance(self):
        problem = prepare_solver_problem(
            self.X, self.G, [[1, 1]], [2]
        )

        self.assertEqual((problem.n, problem.p), (4, 2))
        np.testing.assert_allclose(problem.S, self.X.T @ self.X / 4)
        self.assertEqual(problem.A.shape, (1, 2))
        self.assertEqual(problem.b.shape, (1,))

    def test_accepts_no_constraints(self):
        problem = prepare_solver_problem(self.X, self.G, [], [])
        self.assertEqual(problem.A.shape, (0, 2))

    def test_requires_one_constraint_column_per_node(self):
        with self.assertRaisesRegex(ValueError, "2 columns"):
            prepare_solver_problem(self.X, self.G, [[1]], [2])

    def test_rejects_wrong_graph_shape(self):
        with self.assertRaisesRegex(ValueError, "G must have shape"):
            prepare_solver_problem(self.X, np.zeros((3, 3)), [[1, 1]], [2])

    def test_rejects_a_directed_cycle(self):
        cyclic_graph = np.array([[0, 1], [1, 0]])
        with self.assertRaisesRegex(ValueError, "acyclic"):
            prepare_solver_problem(self.X, cyclic_graph, [[1, 1]], [2])

    def test_rejects_inconsistent_equalities(self):
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            prepare_solver_problem(
                self.X,
                self.G,
                [[1, 0], [1, 0]],
                [1, 2],
            )

    def test_rejects_a_constraint_without_a_positive_solution(self):
        with self.assertRaisesRegex(ValueError, "positively feasible"):
            prepare_solver_problem(self.X, self.G, [[1, 1]], [-1])

    def test_accepts_homogeneous_constraints_with_positive_solutions(self):
        problem = prepare_solver_problem(self.X, self.G, [[1, -1]], [0])
        self.assertEqual(problem.A.shape, (1, 2))


class ResultDataclassTests(unittest.TestCase):
    def test_result_dataclasses_store_certification_metadata(self):
        component = ComponentSolution(
            columns=(0, 1),
            omega=np.array([1.0, 1.0]),
            objective=2.0,
            method="partial-homoscedasticity",
            certified_global=True,
            candidates_checked=1,
        )
        result = ConstrainedMLEResult(
            lambda_hat=np.zeros((2, 2)),
            omega_hat=np.ones(2),
            log_likelihood=-3.0,
            certified_global=True,
            components=(component,),
        )

        self.assertTrue(result.certified_global)
        self.assertEqual(result.components[0].method, "partial-homoscedasticity")


class ComputeOLSEstimatesTests(unittest.TestCase):
    def test_computes_edge_coefficient_and_mle_residual_variances(self):
        parent = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        noise = np.array([1.0, -2.0, 2.0, -2.0, 1.0])
        child = 2.0 * parent + noise
        X = np.column_stack((parent, child))
        G = np.array([[0, 1], [0, 0]])

        lambda_hat, r = compute_ols_estimates(X, G)

        expected_lambda = np.array([[0.0, 2.0], [0.0, 0.0]])
        np.testing.assert_allclose(lambda_hat, expected_lambda)
        np.testing.assert_allclose(r, [2.0, 2.8])

    def test_root_residual_variance_is_its_empirical_second_moment(self):
        X = np.array([[-2.0, 1.0], [0.0, -2.0], [2.0, 1.0]])
        G = np.zeros((2, 2), dtype=int)

        lambda_hat, r = compute_ols_estimates(X, G)

        np.testing.assert_allclose(lambda_hat, np.zeros((2, 2)))
        np.testing.assert_allclose(r, np.mean(X**2, axis=0))

    def test_rejects_a_rank_deficient_parent_regression(self):
        parent = np.array([-1.0, 0.0, 1.0])
        X = np.column_stack((parent, 2.0 * parent, 3.0 * parent))
        G = np.array(
            [
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 0],
            ]
        )

        with self.assertRaisesRegex(ValueError, "rank deficient"):
            compute_ols_estimates(X, G)


class AllNearFeasibilityLPTests(unittest.TestCase):
    def test_finds_a_strict_all_near_point(self):
        result = solve_all_near_feasibility_lp([[1, 1]], [2], [1, 1])

        self.assertIsInstance(result, AllNearFeasibilityResult)
        self.assertTrue(result.strict_feasible)
        self.assertAlmostEqual(result.margin, 1.0)
        np.testing.assert_allclose(result.x, [1, 1])
        np.testing.assert_allclose(result.omega, [1, 1])

    def test_returns_variances_on_the_original_scale(self):
        result = solve_all_near_feasibility_lp([[1, 1]], [6], [2, 4])

        np.testing.assert_allclose(result.x, [1, 1])
        np.testing.assert_allclose(result.omega, [2, 4])
        np.testing.assert_allclose(np.array([[1, 1]]) @ result.omega, [6])

    def test_distinguishes_boundary_only_feasibility(self):
        result = solve_all_near_feasibility_lp([[1]], [2], [1])

        self.assertTrue(result.closed_box_feasible)
        self.assertFalse(result.strict_feasible)
        self.assertEqual(result.status, "boundary_only")
        self.assertAlmostEqual(result.margin, 0.0)
        np.testing.assert_allclose(result.x, [2])

    def test_reports_closed_box_infeasibility(self):
        result = solve_all_near_feasibility_lp([[1]], [3], [1])

        self.assertFalse(result.closed_box_feasible)
        self.assertEqual(result.status, "infeasible")
        self.assertIsNone(result.x)
        self.assertIsNone(result.omega)
        self.assertIsNone(result.margin)

    def test_accepts_an_empty_constraint_system(self):
        result = solve_all_near_feasibility_lp(np.empty((0, 2)), [], [2, 3])

        self.assertTrue(result.strict_feasible)
        self.assertAlmostEqual(result.margin, 1.0)
        np.testing.assert_allclose(result.x, [1, 1])

    def test_rejects_nonpositive_residual_variances(self):
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            solve_all_near_feasibility_lp([[1, 1]], [1], [1, 0])


class SolveAllNearRootTests(unittest.TestCase):
    def setUp(self):
        self.r = np.array([2.0, 2.8])

    def test_returns_ols_solution_when_constraint_is_already_satisfied(self):
        result = solve_all_near_root(self.r, [[1, 1]], [4.8])

        self.assertIsInstance(result, AllNearRootResult)
        np.testing.assert_allclose(result.x, [1, 1], atol=1e-7)
        np.testing.assert_allclose(result.omega, [2.0, 2.8], atol=1e-7)
        np.testing.assert_allclose(np.array([[1, 1]]) @ result.omega, [4.8])
        self.assertTrue(result.certified_global)

    def test_recovers_partial_homoscedastic_solution(self):
        result = solve_all_near_root(self.r, [[1, -1]], [0])

        np.testing.assert_allclose(result.omega, [2.4, 2.4], atol=1e-6)
        np.testing.assert_allclose(result.x, [1.2, 2.4 / 2.8], atol=1e-6)

    def test_handles_redundant_constraint_rows(self):
        result = solve_all_near_root(
            self.r,
            [[1, 1], [2, 2]],
            [4.8, 9.6],
        )

        np.testing.assert_allclose(result.omega, [2.0, 2.8], atol=1e-7)
        self.assertLess(result.constraint_violation, 1e-8)

    def test_rejects_a_constraint_without_a_strict_all_near_point(self):
        with self.assertRaisesRegex(ValueError, "no strictly all-near"):
            solve_all_near_root([2.0 / 3.0], [[1]], [2])

    def test_rejects_nonpositive_residual_variances(self):
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            solve_all_near_root([1.0, 0.0], [[1, 1]], [1])


if __name__ == "__main__":
    unittest.main()
