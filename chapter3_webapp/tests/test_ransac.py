import unittest

import numpy as np

from utils.ransac import (
    DegenerateSampleError,
    RansacError,
    classify_inliers,
    consensus_score,
    deterministic_pairs,
    fit_ordinary_least_squares,
    fit_tls,
    generate_dataset,
    guided_dataset,
    line_from_two_points,
    normalize_line,
    perpendicular_distances,
    run_ransac,
)


class LineGeometryTests(unittest.TestCase):
    def test_line_from_two_points_and_perpendicular_distance(self) -> None:
        points = np.array([[0.0, 1.0], [2.0, 5.0]])
        model = line_from_two_points(points[0], points[1])
        np.testing.assert_allclose(perpendicular_distances(points, model), 0.0, atol=1e-12)
        query = np.array([[0.0, 0.0]])
        expected = 1.0 / np.sqrt(5.0)
        self.assertAlmostEqual(float(perpendicular_distances(query, model)[0]), expected)

    def test_vertical_line_has_no_slope_special_case(self) -> None:
        model = line_from_two_points(np.array([2.0, -4.0]), np.array([2.0, 7.0]))
        np.testing.assert_allclose(model, np.array([1.0, 0.0, -2.0]), atol=1e-12)
        distances = perpendicular_distances(np.array([[5.0, 100.0], [1.5, -9.0]]), model)
        np.testing.assert_allclose(distances, np.array([3.0, 0.5]))

    def test_classification_is_inclusive_and_score_matches(self) -> None:
        model = normalize_line([0.0, 1.0, 0.0])
        points = np.array([[0.0, 0.2], [1.0, -0.2], [2.0, 0.200001]])
        mask = classify_inliers(points, model, 0.2)
        np.testing.assert_array_equal(mask, np.array([True, True, False]))
        self.assertEqual(consensus_score(points, model, 0.2), 2)

    def test_zero_epsilon_keeps_the_two_sampled_points(self) -> None:
        points = np.array([[0.0, 1.0], [2.0, 5.0]])
        result = run_ransac(points, epsilon=0.0, seed=0, iterations=1)
        self.assertEqual(result.best_trial.score, 2)
        np.testing.assert_array_equal(result.best_trial.inlier_mask, [True, True])

    def test_duplicate_and_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(DegenerateSampleError):
            line_from_two_points(np.array([1.0, 1.0]), np.array([1.0, 1.0]))
        with self.assertRaises(RansacError):
            perpendicular_distances(np.empty((0, 2)), np.array([1.0, 0.0, 0.0]))
        with self.assertRaises(RansacError):
            classify_inliers(np.array([[0.0, 0.0]]), np.array([1.0, 0.0, 0.0]), -1.0)


class RansacAlgorithmTests(unittest.TestCase):
    def test_sampling_is_deterministic_and_skips_duplicate_coordinates(self) -> None:
        points = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        first = deterministic_pairs(points, seed=9, count=8)
        second = deterministic_pairs(points, seed=9, count=8)
        self.assertEqual(first, second)
        for left, right in first:
            self.assertFalse(np.array_equal(points[left], points[right]))

    def test_tls_refit_known_horizontal_and_vertical_lines(self) -> None:
        horizontal = np.array([[-2.0, 3.0], [0.0, 3.0], [5.0, 3.0]])
        vertical = np.array([[4.0, -2.0], [4.0, 0.0], [4.0, 8.0]])
        np.testing.assert_allclose(fit_tls(horizontal), [0.0, 1.0, -3.0], atol=1e-12)
        np.testing.assert_allclose(fit_tls(vertical), [1.0, 0.0, -4.0], atol=1e-12)

    def test_tls_reduces_consensus_perpendicular_error(self) -> None:
        data = guided_dataset()
        result = run_ransac(data.points, epsilon=0.22, seed=7647, iterations=3)
        self.assertIsNotNone(result.best_trial)
        self.assertIsNotNone(result.refit_model)
        best = result.best_trial
        before = np.square(best.distances[best.inlier_mask]).sum()
        after = np.square(
            perpendicular_distances(data.points[best.inlier_mask], result.refit_model)
        ).sum()
        self.assertLess(after, before)

    def test_guided_fixture_matches_approved_numerical_story(self) -> None:
        data = guided_dataset()
        result = run_ransac(data.points, epsilon=0.22, seed=7647, iterations=3)
        self.assertEqual([trial.pair for trial in result.trials], [(14, 6), (5, 2), (2, 11)])
        self.assertEqual([trial.score for trial in result.trials], [3, 7, 14])
        self.assertEqual(result.best_trial.iteration, 3)
        np.testing.assert_array_equal(result.best_trial.inlier_mask, data.generated_inlier_labels)

    def test_generated_dataset_and_ransac_repeat_exactly(self) -> None:
        first = generate_dataset(25, 15, 0.12, seed=123)
        second = generate_dataset(25, 15, 0.12, seed=123)
        np.testing.assert_array_equal(first.points, second.points)
        np.testing.assert_array_equal(first.generated_inlier_labels, second.generated_inlier_labels)
        result_1 = run_ransac(first.points, epsilon=0.25, seed=777, iterations=30)
        result_2 = run_ransac(second.points, epsilon=0.25, seed=777, iterations=30)
        self.assertEqual(
            [trial.pair for trial in result_1.trials],
            [trial.pair for trial in result_2.trials],
        )
        np.testing.assert_allclose(result_1.refit_model, result_2.refit_model)

    def test_zero_iterations_returns_no_model(self) -> None:
        points = np.array([[0.0, 0.0], [1.0, 1.0]])
        result = run_ransac(points, epsilon=0.1, seed=1, iterations=0)
        self.assertEqual(result.trials, ())
        self.assertIsNone(result.best_trial)
        self.assertIsNone(result.refit_model)

    def test_all_duplicate_cloud_has_no_valid_model(self) -> None:
        points = np.ones((5, 2))
        with self.assertRaises(DegenerateSampleError):
            run_ransac(points, epsilon=0.1, seed=1, iterations=1)
        with self.assertRaises(DegenerateSampleError):
            fit_tls(points)

    def test_ordinary_least_squares_vertical_fallback(self) -> None:
        points = np.array([[2.0, -1.0], [2.0, 0.0], [2.0, 4.0]])
        np.testing.assert_allclose(
            fit_ordinary_least_squares(points), np.array([1.0, 0.0, -2.0]), atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
