import unittest

import numpy as np

from utils.hough import (
    ALIGNED, DEFAULT_POINTS, accumulate, bin_index, empty_votes, line_from_points,
    line_segment, make_grid, peak_geometry, peaks, point_votes, signed_rho,
    vote_all, vote_next, vote_remaining,
)
from utils.hough_visualization import accumulator_figure, continuous_figure, image_figure


class HoughNumericalTests(unittest.TestCase):
    def test_known_rho_and_signed_negative(self):
        self.assertAlmostEqual(signed_rho((1, 2), 0), 1)
        self.assertAlmostEqual(signed_rho((1, 2), 90), 2)
        self.assertLess(signed_rho((1, 2), 179), 0)
        self.assertAlmostEqual(signed_rho((-2, 0), 0), -2)

    def test_sinusoid_samples_match_formula(self):
        angles = np.array([0, 30, 45, 90, 179.9])
        expected = np.cos(np.deg2rad(angles)) + 2 * np.sin(np.deg2rad(angles))
        np.testing.assert_allclose(signed_rho((1, 2), angles), expected)

    def test_two_point_intersection_and_aligned_points(self):
        rho, theta = line_from_points((1, 2), (3, 1))
        self.assertAlmostEqual(rho, np.sqrt(5))
        self.assertAlmostEqual(theta, np.rad2deg(np.arctan2(2, 1)))
        for point in ALIGNED:
            self.assertAlmostEqual(signed_rho(point, theta), rho)

    def test_vertical_horizontal_and_coincident(self):
        for first, second in (((3, 0), (3, 2)), ((0, 2), (3, 2))):
            rho, theta = line_from_points(first, second)
            self.assertAlmostEqual(signed_rho(first, theta), rho)
            self.assertAlmostEqual(signed_rho(second, theta), rho)
            segment = line_segment(rho, theta, (-1, 5, -1, 4))
            self.assertIsNotNone(segment)
            self.assertTrue(np.isfinite(segment).all())
        with self.assertRaises(ValueError):
            line_from_points((1, 2), (1, 2))

    def test_half_open_bins_and_boundaries(self):
        edges = np.array([-5.0, -4.5, -4.0])
        self.assertEqual(bin_index(-5.0, edges), 0)
        self.assertEqual(bin_index(-4.5, edges), 1)
        with self.assertRaises(ValueError):
            bin_index(-4.0, edges)
        with self.assertRaises(ValueError):
            bin_index(-5.01, edges)
        with self.assertRaises(ValueError):
            bin_index(float('nan'), edges)

    def test_reference_grid_and_votes(self):
        grid = make_grid()
        self.assertEqual(grid.shape, (20, 18))
        np.testing.assert_allclose(grid.theta_centers, np.arange(5, 180, 10))
        np.testing.assert_allclose(grid.rho_centers, np.arange(-4.75, 5, 0.5))
        for point in DEFAULT_POINTS:
            self.assertEqual(len(point_votes(point, grid)), 18)
        counts = accumulate(DEFAULT_POINTS, grid)
        self.assertEqual(int(counts.sum()), 162)
        self.assertEqual(int(counts[14, 6]), 6)
        self.assertEqual(peaks(counts), (6, ((14, 6),)))
        self.assertEqual(sorted(counts.flat)[-2], 3)
        self.assertEqual(peak_geometry(grid, 14, 6), (2.25, 65.0))

    def test_incremental_batch_and_reset(self):
        grid = make_grid()
        state = empty_votes(DEFAULT_POINTS, grid)
        state = vote_next(state, DEFAULT_POINTS, grid, 0)
        self.assertEqual(int(state.counts.sum()), 1)
        self.assertEqual(state.next_indices[0], 1)
        state = vote_remaining(state, DEFAULT_POINTS, grid, 0)
        self.assertEqual(int(state.counts.sum()), 18)
        state = vote_all(state, DEFAULT_POINTS, grid)
        np.testing.assert_array_equal(state.counts, accumulate(DEFAULT_POINTS, grid))
        self.assertEqual(int(vote_all(state, DEFAULT_POINTS, grid).counts.sum()), 162)
        self.assertEqual(int(empty_votes(DEFAULT_POINTS, grid).counts.sum()), 0)

    def test_custom_range_resolution_and_ties(self):
        points = ((12.0, -7.0), (0.0, 0.0))
        grid = make_grid(5.0, 0.25, points)
        self.assertGreater(grid.rho_edges[-1], 12)
        self.assertEqual(int(accumulate(points, grid).sum()), 72)
        coarse = make_grid(30.0, 1.0, DEFAULT_POINTS)
        self.assertEqual(coarse.shape, (10, 6))
        self.assertEqual(int(accumulate(DEFAULT_POINTS, coarse).sum()), 54)
        odd = make_grid(7.0, 0.5)
        np.testing.assert_allclose(np.diff(odd.theta_edges)[:-1], 7.0)
        self.assertEqual(odd.theta_edges[-1], 180.0)
        self.assertLess(odd.theta_edges[-1] - odd.theta_edges[-2], 7.0)
        self.assertEqual(peaks(np.array([[2, 1], [0, 2]])), (2, ((0, 0), (1, 1))))

    def test_finite_projection_and_candidate(self):
        for angle in (0, 1e-8, 45, 90, 179.9, 179.999999):
            rho = signed_rho((-2.5, 3.5), angle)
            self.assertTrue(np.isfinite(rho))
            segment = line_segment(rho, angle, (-4, 5, -1, 5))
            self.assertIsNotNone(segment)
            self.assertTrue(np.isfinite(segment).all())

    def test_two_panels_share_current_theta_and_rho(self):
        theta = 123.4
        rho = signed_rho(DEFAULT_POINTS[0], theta)
        image = image_figure(DEFAULT_POINTS, active=0, theta=theta)
        hough = continuous_figure(DEFAULT_POINTS, active=0, theta=theta)
        marker = next(trace for trace in hough.data if trace.name == "(θ, ρ) hiện tại")
        self.assertAlmostEqual(marker.x[0], theta)
        self.assertAlmostEqual(marker.y[0], rho)
        candidate = next(trace for trace in image.data if trace.name == "Candidate qua điểm đang chọn")
        for x, y in zip(candidate.x, candidate.y):
            self.assertAlmostEqual(signed_rho((x, y), theta), rho)

    def test_accumulator_heatmap_uses_true_counts(self):
        grid = make_grid()
        counts = accumulate(DEFAULT_POINTS, grid)
        figure = accumulator_figure(counts, grid, show_peak=True)
        np.testing.assert_array_equal(figure.data[0].z, counts)
        marker = next(trace for trace in figure.data if trace.name == "Peak")
        self.assertEqual((marker.x[0], marker.y[0]), (65.0, 2.25))


if __name__ == '__main__':
    unittest.main()
