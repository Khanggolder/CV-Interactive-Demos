"""Independent geometry and finite-vote regression checks for Hough circles."""

import unittest
from dataclasses import replace

import numpy as np

from utils.hough_circle import (
    ANGLES, CENTER, RADIUS, PRESETS, Dataset, Observation, bin_index,
    candidate_center, cell_parameters, circle_circle_intersections,
    circle_from_three_points, circle_point, common_centers, empty_votes,
    estimate_accumulator_memory, generate_circle_example, gradient_candidates,
    make_grid, peaks, preset_dataset, synthetic_gradient, synthetic_intensity,
    top_cells, vote_all, vote_events, vote_next, vote_previous, vote_remaining,
)
from utils.hough_circle_visualization import image_figure


class HoughCircleGeometryTests(unittest.TestCase):
    def setUp(self):
        self.dataset = generate_circle_example()
        self.points = np.array([p.xy for p in self.dataset.points])

    def test_reference_center_radius_and_angles(self):
        self.assertEqual(CENTER, (1.5, .5))
        self.assertEqual(RADIUS, 2.)
        self.assertEqual(ANGLES, (180, 60, 300, 0, 120, 240))

    def test_six_points_independent_trigonometric_reference(self):
        expected = np.column_stack((1.5+2*np.cos(np.deg2rad(ANGLES)),
                                    .5+2*np.sin(np.deg2rad(ANGLES))))
        np.testing.assert_allclose(self.points, expected, atol=1e-14, rtol=0)
        self.assertEqual(len(self.points), 6)

    def test_each_point_satisfies_circle_equation(self):
        residuals = np.sum((self.points-np.array(CENTER))**2, axis=1)-RADIUS**2
        np.testing.assert_allclose(residuals, 0, atol=1e-12)

    def test_each_distance_is_two(self):
        np.testing.assert_allclose(np.linalg.norm(self.points-CENTER, axis=1), 2, atol=1e-12)

    def test_candidate_locus_and_negative_coordinates(self):
        for angle in np.linspace(-180, 180, 37):
            candidate = candidate_center((-2.75, 1.125), 1.4, angle)
            self.assertAlmostEqual(np.linalg.norm(candidate-(-2.75, 1.125)), 1.4)

    def test_full_locus_closes_exactly(self):
        from utils.hough_circle import center_locus
        locus = center_locus(self.points[0], 2., 361)
        np.testing.assert_allclose(locus[0], locus[-1], atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(locus-self.points[0], axis=1), 2, atol=1e-12)

    def test_two_reference_loci_have_two_real_solutions(self):
        result = common_centers(self.points[:2], 2.)
        self.assertEqual(len(result), 2)
        self.assertTrue(any(np.allclose(value, CENTER) for value in result))
        self.assertTrue(any(np.allclose(value, (.5, .5+np.sqrt(3))) for value in result))

    def test_tangent_loci_one_solution(self):
        result = circle_circle_intersections((0, 0), (4, 0), 2, 2)
        self.assertEqual(len(result), 1)
        np.testing.assert_allclose(result[0], (2, 0))

    def test_disjoint_loci_no_solution(self):
        self.assertEqual(circle_circle_intersections((0, 0), (5, 0), 2, 2), ())

    def test_coincident_loci_degenerate(self):
        with self.assertRaisesRegex(ValueError, "vô số"):
            circle_circle_intersections((1, 1), (1, 1), 2, 2)
        self.assertEqual(circle_circle_intersections((1, 1), (1, 1), 2, 3), ())

    def test_third_point_removes_ambiguity(self):
        result = common_centers(self.points[:3], 2.)
        self.assertEqual(len(result), 1)
        np.testing.assert_allclose(result[0], CENTER, atol=1e-12)
        center, radius = circle_from_three_points(self.points[:3])
        np.testing.assert_allclose(center, CENTER, atol=1e-12)
        self.assertAlmostEqual(radius, 2.)

    def test_degenerate_three_points(self):
        with self.assertRaises(ValueError):
            circle_from_three_points(((0, 0), (1, 1), (2, 2)))

    def test_analytic_gradient_and_finite_difference(self):
        gradient = synthetic_gradient(self.points)
        expected_direction = (np.asarray(CENTER)-self.points) / 2
        np.testing.assert_allclose(gradient/np.linalg.norm(gradient, axis=1)[:, None],
                                   expected_direction, atol=1e-12)
        h = 1e-6
        finite = np.column_stack([(synthetic_intensity(self.points+h*axis)-
                                   synthetic_intensity(self.points-h*axis))/(2*h)
                                  for axis in np.eye(2)])
        np.testing.assert_allclose(gradient, finite, atol=2e-8)

    def test_gradient_polarity(self):
        inward = synthetic_gradient(self.points)
        outward = synthetic_gradient(self.points, bright_inside=False)
        np.testing.assert_allclose(outward, -inward)
        for point, inside, outside in zip(self.points, inward, outward):
            np.testing.assert_allclose(gradient_candidates(point, 2, inside, "inward")[0], CENTER, atol=1e-12)
            np.testing.assert_allclose(gradient_candidates(point, 2, outside, "outward")[0], CENTER, atol=1e-12)
            both = gradient_candidates(point, 2, inside, "unknown")
            self.assertEqual(len(both), 2)
            np.testing.assert_allclose(both[0], CENTER, atol=1e-12)

    def test_zero_and_invalid_gradients(self):
        with self.assertRaises(ValueError):
            gradient_candidates((0, 0), 2, (0, 0), "inward")
        with self.assertRaises(ValueError):
            gradient_candidates((0, 0), 2, (np.nan, 0), "inward")

    def test_negative_radius_rejected(self):
        with self.assertRaises(ValueError):
            circle_point((0, 0), -2, 30)

    def test_visual_circle_geometry_and_y_up(self):
        figure = image_figure(self.dataset, active=0, radius=2, phi=37)
        self.assertEqual(figure.layout.yaxis.scaleanchor, "x")
        candidate_trace = next(t for t in figure.data if t.name == "Candidate circle")
        center = candidate_center(self.points[0], 2, 37)
        np.testing.assert_allclose(np.hypot(np.asarray(candidate_trace.x)-center[0],
                                            np.asarray(candidate_trace.y)-center[1]), 2, atol=1e-12)


class HoughCircleVotingTests(unittest.TestCase):
    def setUp(self):
        self.dataset = generate_circle_example()
        self.grid = make_grid()

    def test_half_open_center_and_radius_bins(self):
        edges = np.array([-1., 0., 1.])
        self.assertEqual(bin_index(-1, edges), 0)
        self.assertEqual(bin_index(0, edges), 1)
        self.assertIsNone(bin_index(1, edges))
        self.assertIsNone(bin_index(-1.0001, edges))
        self.assertEqual(bin_index(2, self.grid.r_edges), 0)
        self.assertIsNone(bin_index(self.grid.r_edges[-1], self.grid.r_edges))

    def test_positive_fixed_radius_at_zero_bin_edge(self):
        grid = make_grid(radius=.5, radius_width=1.)
        np.testing.assert_allclose(grid.r_edges, (0., 1.))
        self.assertEqual(cell_parameters(grid, (0, 0, 0))[2], .5)

    def test_out_of_range_candidates_not_clipped(self):
        grid = make_grid(center_bounds=(0, 1, 0, 1), center_width=.5)
        one = Dataset("one", (Observation("p", (0., 0.)),), (), "", "")
        events = vote_events(one, grid, angular_samples=4)
        self.assertTrue(any(event.cell is None for event in events[0]))

    def test_angular_oversampling_one_support_per_point_bin(self):
        one = Dataset("one", (self.dataset.points[0],), (), "", "")
        grid = make_grid(center_width=2.)
        events = vote_events(one, grid, angular_samples=180)[0]
        state = vote_all(empty_votes(one, grid, angular_samples=180), one, grid, angular_samples=180)
        self.assertEqual(int(state.counts.sum()), len({e.cell for e in events if e.cell is not None}))
        self.assertLess(int(state.counts.sum()), len(events))
        self.assertLessEqual(int(state.counts.max()), 1)

    def test_duplicate_observations_do_not_double_support(self):
        point = self.dataset.points[0]
        doubled = replace(self.dataset, points=(point, Observation("copy", point.xy, point.gradient)))
        state = vote_all(empty_votes(doubled, self.grid), doubled, self.grid)
        single = replace(self.dataset, points=(point,))
        expected = vote_all(empty_votes(single, self.grid), single, self.grid)
        np.testing.assert_array_equal(state.counts, expected.counts)

    def test_incremental_equals_batch_independent_reference(self):
        first = replace(self.dataset, points=self.dataset.points[:2])
        state = empty_votes(first, self.grid, angular_samples=24)
        for i in range(2):
            for _ in range(24):
                state = vote_next(state, first, self.grid, i, angular_samples=24)
        batch = vote_all(empty_votes(first, self.grid, angular_samples=24), first,
                         self.grid, angular_samples=24)
        np.testing.assert_array_equal(state.counts, batch.counts)
        # Independently construct per-observation sets directly from NumPy trig.
        reference = np.zeros(self.grid.shape, dtype=np.int32)
        for obs in first.points:
            phi = np.deg2rad(np.arange(24)*360/24)
            candidates = np.asarray(obs.xy) + 2*np.column_stack((np.cos(phi), np.sin(phi)))
            cells = set()
            for a, b in candidates:
                col, row = bin_index(a, self.grid.a_edges), bin_index(b, self.grid.b_edges)
                if row is not None and col is not None:
                    cells.add((0, row, col))
            for cell in cells:
                reference[cell] += 1
        np.testing.assert_array_equal(batch.counts, reference)

    def test_previous_step_and_rerun_do_not_duplicate(self):
        state = empty_votes(self.dataset, self.grid)
        once = vote_next(state, self.dataset, self.grid, 0)
        self.assertEqual(int(once.counts.sum()), 1)
        self.assertEqual(int(once.counts.sum()), 1)  # reading/rerun is pure
        back = vote_previous(once, self.dataset, self.grid, 0)
        self.assertEqual(int(back.counts.sum()), 0)
        again = vote_next(back, self.dataset, self.grid, 0)
        np.testing.assert_array_equal(again.counts, once.counts)

    def test_vote_selected_and_all(self):
        state = empty_votes(self.dataset, self.grid)
        selected = vote_remaining(state, self.dataset, self.grid, 0)
        self.assertGreater(selected.counts.sum(), 0)
        all_points = vote_all(selected, self.dataset, self.grid)
        clean_batch = vote_all(empty_votes(self.dataset, self.grid), self.dataset, self.grid)
        np.testing.assert_array_equal(all_points.counts, clean_batch.counts)

    def test_stale_radius_resolution_dataset_rejected(self):
        state = empty_votes(self.dataset, self.grid)
        with self.assertRaises(ValueError):
            vote_next(state, self.dataset, make_grid(radius=1), 0)
        with self.assertRaises(ValueError):
            vote_next(state, self.dataset, make_grid(center_width=1), 0)
        with self.assertRaises(ValueError):
            vote_next(state, preset_dataset(PRESETS[2]), self.grid, 0)

    def test_fixed_radius_peak_and_bin_center(self):
        state = vote_all(empty_votes(self.dataset, self.grid, "gradient"), self.dataset,
                         self.grid, mode="gradient")
        self.assertEqual(int(state.counts.sum()), 6)
        maximum, winners = peaks(state.counts)
        self.assertEqual((maximum, len(winners)), (6, 1))
        a, b, r = cell_parameters(self.grid, winners[0])
        self.assertEqual(r, 2.)
        self.assertLess(np.hypot(a-1.5, b-.5), .4)
        self.assertNotEqual((a, b), CENTER)

    def test_unknown_radius_layers(self):
        grid = make_grid(radius_range=(.25, 3.25), radius_width=.5)
        state = vote_all(empty_votes(self.dataset, grid, "gradient"), self.dataset,
                         grid, mode="gradient")
        self.assertEqual(grid.shape[0], 6)
        self.assertEqual(int(state.counts.sum()), 36)
        self.assertGreaterEqual(int(state.counts.max()), 6)
        self.assertTrue(any(np.isclose(cell_parameters(grid, cell)[2], 2.) for _, cell in top_cells(state.counts)))

    def test_peak_ties_explicit_and_deterministic(self):
        counts = np.zeros((1, 2, 2), dtype=np.int32)
        counts[0, 1, 0] = counts[0, 0, 1] = 3
        self.assertEqual(peaks(counts), (3, ((0, 0, 1), (0, 1, 0))))

    def test_peak_tie_count_not_truncated(self):
        counts = np.ones((1, 3, 5), dtype=np.int32)
        maximum, winners = peaks(counts)
        self.assertEqual(maximum, 1)
        self.assertEqual(len(winners), 15)

    def test_wrong_polarity_moves_vote(self):
        inward = vote_all(empty_votes(self.dataset, self.grid, "gradient"), self.dataset,
                          self.grid, mode="gradient")
        wrong = vote_all(empty_votes(self.dataset, self.grid, "gradient", polarity="outward"),
                         self.dataset, self.grid, mode="gradient", polarity="outward")
        self.assertFalse(np.array_equal(inward.counts, wrong.counts))
        unknown = vote_all(empty_votes(self.dataset, self.grid, "unknown"), self.dataset,
                           self.grid, mode="unknown")
        self.assertGreater(int(unknown.counts.sum()), int(inward.counts.sum()))

    def test_bright_outside_preset_uses_outward_vote(self):
        dataset = preset_dataset(PRESETS[4])
        state = vote_all(empty_votes(dataset, self.grid, "gradient", polarity="outward"),
                         dataset, self.grid, mode="gradient", polarity="outward")
        self.assertEqual(int(state.counts.sum()), 6)
        self.assertEqual(peaks(state.counts)[0], 6)

    def test_radius_bin_width_change_invalidates_votes(self):
        state = empty_votes(self.dataset, self.grid)
        changed = make_grid(radius_width=.25)
        with self.assertRaisesRegex(ValueError, "cấu hình cũ"):
            vote_next(state, self.dataset, changed, 0)

    def test_missing_gradient_has_no_fabricated_vote(self):
        point = Observation("custom", (1., 2.), None, False)
        dataset = Dataset("custom", (point,), (), "", "")
        events = vote_events(dataset, self.grid, mode="gradient")
        self.assertEqual(events, ((),))
        state = vote_all(empty_votes(dataset, self.grid, "gradient"), dataset, self.grid, mode="gradient")
        self.assertEqual(int(state.counts.sum()), 0)

    def test_zero_gradient_safe(self):
        dataset = Dataset("zero", (Observation("z", (1., 2.), (0., 0.)),), (), "", "")
        self.assertEqual(vote_events(dataset, self.grid, mode="gradient"), ((),))

    def test_competing_circles_and_presets_deterministic(self):
        for name in PRESETS:
            self.assertEqual(preset_dataset(name), preset_dataset(name))
        dataset = preset_dataset(PRESETS[5])
        grid = make_grid(radius_range=(.375, 3.625), radius_width=.25)
        state = vote_all(empty_votes(dataset, grid, "gradient"), dataset, grid, mode="gradient")
        self.assertEqual(int(state.counts.sum()), 12*grid.shape[0])
        self.assertGreater(len(top_cells(state.counts)), 1)

    def test_memory_guard_and_finite_outputs(self):
        self.assertEqual(estimate_accumulator_memory((2, 3, 4)), 96)
        self.assertEqual(estimate_accumulator_memory((10**10, 10**10, 10**10)), 4*10**30)
        with self.assertRaisesRegex(ValueError, "quá lớn"):
            make_grid(center_bounds=(-100, 100, -100, 100), center_width=.01,
                      radius_range=(.5, 3.5), radius_width=.25)
        for name in PRESETS:
            data = preset_dataset(name)
            self.assertTrue(np.isfinite(np.array([p.xy for p in data.points])).all())


if __name__ == "__main__":
    unittest.main()
