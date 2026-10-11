"""Numerical regression for the LSD LSR/NFA teaching model."""

import math
import unittest

import numpy as np

from utils.lsd import (DEFAULT_TAU, PRESETS, analyze_field, angular_difference_2pi,
                       angular_difference_pi, compute_binomial_tail, compute_gradient_field,
                       compute_level_line_angles, compute_nfa, fit_lsr_segment,
                       generate_lsd_reference_field, grow_lsr, log_binomial_tail,
                       neighbors, preset_field, replay_lsr_events, select_lsd_seed,
                       validate_intensity_field, validate_parameters)
from utils.lsd_visualization import field_figure


class LSDNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.field = generate_lsd_reference_field()
        cls.data = analyze_field(cls.field)

    def test_reference_field_formula_and_determinism(self):
        self.assertEqual(self.field.shape, (10, 12))
        y, x = np.indices((10, 12), dtype=float)
        expected = (.5 + .42*np.tanh((y-.55*x-1.3)/.8)
                    + .40*np.exp(-((x-8.8)**2+(y-1.6)**2)/(2*.75**2))
                    + np.random.default_rng(9).normal(0, .055, (10, 12)))
        np.testing.assert_array_equal(self.field, expected)
        np.testing.assert_array_equal(self.field, generate_lsd_reference_field())

    def test_reference_not_clipped(self):
        np.testing.assert_array_equal(self.field, self.data["intensity"])

    def test_gradient_is_numpy_gradient_and_hypot(self):
        gy, gx = np.gradient(self.field)
        np.testing.assert_array_equal(self.data["gx"], gx)
        np.testing.assert_array_equal(self.data["gy"], gy)
        np.testing.assert_allclose(self.data["magnitude"], np.sqrt(gx*gx+gy*gy))

    def test_level_lines_perpendicular_directed(self):
        d = self.data
        theta = d["angles"]
        np.testing.assert_allclose(np.cos(theta)*d["gx"]+np.sin(theta)*d["gy"], 0, atol=1e-14)
        np.testing.assert_allclose(theta, (np.arctan2(d["gy"], d["gx"])-math.pi/2)%(2*math.pi))

    def test_zero_gradient_has_no_eligible_seed(self):
        field = np.ones((4, 4))
        gx, gy, magnitude = compute_gradient_field(field)
        self.assertTrue(np.all(magnitude == 0))
        self.assertIsNone(select_lsd_seed(magnitude, np.zeros((4, 4), bool)))
        self.assertEqual(analyze_field(field)["tests"], 0)
        self.assertTrue(np.isfinite(compute_level_line_angles(gx, gy)).all())

    def test_angular_axial_and_directed_opposites(self):
        a, b = np.deg2rad((5, 185))
        self.assertAlmostEqual(angular_difference_pi(a, b), 0)
        self.assertAlmostEqual(angular_difference_2pi(a, b), math.pi)
        self.assertAlmostEqual(angular_difference_pi(np.deg2rad(5), np.deg2rad(175)), np.deg2rad(10))

    def test_strict_tolerance_boundary(self):
        angles = np.zeros((4, 4))
        angles[1, 2] = DEFAULT_TAU
        magnitude = np.zeros((4, 4))
        magnitude[1, 1] = 2
        magnitude[1, 2] = 1
        region = grow_lsr((1, 1), angles, magnitude, np.zeros((4, 4), bool))
        self.assertEqual(region["pixels"], [(1, 1)])
        self.assertFalse(region["events"][0]["accepted"])

    def test_seed_strongest_unused_and_row_major_tie(self):
        magnitude = np.zeros((4, 4))
        magnitude[1, 3] = magnitude[2, 1] = 2
        used = np.zeros((4, 4), bool)
        self.assertEqual(select_lsd_seed(magnitude, used), (1, 3))
        used[1, 3] = True
        self.assertEqual(select_lsd_seed(magnitude, used), (2, 1))
        self.assertIsNone(select_lsd_seed(magnitude, used, 3))

    def test_neighbors_order_and_bounds(self):
        self.assertEqual(list(neighbors((0, 0), (4, 4))), [(0, 1), (1, 0), (1, 1)])
        self.assertEqual(list(neighbors((2, 2), (4, 4))),
                         [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)])

    def test_bfs_fixed_seed_and_rejected_not_used(self):
        angles = np.zeros((4, 4))
        angles[1, 2] = .3
        angles[1, 3] = .6
        magnitude = np.zeros((4, 4))
        magnitude[1, 1:4] = 1
        used = np.zeros((4, 4), bool)
        region = grow_lsr((1, 1), angles, magnitude, used, tau=.4)
        self.assertEqual(region["pixels"], [(1, 1), (1, 2)])
        self.assertEqual(region["events"][-1]["reference"], 0)
        self.assertFalse(used[1, 3])
        self.assertEqual(set(region["pixels"]), set(map(tuple, np.argwhere(used))))

    def test_replay_prefix_is_exact_and_idempotent(self):
        region = self.data["regions"][0]
        for count in range(len(region["events"])+1):
            state = replay_lsr_events(region, count)
            expected = [region["seed"]] + [e["pixel"] for e in region["events"][:count] if e["accepted"]]
            self.assertEqual(state["members"], expected)
            self.assertEqual(len(state["members"]), len(set(state["members"])))
        self.assertEqual(replay_lsr_events(region, 0)["members"], [region["seed"]])
        self.assertEqual(replay_lsr_events(region, len(region["events"]))["members"], region["pixels"])
        self.assertEqual(replay_lsr_events(region, 3), replay_lsr_events(region, 3))

    def test_fit_weighted_centroid_and_pca(self):
        region = self.data["regions"][0]
        fit = region["fit"]
        pixels = np.asarray(region["pixels"])
        xy = pixels[:, ::-1]
        weights = self.data["magnitude"][tuple(pixels.T)]
        expected = np.sum(weights[:, None]*xy, axis=0)/np.sum(weights)
        np.testing.assert_allclose(fit["center"], expected)
        scatter = ((xy-expected)*weights[:, None]).T@(xy-expected)
        direction = fit["tangent"]
        self.assertAlmostEqual(direction@scatter@direction, np.linalg.eigvalsh(scatter)[-1])
        self.assertGreaterEqual(direction@np.sum(weights[:, None]*np.stack(
            (np.cos(self.data["angles"][tuple(pixels.T)]),
             np.sin(self.data["angles"][tuple(pixels.T)])), axis=-1), axis=0), -1e-12)

    def test_finite_endpoints_and_rectangle_contains_squares(self):
        fit = self.data["regions"][0]["fit"]
        self.assertTrue(np.isfinite(fit["endpoints"]).all())
        self.assertGreater(np.linalg.norm(np.diff(fit["endpoints"], axis=0)), 0)
        for y, x in self.data["regions"][0]["pixels"]:
            for dx in (-.5, .5):
                for dy in (-.5, .5):
                    point = np.array((x+dx, y+dy))-fit["center"]
                    box = fit["rectangle"]-fit["center"]
                    for direction in (fit["tangent"], fit["normal"]):
                        projection = point@direction
                        limits = box@direction
                        self.assertGreaterEqual(projection+1e-12, min(limits))
                        self.assertLessEqual(projection-1e-12, max(limits))

    def test_degenerate_one_pixel_fit(self):
        magnitude = np.zeros((4, 4))
        magnitude[1, 1] = 1
        angles = np.zeros((4, 4))
        region = {"seed": (1, 1), "reference": 0., "pixels": [(1, 1)]}
        fit = fit_lsr_segment(region, magnitude, angles)
        self.assertEqual((fit["n"], fit["k"]), (1, 1))
        self.assertTrue(np.isfinite(fit["rectangle"]).all())

    def test_directed_alignment_independent(self):
        for region in self.data["regions"]:
            pixels = np.array(region["pixels"])
            angles = self.data["angles"][tuple(pixels.T)]
            direct = np.cos(angles-region["fit"]["angle"]) > np.cos(DEFAULT_TAU)
            np.testing.assert_array_equal(direct, region["fit"]["aligned"])
            self.assertEqual(int(direct.sum()), region["fit"]["k"])

    def test_binomial_tails_independent_cases(self):
        self.assertAlmostEqual(compute_binomial_tail(4, 2, .125), 1.5771484375/20)
        self.assertAlmostEqual(compute_binomial_tail(0, 0, .5), 1)
        self.assertEqual(compute_binomial_tail(4, 0, 0), 1)
        self.assertEqual(compute_binomial_tail(4, 2, 0), 0)
        self.assertEqual(compute_binomial_tail(4, 2, 1), 1)
        self.assertAlmostEqual(compute_binomial_tail(4, 4, .125), .125**4)
        self.assertAlmostEqual(compute_nfa(33, 33, 20, .125), 20*.125**33, places=40)

    def test_stable_large_tail_and_extreme_p(self):
        value = log_binomial_tail(1600, 1600, .01)
        self.assertTrue(np.isfinite(value))
        self.assertAlmostEqual(value, 1600*math.log(.01))
        self.assertGreaterEqual(compute_binomial_tail(1000, 100, .999), 0)
        self.assertLessEqual(compute_binomial_tail(1000, 100, .999), 1)

    def test_reference_candidates_and_decisions(self):
        data = self.data
        self.assertEqual(data["tests"], 20)
        a, b = data["regions"][:2]
        self.assertEqual((a["seed"], a["fit"]["n"], a["fit"]["k"]), ((6, 8), 33, 33))
        self.assertEqual((b["seed"], b["fit"]["n"], b["fit"]["k"]), ((1, 0), 4, 2))
        self.assertAlmostEqual(a["nfa"], 3.155443620884047e-29, delta=1e-40)
        self.assertAlmostEqual(b["nfa"], 1.5771484375, places=10)
        self.assertTrue(a["accept"])
        self.assertFalse(b["accept"])

    def test_reference_full_seed_scan(self):
        used = np.zeros(self.field.shape, bool)
        for region in self.data["regions"]:
            self.assertEqual(select_lsd_seed(self.data["magnitude"], used), region["seed"])
            for pixel in region["pixels"]:
                self.assertFalse(used[pixel])
                used[pixel] = True
        self.assertIsNone(select_lsd_seed(self.data["magnitude"], used))

    def test_tau_recomputes_region_and_nfa(self):
        changed = analyze_field(self.field, tau=np.deg2rad(30))
        self.assertEqual(changed["tau"], np.deg2rad(30))
        self.assertNotEqual((changed["tests"], changed["regions"][0]["fit"]["n"]),
                            (self.data["tests"], self.data["regions"][0]["fit"]["n"]))
        first = changed["regions"][0]
        self.assertAlmostEqual(first["log_nfa"], math.log(changed["tests"])
                               + log_binomial_tail(first["fit"]["n"], first["fit"]["k"], 1/6))

    def test_magnitude_recomputes_seed_and_scan(self):
        changed = analyze_field(self.field, min_magnitude=.2)
        self.assertNotEqual(changed["tests"], self.data["tests"])
        self.assertTrue(all(changed["magnitude"][r["seed"]] >= .2 for r in changed["regions"]))

    def test_epsilon_only_changes_decision(self):
        changed = analyze_field(self.field, epsilon=2)
        self.assertEqual(changed["tests"], self.data["tests"])
        np.testing.assert_array_equal(changed["regions"][1]["fit"]["endpoints"],
                                      self.data["regions"][1]["fit"]["endpoints"])
        self.assertTrue(changed["regions"][1]["accept"])
        self.assertAlmostEqual(changed["regions"][1]["nfa"], self.data["regions"][1]["nfa"])

    def test_manual_region_does_not_mutate_official_scan(self):
        data = self.data
        used = np.zeros(self.field.shape, bool)
        region = grow_lsr((1, 0), data["angles"], data["magnitude"], used)
        self.assertEqual(region["seed"], (1, 0))
        self.assertEqual(data["tests"], 20)
        self.assertEqual(data["regions"][0]["seed"], (6, 8))

    def test_presets_deterministic_distinct_and_finite(self):
        fields = [preset_field(p) for p in PRESETS]
        for p, field in zip(PRESETS, fields):
            np.testing.assert_array_equal(field, preset_field(p))
            self.assertTrue(np.isfinite(field).all())
            result = analyze_field(field)
            self.assertTrue(all(np.isfinite(r["fit"]["rectangle"]).all() for r in result["regions"]))
        self.assertFalse(np.array_equal(fields[0], fields[-1]))

    def test_invalid_inputs_and_parameters(self):
        for field in (np.array([1, 2]), np.full((4, 4), np.nan), np.ones((41, 4))):
            with self.assertRaises(ValueError):
                validate_intensity_field(field)
        for kwargs in ({"tau": 0}, {"tau": math.pi/2}, {"min_magnitude": -1}, {"epsilon": 0}):
            with self.assertRaises(ValueError):
                validate_parameters(**kwargs)
        with self.assertRaises(ValueError):
            log_binomial_tail(4, 5, .5)

    def test_custom_gradient_and_region_validation(self):
        with self.assertRaises(ValueError):
            compute_level_line_angles(np.ones((4, 4)), np.full((4, 4), np.nan))
        with self.assertRaises(ValueError):
            select_lsd_seed(np.full((4, 4), np.inf), np.zeros((4, 4), bool))
        with self.assertRaises(ValueError):
            grow_lsr((1, 1), np.full((4, 4), np.nan), np.ones((4, 4)), np.zeros((4, 4), bool))
        with self.assertRaises(ValueError):
            fit_lsr_segment({"pixels": [(1, 1), (1, 1)], "reference": 0.},
                            np.ones((4, 4)), np.zeros((4, 4)))

    def test_visualization_y_up_and_real_geometry(self):
        figure = field_figure(self.data, region=self.data["regions"][0], fit=True, rectangle=True,
                              nfa=True, view="Level lines")
        corners = self.data["regions"][0]["fit"]["rectangle"]
        self.assertLess(figure.layout.xaxis.range[0], float(corners[:, 0].min()))
        self.assertGreater(figure.layout.xaxis.range[1], float(corners[:, 0].max()))
        self.assertLess(figure.layout.yaxis.range[0], float(corners[:, 1].min()))
        self.assertGreater(figure.layout.yaxis.range[1], float(corners[:, 1].max()))
        self.assertEqual(figure.layout.yaxis.scaleanchor, "x")
        line = next(t for t in figure.data if t.name == "Finite segment")
        np.testing.assert_allclose(line.x, self.data["regions"][0]["fit"]["endpoints"][:, 0])


if __name__ == "__main__":
    unittest.main()
