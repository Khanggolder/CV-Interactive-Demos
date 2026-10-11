import unittest

import numpy as np

from utils.canny import (KX, KY, PRESETS, base_stages, canny_stages, classify,
                         correlate_reflect, gaussian_kernel, gradient, hysteresis,
                         inspect_nms, nms_examples, non_maximum_suppression,
                         normalize_magnitude, normalize_nms, preset_image,
                         replay_frame, sector, synthetic_input, threshold_stages,
                         weak_examples)
from utils.canny_visualization import STAGES, stage_data


class GaussianTests(unittest.TestCase):
    def test_dimensions_sum_symmetry(self):
        kernel = gaussian_kernel()
        self.assertEqual(kernel.shape, (5, 5))
        self.assertAlmostEqual(kernel.sum(), 1)
        np.testing.assert_allclose(kernel, np.flip(kernel), rtol=0, atol=1e-15)

    def test_sigma_recomputes_weights(self):
        self.assertGreater(gaussian_kernel(sigma=.7)[2, 2], gaussian_kernel(sigma=2)[2, 2])

    def test_invalid_kernel(self):
        for args in ((4, 1), (5, 0), (1, 1), (5, float("nan"))):
            with self.subTest(args=args), self.assertRaises(ValueError):
                gaussian_kernel(*args)

    def test_reflect_not_edge_or_zero(self):
        image = np.arange(9).reshape(3, 3).astype(float)
        kernel = np.zeros((3, 3)); kernel[0, 0] = 1
        self.assertEqual(correlate_reflect(image, kernel)[0, 0], image[1, 1])

    def test_reference_smoothing_matches_hand_sum(self):
        image = np.arange(25).reshape(5, 5) / 255
        kernel = gaussian_kernel()
        expected = (np.pad(image, 2, mode="reflect")[:5, :5] * kernel).sum()
        self.assertAlmostEqual(correlate_reflect(image, kernel)[0, 0], expected)


class GradientTests(unittest.TestCase):
    def test_masks_and_sign(self):
        np.testing.assert_array_equal(KX, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        np.testing.assert_array_equal(KY, [[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        image = np.tile(np.array([0., 1., 2., 3., 4.]), (5, 1))
        gx, gy, magnitude, direction = gradient(image)
        self.assertEqual(gx[2, 2], 8)
        self.assertEqual(gy[2, 2], 0)
        self.assertEqual(magnitude[2, 2], 8)
        self.assertEqual(direction[2, 2], 0)
        self.assertEqual(gradient(-image)[0][2, 2], -8)

    def test_vertical_direction_and_zero(self):
        image = np.repeat(np.arange(5.)[:, None], 5, axis=1)
        gx, gy, magnitude, direction = gradient(image)
        self.assertEqual(gx[2, 2], 0)
        self.assertEqual(gy[2, 2], 8)
        self.assertAlmostEqual(np.rad2deg(direction[2, 2]), 90)
        flat = gradient(np.full((5, 5), 3.0))
        self.assertAlmostEqual(flat[2][2, 2], 0)
        # atan2(0,0) is only a numeric placeholder; UI marks direction Undefined.
        self.assertTrue(np.isfinite(flat[3]).all())

    def test_percentile_scale_clipping(self):
        raw = np.arange(201.).reshape(1, 201)
        normalized, scale = normalize_magnitude(raw)
        self.assertAlmostEqual(scale, np.percentile(raw, 99.5))
        self.assertEqual(normalized[0, -1], 1)
        self.assertTrue(np.isfinite(normalize_magnitude(np.zeros((5, 5)))[0]).all())


class NMSTests(unittest.TestCase):
    def _fixture(self, center, angle):
        magnitude = np.zeros((5, 5))
        magnitude[2, 2] = center
        direction = np.full((5, 5), np.deg2rad(angle))
        return magnitude, direction

    def test_each_sector_neighbors(self):
        expected = {0: ((2, 1), (2, 3)), 45: ((1, 1), (3, 3)),
                    90: ((1, 2), (3, 2)), 135: ((1, 3), (3, 1))}
        for angle, neighbors in expected.items():
            with self.subTest(angle=angle):
                m, d = self._fixture(2, angle)
                info = inspect_nms(m, d, (2, 2))
                self.assertEqual((info["q1"], info["q2"]), neighbors)
                self.assertTrue(info["keep"])
                self.assertEqual(non_maximum_suppression(m, d)[2, 2], 2)
                m[neighbors[0]] = 3
                self.assertFalse(inspect_nms(m, d, (2, 2))["keep"])
                self.assertEqual(non_maximum_suppression(m, d)[2, 2], 0)

    def test_boundary_angles(self):
        for angle, label in ((22.5, 45), (67.5, 90), (112.5, 135), (157.5, 0)):
            self.assertEqual(sector(np.deg2rad(angle))[2], label)

    def test_ties_and_border(self):
        m, d = self._fixture(2, 0)
        m[2, 1] = m[2, 3] = 2
        self.assertTrue(inspect_nms(m, d, (2, 2))["keep"])
        output = non_maximum_suppression(m, d)
        self.assertEqual(output[2, 2], 2)
        self.assertTrue(np.all(output[0] == 0))
        self.assertTrue(np.all(output[:, -1] == 0))
        with self.assertRaises(ValueError):
            inspect_nms(m, d, (0, 0))

    def test_secondary_nms_normalization(self):
        data = np.array([[0., .2], [.4, .8]])
        np.testing.assert_allclose(normalize_nms(data), data / .8)
        np.testing.assert_array_equal(normalize_nms(np.zeros((2, 2))), np.zeros((2, 2)))

    def test_default_reference_points(self):
        arrays = canny_stages(synthetic_input())
        examples = nms_examples(arrays)
        self.assertEqual(examples["keep"], (130, 116))
        self.assertEqual(examples["suppress"], (130, 115))
        for name in ("keep", "suppress"):
            point = examples[name]
            info = inspect_nms(arrays["magnitude"], arrays["direction"], point)
            self.assertEqual(info["keep"], arrays["nms"][point] > 0)


class ThresholdHysteresisTests(unittest.TestCase):
    def test_three_classes_and_zero_exclusion(self):
        nms = np.array([[0., .09, .10, .349, .35, 1.]])
        strong, weak = classify(nms, .10, .35)
        np.testing.assert_array_equal(strong, [[0, 0, 0, 0, 1, 1]])
        np.testing.assert_array_equal(weak, [[0, 0, 1, 1, 0, 0]])

    def test_invalid_threshold_rejected(self):
        for low, high in ((0, .5), (.5, .5), (.9, .1), (.1, 1.1), (np.nan, .5)):
            with self.subTest(low=low, high=high), self.assertRaises(ValueError):
                classify(np.zeros((3, 3)), low, high)

    def test_diagonal_multihop_and_isolated(self):
        strong = np.zeros((7, 7), bool); strong[1, 1] = True
        weak = np.zeros((7, 7), bool)
        weak[2, 2] = weak[3, 3] = weak[4, 4] = weak[6, 0] = weak[6, 1] = True
        replay = hysteresis(strong, weak)
        self.assertTrue(replay.accepted[4, 4])
        self.assertFalse(replay.accepted[6, 0])
        self.assertFalse(replay.accepted[6, 1])
        self.assertTrue(replay.accepted[1, 1])
        np.testing.assert_array_equal(replay_frame(replay, 0)["accepted"], strong)
        np.testing.assert_array_equal(replay_frame(replay, len(replay.events))["accepted"], replay.accepted)

    def test_event_order_and_backward_replay(self):
        strong = np.zeros((4, 4), bool); strong[1, 1] = True
        weak = np.zeros_like(strong); weak[1, 2] = weak[2, 1] = True
        replay = hysteresis(strong, weak)
        self.assertEqual(replay.events[0], ((1, 1), ((1, 2), (2, 1))))
        self.assertEqual(replay_frame(replay, 1)["accepted_weak"], 2)
        self.assertEqual(replay_frame(replay, 0)["accepted_weak"], 0)
        self.assertEqual(replay.events, hysteresis(strong, weak).events)


class PipelineTests(unittest.TestCase):
    def test_reference_image_and_reproducibility(self):
        np.testing.assert_array_equal(synthetic_input(), synthetic_input())
        self.assertEqual(synthetic_input().shape, (224, 224))
        self.assertEqual(synthetic_input().dtype, np.uint8)
        for name in PRESETS:
            np.testing.assert_array_equal(preset_image(name), preset_image(name))

    def test_default_counts_and_examples(self):
        arrays = canny_stages(preset_image(PRESETS[0]))
        self.assertEqual(int(arrays["strong"].sum()), 1500)
        self.assertEqual(int(arrays["weak"].sum()), 447)
        self.assertEqual(int((arrays["weak"] & arrays["edges"]).sum()), 155)
        self.assertEqual(int((arrays["weak"] & ~arrays["edges"]).sum()), 292)
        examples = weak_examples(arrays)
        self.assertTrue(arrays["edges"][examples["connected"]])
        self.assertFalse(arrays["edges"][examples["isolated"]])

    def test_connected_and_isolated_presets_are_real(self):
        connected = canny_stages(preset_image(PRESETS[3]))
        isolated = canny_stages(preset_image(PRESETS[4]))
        self.assertGreater(np.sum(connected["weak"] & connected["edges"]), 0)
        self.assertGreater(np.sum(isolated["weak"] & ~isolated["edges"]), 0)

    def test_parameter_dependencies_and_binary_result(self):
        image = preset_image(PRESETS[0])
        base = base_stages(image)
        changed_sigma = base_stages(image, 2.5)
        self.assertFalse(np.array_equal(base["smoothed"], changed_sigma["smoothed"]))
        self.assertFalse(np.array_equal(base["nms"], changed_sigma["nms"]))
        result = threshold_stages(base, .1, .35)
        changed_threshold = threshold_stages(base, .05, .5)
        np.testing.assert_array_equal(result["smoothed"], changed_threshold["smoothed"])
        self.assertFalse(np.array_equal(result["strong"], changed_threshold["strong"]))
        self.assertEqual(set(np.unique(result["edges"])), {False, True})
        self.assertTrue(np.all(result["edges"][result["strong"]]))
        self.assertFalse(np.any(result["edges"] & ~(result["strong"] | result["weak"])))

    def test_stage_views_use_same_shape_and_domain(self):
        arrays = canny_stages(preset_image(PRESETS[1]))
        for stage in STAGES:
            data, _, _, _ = stage_data(arrays, stage)
            self.assertEqual(data.shape, arrays["original"].shape)
        threshold, _, _, _ = stage_data(arrays, "Double Threshold")
        self.assertEqual(set(np.unique(threshold)), {0, 2})


if __name__ == "__main__":
    unittest.main()
