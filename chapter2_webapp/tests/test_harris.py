import unittest

import numpy as np

from utils.harris import (KX, KY, PRESET_NAMES, classify_eigenvalues, eigenvalues,
                          energy_grid, gradients, harris_response, integration_kernel,
                          nms_decision, pipeline, presets, shift_energy,
                          shi_tomasi_response, taylor_energy, tensor_at, tensor_maps,
                          threshold_nms)


class HarrisNumerics(unittest.TestCase):
    def setUp(self):
        self.images = presets()
        self.at = (20, 20)

    def test_all_presets_deterministic(self):
        self.assertEqual(len(self.images), 8)
        for name in PRESET_NAMES:
            self.assertEqual(self.images[name].shape, (41, 41))
            np.testing.assert_array_equal(self.images[name], presets()[name])

    def test_flat_energy_and_no_wrap(self):
        self.assertEqual(shift_energy(self.images[PRESET_NAMES[0]], *self.at, 9, 2, -1), 0)
        with self.assertRaises(ValueError):
            shift_energy(self.images[PRESET_NAMES[3]], 1, 1, 9, -2, 0)

    def test_edge_directional_energy(self):
        horizontal = self.images[PRESET_NAMES[1]]
        vertical = self.images[PRESET_NAMES[2]]
        self.assertEqual(shift_energy(horizontal, *self.at, 9, 2, 0), 0)
        self.assertGreater(shift_energy(horizontal, *self.at, 9, 0, 2), 0)
        self.assertEqual(shift_energy(vertical, *self.at, 9, 0, 2), 0)
        self.assertGreater(shift_energy(vertical, *self.at, 9, 2, 0), 0)

    def test_corner_energy_two_directions(self):
        corner = self.images[PRESET_NAMES[3]]
        self.assertGreater(shift_energy(corner, *self.at, 9, 1, 0), 0)
        self.assertGreater(shift_energy(corner, *self.at, 9, 0, 1), 0)
        self.assertEqual(energy_grid(corner, *self.at, 9)[3, 3], 0)

    def test_sobel_kernels_and_ramp(self):
        self.assertEqual(KX.shape, (3, 3))
        np.testing.assert_array_equal(KY, KX.T)
        yy, xx = np.indices((41, 41))
        ix, iy = gradients(xx + 2*yy)
        self.assertAlmostEqual(ix[20, 20], 1)
        self.assertAlmostEqual(iy[20, 20], 2)

    def test_taylor_on_linear_fixture(self):
        yy, xx = np.indices((41, 41))
        ramp = (xx + 2*yy) / 120
        self.assertAlmostEqual(shift_energy(ramp, 20, 20, 7, 1, 1),
                               taylor_energy(ramp, 20, 20, 7, 1, 1), places=12)

    def test_tensor_symmetry_psd_and_eigen_order(self):
        p = pipeline(self.images[PRESET_NAMES[3]])
        m = tensor_at(p.maps, *self.at)
        np.testing.assert_allclose(m, m.T)
        self.assertGreaterEqual(np.linalg.eigvalsh(m).min(), -1e-12)
        self.assertGreaterEqual(p.large[self.at], p.small[self.at])
        self.assertGreaterEqual(p.small[self.at], -1e-12)

    def test_harris_det_trace_and_k(self):
        maps = tensor_maps(self.images[PRESET_NAMES[3]])
        m = tensor_at(maps, *self.at)
        r = harris_response(maps, .05)
        self.assertAlmostEqual(r[self.at], np.linalg.det(m) - .05*np.trace(m)**2)
        self.assertLess(harris_response(maps, .10)[self.at], r[self.at])

    def test_box_gaussian_weighting_differ(self):
        image = self.images[PRESET_NAMES[3]]
        box = tensor_maps(image, 5, 1.2, "Box")
        gaussian = tensor_maps(image, 5, 1.2, "Gaussian")
        self.assertNotAlmostEqual(box[0][self.at], gaussian[0][self.at])

    def test_tensor_entry_is_weighted_gradient_product(self):
        image = self.images[PRESET_NAMES[3]]
        ix, iy = gradients(image)
        weights = integration_kernel(5, 1.2, "Gaussian")
        self.assertAlmostEqual(float(weights.sum()), 1)
        maps = tensor_maps(image)
        self.assertAlmostEqual(maps[1][self.at],
                               np.sum((ix*iy)[18:23, 18:23] * weights))

    def test_shi_tomasi_matches_small_eigenvalue(self):
        maps = tensor_maps(self.images[PRESET_NAMES[3]])
        _, small = eigenvalues(maps)
        np.testing.assert_allclose(shi_tomasi_response(maps), small)

    def test_classification(self):
        self.assertEqual(classify_eigenvalues(0, 0), "Vùng phẳng")
        self.assertEqual(classify_eigenvalues(.1, .001), "Cạnh")
        self.assertEqual(classify_eigenvalues(.1, .04), "Góc")

    def test_nms_threshold_ties_and_border(self):
        a = np.zeros((9, 9))
        a[3, 3] = a[3, 4] = 2
        a[0, 0] = 10
        mask = threshold_nms(a, 1, 3)
        self.assertTrue(mask[3, 3])
        self.assertFalse(mask[3, 4])
        self.assertFalse(mask[0, 0])
        self.assertFalse(threshold_nms(a, 2, 3).any())
        self.assertTrue(nms_decision(a, 1, 3, 3, 3)[0])
        self.assertIn("đồng hạng", nms_decision(a, 1, 3, 3, 4)[1])
        self.assertIn("biên", nms_decision(a, 1, 3, 0, 0)[1])
        self.assertIn("ngưỡng", nms_decision(a, 2, 3, 3, 3)[1])

    def test_rotation_scale_noise_no_nonfinite(self):
        base = self.images[PRESET_NAMES[3]]
        scaled = self.images[PRESET_NAMES[7]]
        self.assertLess(float(scaled.sum()), float(base.sum()))
        self.assertFalse(np.array_equal(scaled, base))
        for name in PRESET_NAMES[5:]:
            p = pipeline(self.images[name])
            self.assertTrue(np.all(np.isfinite(p.harris)))
            threshold = .08 * max(0, float(p.harris.max()))
            self.assertGreaterEqual(int(threshold_nms(p.harris, threshold).sum()), 0)


if __name__ == "__main__":
    unittest.main()
