import unittest

import cv2
import numpy as np

from utils.scale_space import (PRESET_NAMES, build_pyramid, detect_extrema,
    dog, filter_contrast, gaussian_blur, gaussian_kernel, incremental_sigma,
    inspect_neighbors, log_dog_comparison, log_from_level, log_response, map_octave, presets,
    replay, scale_curve, semigroup)


class ScaleSpaceNumerics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.images = presets()

    def test_kernel_symmetric_normalized_and_truncated(self):
        k = gaussian_kernel(1.6)
        self.assertEqual(k.shape, (15,15))
        np.testing.assert_allclose(k, k[::-1, ::-1])
        self.assertAlmostEqual(float(k.sum()), 1)

    def test_invalid_sigma(self):
        for sigma in (0,-1,float("nan")):
            with self.assertRaises(ValueError):
                gaussian_kernel(sigma)

    def test_constant_and_reference_blur(self):
        image = np.full((33,33),.4)
        np.testing.assert_allclose(gaussian_blur(image,2.1),image,atol=1e-14)
        rng = np.random.default_rng(21)
        image = rng.random((33,33))
        expected = cv2.filter2D(image,-1,gaussian_kernel(1.4),
                                borderType=cv2.BORDER_REFLECT_101)
        np.testing.assert_allclose(gaussian_blur(image,1.4),expected,atol=1e-12)

    def test_sigma_smoothing_and_semigroup(self):
        image = self.images[PRESET_NAMES[4]]
        self.assertLess(float(np.var(gaussian_blur(image,4))),
                        float(np.var(gaussian_blur(image,.7))))
        r = semigroup(image,1.2,3)
        self.assertAlmostEqual(r["delta"],np.sqrt(9-1.44))
        self.assertLess(r["mae"],1e-4)
        self.assertLess(r["mae"],r["wrong_mae"])
        with self.assertRaises(ValueError):
            incremental_sigma(2,1)

    def test_pyramid_levels_sigma_dimensions_and_transition(self):
        p = build_pyramid(self.images[PRESET_NAMES[0]])
        self.assertEqual([o.gaussians[0].shape for o in p],
                         [(129,129),(65,65),(33,33)])
        for o in p:
            self.assertEqual(len(o.gaussians),6)
            self.assertEqual(len(o.dogs),5)
            self.assertAlmostEqual(o.local_sigmas[1]/o.local_sigmas[0],2**(1/3))
            self.assertAlmostEqual(o.global_sigmas[2],o.local_sigmas[2]*2**o.index)
            for j in range(5):
                np.testing.assert_allclose(o.dogs[j],o.gaussians[j+1]-o.gaussians[j])
                self.assertAlmostEqual(o.increments[j+1]**2,
                                       o.local_sigmas[j+1]**2-o.local_sigmas[j]**2)
        np.testing.assert_array_equal(p[1].gaussians[0],p[0].gaussians[3][::2,::2])
        self.assertEqual(map_octave(5,7,2),(20,28))

    def test_incremental_pyramid_matches_direct_blur_approximately(self):
        image = self.images[PRESET_NAMES[0]]
        octave = build_pyramid(image,octaves=1)[0]
        for j in (1,2,3,4,5):
            direct = gaussian_blur(image,octave.local_sigmas[j])
            self.assertLess(float(np.mean(np.abs(direct-octave.gaussians[j]))),3e-4)

    def test_octave_transition_preserves_global_scale(self):
        p = build_pyramid(self.images[PRESET_NAMES[0]])
        for o in range(2):
            self.assertAlmostEqual(p[o].global_sigmas[3],p[o+1].global_sigmas[0])
            self.assertEqual(p[o+1].gaussians[0].shape,
                             p[o].gaussians[3][::2,::2].shape)

    def test_pyramid_parameter_boundaries(self):
        image = self.images[PRESET_NAMES[0]]
        for kwargs in ({"octaves":5},{"intervals":1},{"base_sigma":0},
                       {"input_sigma":-1}):
            with self.assertRaises(ValueError):
                build_pyramid(image,**kwargs)

    def test_flat_dog_signed_zero(self):
        p = build_pyramid(self.images[PRESET_NAMES[7]],octaves=1)[0]
        for d in p.dogs:
            self.assertLess(float(np.max(np.abs(d))),1e-12)

    def test_input_blur_and_invalid_schedule(self):
        p = build_pyramid(self.images[PRESET_NAMES[0]],input_sigma=.5,octaves=1)
        self.assertAlmostEqual(p[0].increments[0],np.sqrt(1.6**2-.5**2))
        with self.assertRaises(ValueError):
            build_pyramid(self.images[PRESET_NAMES[0]],input_sigma=1.6)

    def test_log_signed_normalization_and_polarity(self):
        bright = self.images[PRESET_NAMES[1]]
        dark = self.images[PRESET_NAMES[2]]
        _,lap,norm = log_response(bright,5)
        self.assertLess(lap[64,64],0)
        np.testing.assert_allclose(norm,25*lap)
        _,dark_lap,_ = log_response(dark,5)
        np.testing.assert_allclose(dark_lap,-lap,atol=1e-12)
        _,flat,flat_norm = log_response(self.images[PRESET_NAMES[7]],2)
        self.assertLess(float(np.max(np.abs(flat))),1e-12)
        self.assertLess(float(np.max(np.abs(flat_norm))),1e-12)

    def test_log_matches_independent_five_point_stencil(self):
        image = self.images[PRESET_NAMES[1]]
        smooth,lap,_ = log_response(image,3)
        row,col = 55,60
        reference = (smooth[row-1,col]+smooth[row+1,col]+
                     smooth[row,col-1]+smooth[row,col+1]-4*smooth[row,col])
        self.assertAlmostEqual(lap[row,col],reference,places=12)

    def test_lab_log_uses_total_local_sigma_not_increment(self):
        octave = build_pyramid(self.images[PRESET_NAMES[0]],octaves=1)[0]
        level = octave.gaussians[2]
        lap,norm = log_from_level(level,octave.local_sigmas[2])
        np.testing.assert_allclose(norm,octave.local_sigmas[2]**2*lap)
        self.assertNotAlmostEqual(octave.local_sigmas[2],octave.increments[2])

    def test_blob_scale_is_measured(self):
        image = self.images[PRESET_NAMES[0]]
        sigmas = np.linspace(.7,14,60)
        peaks = []
        for row,col in ((36,35),(87,60),(56,95)):
            _,norm = scale_curve(image,row,col,sigmas)
            peaks.append(float(sigmas[np.argmax(np.abs(norm))]))
        self.assertLess(peaks[0],peaks[1])
        self.assertLess(peaks[1],peaks[2])

    def test_dog_order_sign_and_approximation(self):
        image = self.images[PRESET_NAMES[1]]
        comparison = log_dog_comparison(image,2,2**(1/3))
        np.testing.assert_allclose(comparison["dog"],comparison["high"]-comparison["low"])
        self.assertLess(comparison["dog"][64,64],0)
        self.assertLess(comparison["mae"],.02)
        self.assertTrue(np.any(comparison["dog"] < 0))
        with self.assertRaises(ValueError):
            dog(np.ones((9,9)),np.ones((10,10)))

    def test_exact_26_neighbor_order_and_replay(self):
        stack = [np.zeros((9,9)) for _ in range(3)]
        stack[1][4,4] = 5
        events = inspect_neighbors(stack,1,4,4)
        self.assertEqual(len(events),26)
        self.assertEqual([sum(e.layer == j for e in events) for j in range(3)], [9,8,9])
        self.assertEqual((events[0].layer,events[0].row,events[0].col),(0,3,3))
        self.assertNotIn((1,4,4),[(e.layer,e.row,e.col) for e in events])
        self.assertEqual(replay(events,0),"Đang kiểm tra")
        self.assertEqual(replay(events,26),"MAX")
        self.assertEqual(detect_extrema(stack),[(1,4,4,5.0,"MAX")])
        stack[1][4,4] = -5
        self.assertEqual(replay(inspect_neighbors(stack,1,4,4),26),"MIN")
        stack[1][4,4] = 0
        self.assertEqual(detect_extrema(stack),[])

    def test_replay_rejection_occurs_on_first_invalidating_comparison(self):
        stack = [np.zeros((9,9)) for _ in range(3)]
        stack[1][4,4] = 2
        stack[0][3,3] = 3
        stack[0][3,4] = 1
        events = inspect_neighbors(stack,1,4,4)
        self.assertEqual(replay(events,1),"Còn khả năng MIN")
        self.assertEqual(replay(events,2),"Bị loại")
        self.assertEqual(replay(events,26),"Bị loại")

    def test_equal_value_plateau_rejected(self):
        stack = [np.zeros((9,9)) for _ in range(3)]
        stack[1][4,4] = 5
        stack[1][4,5] = 5
        self.assertNotIn((1,4,4,5.0,"MAX"),detect_extrema(stack))
        self.assertEqual(replay(inspect_neighbors(stack,1,4,4),26),"Bị loại")

    def test_rejection_borders_and_contrast(self):
        stack = [np.zeros((9,9)) for _ in range(3)]
        stack[1][0,0] = 10
        stack[1][4,4] = .02
        with self.assertRaises(ValueError):
            inspect_neighbors(stack,1,0,0)
        with self.assertRaises(ValueError):
            inspect_neighbors(stack,0,4,4)
        found = detect_extrema(stack)
        self.assertEqual(len(found),1)
        self.assertEqual(len(filter_contrast(found,.01)),1)
        self.assertEqual(len(filter_contrast(found,.03)),0)

    def test_presets_repeatable_and_real_extrema(self):
        a,b = presets(),presets()
        self.assertEqual(len(a),8)
        for name in PRESET_NAMES:
            np.testing.assert_array_equal(a[name],b[name])
        p = build_pyramid(a[PRESET_NAMES[0]])
        self.assertGreater(sum(len(detect_extrema(o.dogs)) for o in p),0)
        flat = build_pyramid(a[PRESET_NAMES[7]],octaves=1)
        self.assertEqual(detect_extrema(flat[0].dogs),[])


if __name__ == "__main__":
    unittest.main()
