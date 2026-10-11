import unittest

import numpy as np

from utils.sobel import (DEFAULT_CENTER, KX, KY, PRESET_NAMES, compute_gradient_maps,
                         compute_sobel_at, create_sobel_presets, edit_pixel, extract_patch,
                         normalize_for_visualization, screen_vector, valid_center,
                         validate_grayscale_image)
from utils.sobel_visualization import matrix_figure, vector_figure


class SobelNumericalTests(unittest.TestCase):
    def setUp(self):
        self.presets = create_sobel_presets()

    def test_masks_are_approved_direct_correlation(self):
        np.testing.assert_array_equal(KX, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        np.testing.assert_array_equal(KY, [[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    def test_approved_patch_and_products(self):
        image = self.presets[PRESET_NAMES[0]]
        patch = extract_patch(image, DEFAULT_CENTER)
        np.testing.assert_array_equal(patch, [[20, 20, 200]] * 3)
        result = compute_sobel_at(image, DEFAULT_CENTER)
        np.testing.assert_array_equal(result.products_x, [[-20, 0, 200], [-40, 0, 400], [-20, 0, 200]])
        np.testing.assert_array_equal(result.products_y, [[-20, -40, -200], [0, 0, 0], [20, 40, 200]])
        np.testing.assert_array_equal(result.row_sums_x, [180, 360, 180])
        np.testing.assert_array_equal(result.row_sums_y, [-260, 0, 260])

    def test_approved_result(self):
        result = compute_sobel_at(self.presets[PRESET_NAMES[0]], DEFAULT_CENTER)
        self.assertEqual((result.gx, result.gy, result.magnitude, result.theta_degrees), (720, 0, 720, 0))

    def test_horizontal(self):
        result = compute_sobel_at(self.presets[PRESET_NAMES[1]], DEFAULT_CENTER)
        self.assertEqual((result.gx, result.gy, result.theta_degrees), (0, 720, 90))

    def test_diagonal(self):
        result = compute_sobel_at(self.presets[PRESET_NAMES[2]], DEFAULT_CENTER)
        np.testing.assert_array_equal(result.patch, [[20,20,200],[20,200,200],[200,200,200]])
        self.assertEqual((result.gx, result.gy), (540,540))
        self.assertAlmostEqual(result.magnitude, np.hypot(540,540))
        self.assertEqual(result.theta_degrees, 45)

    def test_flat_has_undefined_direction(self):
        result = compute_sobel_at(self.presets[PRESET_NAMES[3]], DEFAULT_CENTER)
        self.assertEqual((result.gx,result.gy,result.magnitude), (0,0,0))
        self.assertIsNone(result.theta_degrees)

    def test_reversed_contrast(self):
        result = compute_sobel_at(self.presets[PRESET_NAMES[4]], DEFAULT_CENTER)
        self.assertEqual((result.gx,result.gy,result.theta_degrees), (-720,0,180))

    def test_noisy_is_deterministic(self):
        a = self.presets[PRESET_NAMES[5]]
        b = create_sobel_presets()[PRESET_NAMES[5]]
        np.testing.assert_array_equal(a,b)
        self.assertFalse(np.array_equal(a,self.presets[PRESET_NAMES[0]]))

    def test_independent_numpy_reference_for_every_interior(self):
        for image in self.presets.values():
            for row in range(1,image.shape[0]-1):
                for col in range(1,image.shape[1]-1):
                    patch = image[row-1:row+2,col-1:col+2].astype(np.int64)
                    # Independent separable weighted differences, not product matrix sum.
                    gx = sum((1,2,1)[r] * int(patch[r,2]-patch[r,0]) for r in range(3))
                    gy = sum((1,2,1)[c] * int(patch[2,c]-patch[0,c]) for c in range(3))
                    actual = compute_sobel_at(image,(row,col))
                    self.assertEqual((gx,gy),(actual.gx,actual.gy))
                    self.assertAlmostEqual(actual.magnitude,np.hypot(gx,gy))
                    if gx or gy:
                        self.assertAlmostEqual(actual.theta_degrees,np.degrees(np.arctan2(gy,gx)))
                    else:
                        self.assertIsNone(actual.theta_degrees)

    def test_uint8_does_not_overflow(self):
        image = np.array([[0,0,255]]*3,dtype=np.uint8)
        self.assertEqual(compute_sobel_at(image,(1,1)).gx,1020)
        self.assertEqual(compute_sobel_at(np.fliplr(image),(1,1)).gx,-1020)

    def test_maps_match_pixel_arithmetic_and_nan_borders(self):
        image = self.presets[PRESET_NAMES[5]]
        maps = compute_gradient_maps(image)
        for data in maps.values():
            self.assertEqual(data.shape,image.shape)
            self.assertTrue(np.isnan(data[0]).all())
            self.assertTrue(np.isnan(data[:,-1]).all())
        for row in range(1,4):
            for col in range(1,6):
                result = compute_sobel_at(image,(row,col))
                for layer,value in (("Gx",result.gx),("Gy",result.gy),("Magnitude",result.magnitude)):
                    self.assertAlmostEqual(maps[layer][row,col],value)

    def test_flat_direction_map_is_nan(self):
        maps = compute_gradient_maps(self.presets[PRESET_NAMES[3]])
        self.assertTrue(np.isnan(maps["Direction"]).all())

    def test_valid_centers(self):
        image = np.zeros((3,3),dtype=np.uint8)
        self.assertTrue(valid_center(image,(1,1)))
        self.assertFalse(valid_center(image,(0,1)))
        self.assertFalse(valid_center(image,(1,2)))
        with self.assertRaises(ValueError): extract_patch(image,(0,0))

    def test_invalid_image_and_edit(self):
        for image in (np.zeros((2,3),dtype=int), np.zeros((3,3,2),dtype=int),
                      np.full((3,3),-1), np.full((3,3),256), np.ones((3,3),dtype=float)):
            with self.assertRaises(ValueError): validate_grayscale_image(image)
        image = self.presets[PRESET_NAMES[0]]
        for row,col,value in ((-1,0,1),(1,7,1),(1,1,-1),(1,1,256),(1,1,2.5)):
            with self.assertRaises(ValueError): edit_pixel(image,row,col,value)

    def test_edit_copy_changes_real_response(self):
        original = self.presets[PRESET_NAMES[0]]
        changed = edit_pixel(original,2,4,255)
        self.assertEqual(original[2,4],200)
        self.assertEqual(changed[2,4],255)
        self.assertEqual(compute_sobel_at(changed,DEFAULT_CENTER).gx,830)

    def test_screen_vector_flips_vertical_only(self):
        self.assertEqual(screen_vector(540,540),(540,-540))
        self.assertEqual(screen_vector(-720,0),(-720,0))

    def test_display_normalization_does_not_change_raw(self):
        raw = np.array([[-10.,0.,10.,np.nan]])
        copy = raw.copy()
        shown = normalize_for_visualization(raw,signed=True)
        np.testing.assert_allclose(shown[0,:3],[0,0.5,1])
        self.assertTrue(np.isnan(shown[0,3]))
        np.testing.assert_array_equal(raw,copy)
        with self.assertRaises(ValueError): normalize_for_visualization(raw)

    def test_plot_raw_labels_and_vector_orientation(self):
        maps = compute_gradient_maps(self.presets[PRESET_NAMES[4]])
        figure = matrix_figure(maps["Gx"],"Gx",center=DEFAULT_CENTER,layer="Gx")
        self.assertTrue(any(a.text == "-720" for a in figure.layout.annotations))
        vector = vector_figure(540,540)
        arrow = next(a for a in vector.layout.annotations if a.showarrow)
        self.assertGreater(arrow.x,0)
        self.assertLess(arrow.y,0)


if __name__ == "__main__":
    unittest.main()
