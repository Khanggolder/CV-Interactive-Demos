import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from utils.fourier import (
    apply_frequency_mask,
    compute_fft,
    create_gaussian_lowpass_mask,
    create_highpass_mask,
    create_lowpass_mask,
    reconstruct_image,
    reconstruction_for_display,
)
from utils.geometry import (
    bilinear_sample_details,
    build_transform_matrix,
    inverse_source_coordinate,
    nearest_sample,
    translation_matrix,
    warp_inverse,
)
from utils.image_ops import (
    adjust_brightness_contrast,
    apply_clahe,
    equalize_histogram,
    gamma_correct,
    load_rgb_image,
    mark_pixel_and_patch,
    to_gray,
)
from utils.noise import (
    add_gaussian_noise,
    add_salt_pepper_noise,
    gaussian_filter,
    median_filter,
)


class ImageOpsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
        self.rgb = np.dstack((self.gray, np.flipud(self.gray), self.gray))

    def test_point_operations_preserve_shape_and_range(self) -> None:
        for image in (self.gray, self.rgb):
            for result in (
                adjust_brightness_contrast(image, 3.0, 100),
                adjust_brightness_contrast(image, 0.1, -100),
                gamma_correct(image, 0.1),
                gamma_correct(image, 5.0),
            ):
                self.assertEqual(result.shape, image.shape)
                self.assertEqual(result.dtype, np.uint8)
                self.assertGreaterEqual(result.min(), 0)
                self.assertLessEqual(result.max(), 255)

    def test_point_operation_examples_match_displayed_math(self) -> None:
        pixel = np.array([100], dtype=np.uint8)
        self.assertEqual(int(adjust_brightness_contrast(pixel, 1.5, 30)[0]), 180)
        self.assertEqual(int(adjust_brightness_contrast(pixel, 3.0, 100)[0]), 255)
        self.assertEqual(int(gamma_correct(pixel, 1.0)[0]), 100)

    def test_gray_and_histogram_operations(self) -> None:
        self.assertEqual(to_gray(self.rgb).shape, self.gray.shape)
        self.assertEqual(equalize_histogram(self.gray).shape, self.gray.shape)
        self.assertEqual(equalize_histogram(self.rgb).shape, self.rgb.shape)
        self.assertEqual(apply_clahe(self.gray, 2.0, 8).shape, self.gray.shape)
        self.assertEqual(apply_clahe(self.rgb, 2.0, 8).shape, self.rgb.shape)

    def test_fourier_low_and_high_pass(self) -> None:
        _, shifted = compute_fft(self.gray)
        low = create_lowpass_mask(self.gray.shape, 4)
        high = create_highpass_mask(self.gray.shape, 4)
        np.testing.assert_allclose(low + high, 1.0)

        reconstructed_all = reconstruct_image(shifted)
        np.testing.assert_allclose(reconstructed_all, self.gray, atol=1e-4)

        reconstructed_low = reconstruct_image(apply_frequency_mask(shifted, low))
        reconstructed_high = reconstruct_image(apply_frequency_mask(shifted, high))
        np.testing.assert_allclose(
            reconstructed_low + reconstructed_high, self.gray, atol=1e-4
        )
        self.assertAlmostEqual(
            float(reconstructed_low.mean()), float(self.gray.mean()), places=4
        )
        self.assertAlmostEqual(float(reconstructed_high.mean()), 0.0, places=4)

        low_display = reconstruction_for_display(reconstructed_low)
        high_display = reconstruction_for_display(reconstructed_high, signed=True)
        for display in (low_display, high_display):
            self.assertEqual(display.shape, self.gray.shape)
            self.assertEqual(display.dtype, np.uint8)

        rounded = reconstruction_for_display(
            np.array([[0.49, 0.51, 254.6]], dtype=np.float32)
        )
        np.testing.assert_array_equal(rounded, np.array([[0, 1, 255]], dtype=np.uint8))

    def test_gaussian_frequency_mask_and_reconstruction(self) -> None:
        shape = (17, 19)
        sigma_f = 3.0
        mask = create_gaussian_lowpass_mask(shape, sigma_f)
        center_y, center_x = shape[0] // 2, shape[1] // 2
        self.assertEqual(mask.dtype, np.float32)
        self.assertAlmostEqual(float(mask[center_y, center_x]), 1.0, places=6)
        self.assertGreaterEqual(float(mask.min()), 0.0)
        self.assertLessEqual(float(mask.max()), 1.0)
        self.assertAlmostEqual(
            float(mask[center_y + 2, center_x + 3]),
            float(mask[center_y - 2, center_x - 3]),
            places=6,
        )
        self.assertGreater(
            float(mask[center_y, center_x + 1]),
            float(mask[center_y, center_x + 4]),
        )

        _, shifted = compute_fft(self.gray)
        reconstructed = reconstruct_image(
            apply_frequency_mask(
                shifted, create_gaussian_lowpass_mask(self.gray.shape, 4.0)
            )
        )
        self.assertTrue(np.isfinite(reconstructed).all())

    def test_pixel_marker_is_rgb_and_does_not_mutate_input(self) -> None:
        original = self.gray.copy()
        marked = mark_pixel_and_patch(self.gray, 8, 8)
        np.testing.assert_array_equal(self.gray, original)
        self.assertEqual(marked.shape, (16, 16, 3))
        self.assertEqual(marked.dtype, np.uint8)

    def test_png_and_jpeg_decode_as_rgb(self) -> None:
        with TemporaryDirectory() as directory:
            for suffix in ("png", "jpg"):
                path = Path(directory) / f"upload.{suffix}"
                Image.fromarray(self.rgb).save(path)
                decoded = load_rgb_image(path)
                self.assertEqual(decoded.shape, self.rgb.shape)
                self.assertEqual(decoded.dtype, np.uint8)


class NoiseOpsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.full((15, 15, 3), 120, dtype=np.uint8)

    def test_noise_is_seeded_and_does_not_mutate_input(self) -> None:
        original = self.image.copy()
        first = add_gaussian_noise(self.image, 25.0, seed=7)
        second = add_gaussian_noise(self.image, 25.0, seed=7)
        different = add_gaussian_noise(self.image, 25.0, seed=8)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, different))
        np.testing.assert_array_equal(self.image, original)
        self.assertEqual(first.shape, self.image.shape)
        self.assertEqual(first.dtype, np.uint8)

    def test_salt_pepper_impulses_and_seed(self) -> None:
        first = add_salt_pepper_noise(self.image, 0.3, seed=11)
        second = add_salt_pepper_noise(self.image, 0.3, seed=11)
        different = add_salt_pepper_noise(self.image, 0.3, seed=12)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, different))
        changed = np.any(first != self.image, axis=2)
        changed_values = first[changed]
        self.assertTrue(np.all((changed_values == 0) | (changed_values == 255)))
        self.assertTrue(np.all(np.all(changed_values == changed_values[:, :1], axis=1)))

    def test_median_removes_isolated_impulse(self) -> None:
        impulse = np.full((7, 7), 80, dtype=np.uint8)
        impulse[3, 3] = 255
        median = median_filter(impulse, 3)
        self.assertEqual(int(median[3, 3]), 80)
        gaussian = gaussian_filter(impulse, 3, 1.0)
        self.assertGreater(int(gaussian[3, 3]), 80)


class GeometryOpsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.arange(25, dtype=np.uint8).reshape(5, 5)

    def test_identity_rotate_scale_and_translate(self) -> None:
        for matrix in (
            build_transform_matrix(self.image.shape, "Rotate", angle_degrees=0),
            build_transform_matrix(self.image.shape, "Scale", scale=1.0),
            build_transform_matrix(self.image.shape, "Translate", tx=0, ty=0),
        ):
            result = warp_inverse(self.image, matrix, "Nearest Neighbor")
            np.testing.assert_array_equal(result, self.image)

    def test_translation_matrix_and_inverse_point(self) -> None:
        matrix = translation_matrix(3, -2)
        np.testing.assert_allclose(
            matrix,
            np.array([[1, 0, 3], [0, 1, -2], [0, 0, 1]], dtype=float),
        )
        source = inverse_source_coordinate(matrix, 10, 5)
        np.testing.assert_allclose(source, (7, 7))

    def test_nearest_and_bilinear_known_values(self) -> None:
        coordinate, nearest = nearest_sample(self.image, 1.6, 2.2)
        self.assertEqual(coordinate, (2, 2))
        self.assertEqual(int(nearest), int(self.image[2, 2]))

        patch = np.array([[0, 10], [20, 30]], dtype=np.uint8)
        details = bilinear_sample_details(patch, 0.25, 0.5)
        self.assertAlmostEqual(float(details["value"]), 12.5, places=6)


if __name__ == "__main__":
    unittest.main()
