import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from utils.fourier import (
    apply_frequency_mask,
    compute_fft,
    create_highpass_mask,
    create_lowpass_mask,
    reconstruct_image,
    reconstruction_for_display,
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


if __name__ == "__main__":
    unittest.main()
