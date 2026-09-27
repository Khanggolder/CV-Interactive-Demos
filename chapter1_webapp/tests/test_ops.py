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
)
from utils.image_ops import (
    adjust_brightness_contrast,
    apply_clahe,
    equalize_histogram,
    gamma_correct,
    load_rgb_image,
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
        for mask in (low, high):
            reconstructed = reconstruct_image(apply_frequency_mask(shifted, mask))
            self.assertEqual(reconstructed.shape, self.gray.shape)
            self.assertEqual(reconstructed.dtype, np.uint8)

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
