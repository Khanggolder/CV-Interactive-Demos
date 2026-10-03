import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"


class StreamlitAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = AppTest.from_file(str(APP), default_timeout=20).run()
        self.assert_clean()

    def assert_clean(self) -> None:
        self.assertEqual(list(self.app.exception), [])
        self.assertEqual(list(self.app.error), [])

    def navigate(self, page: str) -> None:
        self.app.radio(key="nav_page").set_value(page).run()
        self.assert_clean()

    def test_all_pages_and_interactive_extremes(self) -> None:
        self.app.radio(key="pixel_mode").set_value("Grayscale").run()
        self.assert_clean()

        self.navigate("2. Brightness & Contrast")
        self.app.slider(key="bc_alpha").set_value(3.0)
        self.app.slider(key="bc_beta").set_value(100).run()
        self.assert_clean()
        self.app.slider(key="bc_alpha").set_value(0.1)
        self.app.slider(key="bc_beta").set_value(-100).run()
        self.assert_clean()
        self.app.button[0].click().run()
        self.assert_clean()
        self.assertEqual(self.app.slider(key="bc_alpha").value, 1.0)
        self.assertEqual(self.app.slider(key="bc_beta").value, 0)

        self.navigate("3. Gamma")
        self.assertEqual(
            self.app.radio(key="gamma_mode").value, "Grayscale intensity"
        )
        self.app.slider(key="gamma_value").set_value(0.1).run()
        self.assert_clean()
        self.app.radio(key="gamma_mode").set_value("RGB từng kênh").run()
        self.assert_clean()
        self.app.slider(key="gamma_value").set_value(5.0).run()
        self.assert_clean()

        self.navigate("4. Histogram")
        self.app.radio(key="hist_operation").set_value("Histogram Equalization").run()
        self.assert_clean()
        self.app.radio(key="hist_operation").set_value("CLAHE").run()
        self.assert_clean()

        self.navigate("5. Noise & Filtering")
        self.app.radio(key="noise_type").set_value("Salt-and-pepper noise").run()
        self.assert_clean()
        self.app.checkbox(key="noise_reveal").set_value(True).run()
        self.assert_clean()
        self.app.slider(key="noise_density").set_value(0.12).run()
        self.assert_clean()
        self.assertFalse(self.app.checkbox(key="noise_reveal").value)
        self.app.radio(key="noise_type").set_value("Gaussian noise").run()
        self.assert_clean()
        self.app.checkbox(key="noise_reveal").set_value(True).run()
        self.assert_clean()

        self.navigate("6. Geometric Transform")
        self.app.radio(key="geometry_interpolation").set_value("Bilinear").run()
        self.assert_clean()
        self.app.checkbox(key="geometry_reveal").set_value(True).run()
        self.assert_clean()
        self.app.slider(key="geometry_angle").set_value(45).run()
        self.assert_clean()
        self.assertFalse(self.app.checkbox(key="geometry_reveal").value)
        self.app.radio(key="geometry_operation").set_value("Scale").run()
        self.assert_clean()
        self.app.checkbox(key="geometry_reveal").set_value(True).run()
        self.assert_clean()
        self.app.radio(key="geometry_operation").set_value("Translate").run()
        self.assert_clean()
        self.app.checkbox(key="geometry_reveal").set_value(True).run()
        self.assert_clean()

        self.navigate("7. Fourier")
        self.app.selectbox(key="sample_name").set_value("Nhiều texture / tần số").run()
        self.assert_clean()
        self.app.radio(key="fourier_filter").set_value("Ideal Low-pass").run()
        self.assert_clean()
        self.app.checkbox(key="fourier_reveal").set_value(True).run()
        self.assert_clean()
        self.app.radio(key="fourier_filter").set_value("Gaussian Low-pass").run()
        self.assert_clean()
        self.app.checkbox(key="fourier_reveal").set_value(True).run()
        self.assert_clean()
        self.app.radio(key="fourier_filter").set_value("Compare Ideal vs Gaussian").run()
        self.assert_clean()
        self.app.checkbox(key="fourier_reveal").set_value(True).run()
        self.assert_clean()
        current_sigma = self.app.slider(key="fourier_sigma").value
        self.app.slider(key="fourier_sigma").set_value(current_sigma + 1.0).run()
        self.assert_clean()
        self.assertFalse(self.app.checkbox(key="fourier_reveal").value)
        self.app.radio(key="fourier_filter").set_value("Ideal High-pass").run()
        self.assert_clean()
        self.app.checkbox(key="fourier_reveal").set_value(True).run()
        self.assert_clean()
        self.app.selectbox(key="sample_name").set_value("Cạnh rõ").run()
        self.assert_clean()
        self.assertFalse(self.app.checkbox(key="fourier_reveal").value)

        self.navigate("8. Convolution Kernels")
        self.app.selectbox(key="sample_name").set_value("Tương phản thấp").run()
        self.assert_clean()


if __name__ == "__main__":
    unittest.main()
