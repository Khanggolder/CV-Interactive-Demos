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

        self.navigate("3. Gamma")
        self.app.slider(key="gamma_value").set_value(0.1).run()
        self.assert_clean()
        self.app.slider(key="gamma_value").set_value(5.0).run()
        self.assert_clean()

        self.navigate("4. Histogram")
        self.app.radio(key="hist_operation").set_value("Histogram Equalization").run()
        self.assert_clean()
        self.app.radio(key="hist_operation").set_value("CLAHE").run()
        self.assert_clean()

        self.navigate("5. Fourier")
        self.app.selectbox(key="sample_name").set_value("Nhiều texture / tần số").run()
        self.assert_clean()
        self.app.radio(key="fourier_filter").set_value("Ideal Low-pass").run()
        self.assert_clean()
        self.app.checkbox(key="fourier_reveal").set_value(True).run()
        self.assert_clean()
        self.app.radio(key="fourier_filter").set_value("Ideal High-pass").run()
        self.assert_clean()
        self.app.checkbox(key="fourier_reveal").set_value(True).run()
        self.assert_clean()

        self.navigate("6. Convolution Kernels")
        self.app.selectbox(key="sample_name").set_value("Tương phản thấp").run()
        self.assert_clean()


if __name__ == "__main__":
    unittest.main()
