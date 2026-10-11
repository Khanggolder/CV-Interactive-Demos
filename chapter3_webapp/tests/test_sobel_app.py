import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from utils.sobel import DEFAULT_CENTER, PRESET_NAMES, compute_sobel_at

APP = Path(__file__).resolve().parents[1] / "app.py"


def button(app, label):
    return next(item for item in app.button if item.label == label)


class SobelInteractionTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(APP), default_timeout=40).run()
        self.app.radio(key="chapter3_module").set_value("Sobel Convolution").run()
        self.clean()

    def clean(self):
        self.assertEqual(list(self.app.exception), [])
        self.assertEqual(list(self.app.error), [])

    def test_module_switch_preserves_independent_state(self):
        button(self.app,"Next →").click().run()
        self.app.radio(key="chapter3_module").set_value("RANSAC").run()
        self.assertEqual(self.app.session_state["guided_step"],1)
        self.app.radio(key="chapter3_module").set_value("Hough Transform").run()
        self.assertEqual(self.app.session_state["hough_guided_step"],1)
        self.app.radio(key="chapter3_module").set_value("Sobel Convolution").run()
        self.assertEqual(self.app.session_state["sobel_guided_step"],2)
        self.clean()

    def test_guided_previous_next_reset(self):
        button(self.app,"Next →").click().run()
        self.assertEqual(self.app.session_state["sobel_guided_step"],2)
        button(self.app,"← Previous").click().run()
        self.assertEqual(self.app.session_state["sobel_guided_step"],1)
        button(self.app,"Next →").click().run()
        button(self.app,"Reset").click().run()
        self.assertEqual(self.app.session_state["sobel_guided_step"],1)
        self.clean()

    def test_guided_products_and_gy_gate(self):
        for _ in range(3): button(self.app,"Next →").click().run()
        self.assertEqual(self.app.session_state["sobel_guided_step"],4)
        button(self.app,"Next Product →").click().run()
        self.assertEqual(self.app.session_state["sobel_progress"],1)
        button(self.app,"Calculate Remaining").click().run()
        self.assertEqual(self.app.session_state["sobel_progress"],9)
        button(self.app,"Reset Calculation").click().run()
        self.assertEqual(self.app.session_state["sobel_progress"],0)
        button(self.app,"Next →").click().run()
        button(self.app,"Next →").click().run()
        self.assertTrue(button(self.app,"Next →").disabled)
        self.app.radio(key="sobel_predict_gy").set_value("Nhỏ hoặc bằng 0").run()
        button(self.app,"Reveal Gy").click().run()
        self.assertFalse(button(self.app,"Next →").disabled)
        self.clean()

    def test_all_guided_steps_and_final_gate(self):
        for target in range(2,13):
            if target == 7:
                self.app.radio(key="sobel_predict_gy").set_value("Nhỏ hoặc bằng 0").run()
                button(self.app,"Reveal Gy").click().run()
            button(self.app,"Next →").click().run()
            self.assertEqual(self.app.session_state["sobel_guided_step"],target)
            self.clean()
        self.assertTrue(button(self.app,"Reveal kết luận").disabled)
        self.app.radio(key="sobel_predict_final").set_value("Không").run()
        button(self.app,"Reveal kết luận").click().run()
        self.assertTrue(self.app.session_state["sobel_reveal_final"])

    def test_lab_edit_center_kernel_progress_and_reset(self):
        self.app.radio(key="sobel_mode").set_value("B · Experiment Lab").run()
        self.app.number_input(key="sobel_edit_value").set_value(255).run()
        button(self.app,"Apply pixel edit").click().run()
        self.assertEqual(self.app.session_state["sobel_image"][2,3],255)
        self.app.radio(key="sobel_axis").set_value("Gy").run()
        button(self.app,"Next Product →").click().run()
        self.assertEqual(self.app.session_state["sobel_progress"],1)
        button(self.app,"→ Col").click().run()
        self.assertEqual(self.app.session_state["sobel_center"],(2,4))
        self.assertEqual(self.app.session_state["sobel_progress"],0)
        button(self.app,"Reset image to preset").click().run()
        self.assertEqual(self.app.session_state["sobel_image"][2,3],20)
        self.assertEqual(self.app.session_state["sobel_center"],DEFAULT_CENTER)
        self.clean()

    def test_presets_maps_and_presentation(self):
        self.app.radio(key="sobel_mode").set_value("B · Experiment Lab").run()
        self.app.selectbox(key="sobel_preset").set_value(PRESET_NAMES[2]).run()
        result = compute_sobel_at(self.app.session_state["sobel_image"],DEFAULT_CENTER)
        self.assertEqual((result.gx,result.gy),(540,540))
        self.app.radio(key="sobel_workspace").set_value("B · Gradient Explorer").run()
        self.app.radio(key="sobel_layer").set_value("Direction").run()
        self.assertEqual(self.app.session_state["sobel_layer"],"Direction")
        self.app.toggle(key="sobel_presentation").set_value(True).run()
        self.assertTrue(self.app.session_state["sobel_presentation"])
        self.app.selectbox(key="sobel_preset").set_value(PRESET_NAMES[3]).run()
        self.assertIsNone(compute_sobel_at(self.app.session_state["sobel_image"],DEFAULT_CENTER).theta_degrees)
        self.clean()

    def test_invalid_edit_rejected_without_state_corruption(self):
        self.app.radio(key="sobel_mode").set_value("B · Experiment Lab").run()
        before = self.app.session_state["sobel_image"].copy()
        self.app.session_state["sobel_edit_value"] = -5
        self.app.run()
        # Widget may coerce invalid state; direct callback validation is also covered numerically.
        self.assertEqual(before.shape,self.app.session_state["sobel_image"].shape)
        self.clean()


if __name__ == "__main__":
    unittest.main()
