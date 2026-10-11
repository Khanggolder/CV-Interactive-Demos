import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"


def button(app, label):
    return next(item for item in app.button if item.label == label)


class HoughInteractionTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(APP), default_timeout=30).run()
        self.app.radio(key="chapter3_module").set_value("Hough Transform").run()
        self.assert_clean()

    def assert_clean(self):
        self.assertEqual(list(self.app.exception), [])
        self.assertEqual(list(self.app.error), [])

    def test_module_switch_and_ransac_regression(self):
        self.app.radio(key="chapter3_module").set_value("RANSAC").run()
        self.assert_clean()
        self.assertEqual(self.app.session_state["guided_step"], 1)
        button(self.app, "Next →").click().run()
        self.assertEqual(self.app.session_state["guided_step"], 2)
        self.app.radio(key="chapter3_module").set_value("Hough Transform").run()
        self.assertEqual(self.app.session_state["hough_guided_step"], 1)
        self.app.radio(key="chapter3_module").set_value("RANSAC").run()
        self.assertEqual(self.app.session_state["guided_step"], 2)

    def test_guided_prediction_gate_and_theta(self):
        for _ in range(4):
            button(self.app, "Next →").click().run()
        self.assertEqual(self.app.session_state["hough_guided_step"], 5)
        self.assertTrue(button(self.app, "Next →").disabled)
        self.app.slider(key="hough_guided_theta").set_value(63.4).run()
        self.assertAlmostEqual(self.app.session_state["hough_guided_theta"], 63.4)
        self.app.radio(key="hough_guided_prediction").set_value("Một line đi qua cả hai điểm").run()
        button(self.app, "Reveal giải thích").click().run()
        self.assertFalse(button(self.app, "Next →").disabled)
        button(self.app, "Next →").click().run()
        self.assertEqual(self.app.session_state["hough_guided_step"], 6)
        self.assert_clean()

    def test_guided_votes_and_reset(self):
        self.app.session_state["hough_guided_step"] = 9
        self.app.run()
        button(self.app, "Next Vote").click().run()
        self.assertEqual(int(self.app.session_state["hough_guided_votes"].counts.sum()), 1)
        self.app.run()
        self.assertEqual(int(self.app.session_state["hough_guided_votes"].counts.sum()), 1)
        button(self.app, "Vote Selected Point").click().run()
        self.assertEqual(int(self.app.session_state["hough_guided_votes"].counts.sum()), 18)
        button(self.app, "Next →").click().run()
        button(self.app, "Vote All 9").click().run()
        self.assertEqual(int(self.app.session_state["hough_guided_votes"].counts.sum()), 162)
        button(self.app, "Reset").click().run()
        self.assertEqual(int(self.app.session_state["hough_guided_votes"].counts.sum()), 0)
        self.assert_clean()

    def test_guided_tutorial_reaches_limit_step(self):
        for step in range(2, 13):
            if step == 6:
                self.app.radio(key="hough_guided_prediction").set_value("Một line đi qua cả hai điểm").run()
                button(self.app, "Reveal giải thích").click().run()
            button(self.app, "Next →").click().run()
            self.assertEqual(self.app.session_state["hough_guided_step"], step)
            self.assert_clean()
        self.assertTrue(button(self.app, "Next →").disabled)

    def test_lab_points_votes_resolution_and_preset(self):
        self.app.radio(key="hough_mode").set_value("B · Experiment Lab").run()
        self.app.radio(key="hough_lab_workspace").set_value("B · Accumulator Voting").run()
        self.assertEqual(self.app.slider(key="hough_lab_threshold").value, 4)
        button(self.app, "Next Vote").click().run()
        self.assertEqual(int(self.app.session_state["hough_lab_votes"].counts.sum()), 1)
        button(self.app, "Vote Selected").click().run()
        self.assertEqual(int(self.app.session_state["hough_lab_votes"].counts.sum()), 18)
        button(self.app, "Vote All").click().run()
        self.assertEqual(int(self.app.session_state["hough_lab_votes"].counts.sum()), 162)
        self.app.select_slider(key="hough_lab_theta_width").set_value(30.0).run()
        self.assertEqual(int(self.app.session_state["hough_lab_votes"].counts.sum()), 0)
        button(self.app, "Vote All").click().run()
        self.assertEqual(int(self.app.session_state["hough_lab_votes"].counts.sum()), 54)
        button(self.app, "Reset Accumulator").click().run()
        self.assertEqual(int(self.app.session_state["hough_lab_votes"].counts.sum()), 0)
        self.app.selectbox(key="hough_lab_preset").set_value("C · Hai line cạnh tranh").run()
        button(self.app, "Nạp preset / Reset points").click().run()
        self.assertEqual(len(self.app.session_state["hough_lab_points"]), 12)
        button(self.app, "Về ví dụ mặc định").click().run()
        self.assertEqual(len(self.app.session_state["hough_lab_points"]), 9)
        self.assert_clean()

    def test_custom_point_and_presentation(self):
        self.app.radio(key="hough_mode").set_value("B · Experiment Lab").run()
        self.app.toggle(key="hough_presentation").set_value(True).run()
        self.assertTrue(self.app.session_state["hough_presentation"])
        self.app.number_input(key="hough_add_x").set_value(12.0).run()
        self.app.number_input(key="hough_add_y").set_value(-7.0).run()
        button(self.app, "Thêm point").click().run()
        self.assertEqual(len(self.app.session_state["hough_lab_points"]), 10)
        self.assertGreater(self.app.session_state["hough_lab_votes"].counts.shape[0], 20)
        self.app.radio(key="hough_lab_workspace").set_value("B · Accumulator Voting").run()
        button(self.app, "Next Vote").click().run()
        self.assertEqual(int(self.app.session_state["hough_lab_votes"].counts.sum()), 1)
        button(self.app, "Bỏ custom point đang chọn").click().run()
        self.assertEqual(len(self.app.session_state["hough_lab_points"]), 9)
        self.assertEqual(int(self.app.session_state["hough_lab_votes"].counts.sum()), 0)
        self.assert_clean()

    def test_duplicate_custom_coordinates_do_not_unlock_base_point_deletion(self):
        self.app.radio(key="hough_mode").set_value("B · Experiment Lab").run()
        self.app.number_input(key="hough_add_x").set_value(1.0).run()
        self.app.number_input(key="hough_add_y").set_value(2.0).run()
        button(self.app, "Thêm point").click().run()
        self.assertEqual(len(self.app.session_state["hough_lab_points"]), 10)
        self.app.selectbox(key="hough_lab_point").set_value(0).run()
        self.assertTrue(button(self.app, "Bỏ custom point đang chọn").disabled)
        self.app.selectbox(key="hough_lab_point").set_value(9).run()
        button(self.app, "Bỏ custom point đang chọn").click().run()
        self.assertEqual(len(self.app.session_state["hough_lab_points"]), 9)
        self.assertEqual(self.app.session_state["hough_lab_points"][0], (1.0, 2.0))
        self.assert_clean()


if __name__ == "__main__":
    unittest.main()
