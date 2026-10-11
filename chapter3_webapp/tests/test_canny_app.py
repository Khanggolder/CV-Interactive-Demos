import unittest
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from utils.canny import PRESETS, canny_stages, preset_image

APP = Path(__file__).resolve().parents[1] / "app.py"


def button(app, key):
    return app.button(key=key)


class CannyAppTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(APP), default_timeout=60).run()
        self.app.radio(key="chapter3_module").set_value("Canny Edge Detector").run()
        self.clean()

    def clean(self):
        self.assertEqual([x.message for x in self.app.exception], [])
        self.assertEqual([x.message for x in self.app.error], [])

    def advance(self, target):
        while self.app.session_state["canny_guided_step"] < target:
            step = self.app.session_state["canny_guided_step"]
            if step in (8, 9, 11):
                self.app.radio(key=f"canny_predict_{step}").set_value("Chưa chắc").run()
                button(self.app, f"canny_reveal_button_{step}").click().run()
            button(self.app, "canny_next").click().run()
            self.clean()

    def test_guided_all_steps_and_gates(self):
        self.advance(8)
        self.assertTrue(button(self.app, "canny_next").disabled)
        self.assertEqual(tuple(self.app.session_state["canny_point"]), (130, 116))
        self.advance(9)
        self.assertTrue(button(self.app, "canny_next").disabled)
        self.assertEqual(tuple(self.app.session_state["canny_point"]), (130, 115))
        self.advance(14)
        self.app.radio(key="canny_predict_14").set_value("Không").run()
        button(self.app, "canny_reveal_button_14").click().run()
        self.assertTrue(self.app.session_state["canny_reveal_14"])
        button(self.app, "canny_previous").click().run()
        self.assertEqual(self.app.session_state["canny_guided_step"], 13)
        button(self.app, "canny_reset").click().run()
        self.assertEqual(self.app.session_state["canny_guided_step"], 1)
        self.clean()

    def test_sigma_and_threshold_invalidation(self):
        self.advance(8)
        self.app.radio(key="canny_predict_8").set_value("KEEP").run()
        button(self.app, "canny_reveal_button_8").click().run()
        self.assertTrue(self.app.session_state["canny_reveal_8"])
        self.app.slider(key="canny_sigma").set_value(2.0).run()
        with self.assertRaises(KeyError):
            self.app.session_state["canny_reveal_8"]
        self.assertEqual(self.app.session_state["canny_bfs_step"], 0)
        self.clean()

    def test_lab_stage_preset_and_bfs(self):
        self.app.radio(key="canny_mode").set_value("B · Experiment Lab").run()
        self.app.selectbox(key="canny_preset").set_value(PRESETS[3]).run()
        self.app.selectbox(key="canny_lab_stage").set_value("Double Threshold").run()
        self.assertEqual(self.app.session_state["canny_lab_stage"], "Double Threshold")
        self.app.radio(key="canny_workspace").set_value("B · NMS & Hysteresis Explorer").run()
        self.app.radio(key="canny_explorer").set_value("Hysteresis BFS").run()
        events = canny_stages(preset_image(PRESETS[3]))["replay"].events
        checkpoints = [i for i, (_, added) in enumerate(events, 1) if added]
        button(self.app, "canny_bfs_1").click().run()
        self.assertEqual(self.app.session_state["canny_bfs_step"], checkpoints[0])
        button(self.app, "canny_bfs_20").click().run()
        self.assertEqual(self.app.session_state["canny_bfs_step"], checkpoints[20])
        button(self.app, "canny_bfs_-1").click().run()
        self.assertEqual(self.app.session_state["canny_bfs_step"], checkpoints[19])
        self.app.run()
        self.assertEqual(self.app.session_state["canny_bfs_step"], checkpoints[19])
        total = len(events)
        button(self.app, "canny_bfs_-2").click().run()
        self.assertEqual(self.app.session_state["canny_bfs_step"], total)
        button(self.app, "canny_bfs_0").click().run()
        self.assertEqual(self.app.session_state["canny_bfs_step"], 0)
        self.app.selectbox(key="canny_preset").set_value(PRESETS[4]).run()
        self.assertEqual(self.app.session_state["canny_bfs_step"], 0)
        self.clean()

    def test_nms_quick_examples_and_point_sync(self):
        self.app.radio(key="canny_mode").set_value("B · Experiment Lab").run()
        self.app.radio(key="canny_workspace").set_value("B · NMS & Hysteresis Explorer").run()
        button(self.app, "canny_example_keep").click().run()
        self.assertEqual(tuple(self.app.session_state["canny_point"]), (130, 116))
        button(self.app, "canny_example_suppress").click().run()
        self.assertEqual(tuple(self.app.session_state["canny_point"]), (130, 115))
        self.assertEqual(self.app.session_state["canny_nms_point_col"], 115)
        self.app.number_input(key="canny_nms_point_col").set_value(117).run()
        self.assertEqual(tuple(self.app.session_state["canny_point"]), (130, 117))
        self.clean()

    def test_invalid_threshold_and_recovery(self):
        self.app.radio(key="canny_mode").set_value("B · Experiment Lab").run()
        self.app.number_input(key="canny_low").set_value(.5).run()
        self.assertTrue(self.app.error)
        self.app.number_input(key="canny_low").set_value(.1).run()
        self.clean()

    def test_presentation_and_module_regression(self):
        self.app.toggle(key="canny_presentation").set_value(True).run()
        self.assertTrue(self.app.session_state["canny_presentation"])
        for module in ("Sobel Convolution", "Hough Transform", "RANSAC", "Canny Edge Detector"):
            self.app.radio(key="chapter3_module").set_value(module).run()
            self.clean()
        self.assertEqual(self.app.session_state["canny_guided_step"], 1)

    def test_same_original_across_stage_selection(self):
        self.app.radio(key="canny_mode").set_value("B · Experiment Lab").run()
        self.app.selectbox(key="canny_preset").set_value(PRESETS[2]).run()
        signature = self.app.session_state["canny_signature"]
        point = tuple(self.app.session_state["canny_point"])
        for stage in ("Smoothed", "Gx", "Gy", "Magnitude", "NMS", "Final Edge Map"):
            self.app.selectbox(key="canny_lab_stage").set_value(stage).run()
            self.assertEqual(self.app.session_state["canny_signature"], signature)
            self.assertEqual(tuple(self.app.session_state["canny_point"]), point)
            self.clean()


if __name__ == "__main__":
    unittest.main()
