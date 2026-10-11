"""Streamlit AppTest regression for LSD navigation, replay and state isolation."""

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from utils.lsd import PRESETS

APP = Path(__file__).resolve().parents[1] / "app.py"


class LSDAppTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(APP), default_timeout=60).run()
        self.app.radio(key="chapter3_module").set_value("LSD Line Segment Detector").run()
        self.clean()

    def clean(self):
        self.assertEqual([e.message for e in self.app.exception], [])
        self.assertEqual([e.message for e in self.app.error], [])

    def advance(self, target):
        while self.app.session_state["lsd_guided_step"] < target:
            step = self.app.session_state["lsd_guided_step"]
            if step == 4:
                self.app.selectbox(key="lsd_guided_seed_predict").set_value((6, 8)).run()
                self.app.button(key="lsd_reveal_seed").click().run()
            if step == 6 and self.app.session_state["lsd_guided_replay"] == 0:
                self.app.button(key="lsd_guided_replay_next").click().run()
            if step == 7 and not self.app.button(key="lsd_guided_replay_complete").disabled:
                self.app.button(key="lsd_guided_replay_complete").click().run()
            if step in (11, 12):
                self.app.radio(key=f"lsd_guided_{step}_predict").set_value("ACCEPT" if step == 11 else "REJECT").run()
                self.app.button(key=f"lsd_guided_{step}_calc_button").click().run()
                self.app.button(key=f"lsd_guided_{step}_decision_button").click().run()
            self.app.button(key="lsd_next").click().run()
            self.clean()

    def to_lab(self, workspace="A · Gradient & Region"):
        self.app.radio(key="lsd_mode").set_value("B · Experiment Lab").run()
        if workspace.startswith("B"):
            self.app.radio(key="lsd_lab_workspace").set_value(workspace).run()
        self.clean()

    def test_six_module_navigation_isolated(self):
        for name in ("RANSAC", "Hough Transform", "Sobel Convolution",
                     "Canny Edge Detector", "Hough Circle Transform"):
            self.app.radio(key="chapter3_module").set_value(name).run()
            self.clean()
        self.app.radio(key="chapter3_module").set_value("LSD Line Segment Detector").run()
        self.assertEqual(self.app.session_state["lsd_guided_step"], 1)
        self.clean()

    def test_guided_fourteen_steps_and_gates(self):
        self.advance(4)
        self.assertTrue(self.app.button(key="lsd_next").disabled)
        self.advance(6)
        self.assertTrue(self.app.button(key="lsd_next").disabled)
        self.app.button(key="lsd_guided_replay_next").click().run()
        self.assertEqual(self.app.session_state["lsd_guided_replay"], 1)
        self.advance(7)
        self.assertTrue(self.app.button(key="lsd_next").disabled)
        self.advance(11)
        self.assertTrue(self.app.button(key="lsd_next").disabled)
        self.advance(12)
        self.assertTrue(self.app.button(key="lsd_next").disabled)
        self.advance(14)
        self.assertTrue(self.app.button(key="lsd_next").disabled)
        self.app.button(key="lsd_previous").click().run()
        self.assertEqual(self.app.session_state["lsd_guided_step"], 13)
        self.app.button(key="lsd_reset").click().run()
        self.assertEqual(self.app.session_state["lsd_guided_step"], 1)
        self.clean()

    def test_guided_pixel_and_replay_idempotent(self):
        self.advance(2)
        self.app.selectbox(key="lsd_guided_pixel_x").set_value(3).run()
        self.assertEqual(self.app.session_state["lsd_guided_pixel_x"], 3)
        self.advance(6)
        self.app.button(key="lsd_guided_replay_next").click().run()
        count = self.app.session_state["lsd_guided_replay"]
        self.app.run()
        self.assertEqual(self.app.session_state["lsd_guided_replay"], count)
        self.app.button(key="lsd_guided_replay_batch").click().run()
        self.assertGreater(self.app.session_state["lsd_guided_replay"], count)
        self.app.button(key="lsd_guided_replay_previous").click().run()
        self.app.button(key="lsd_guided_replay_reset").click().run()
        self.assertEqual(self.app.session_state["lsd_guided_replay"], 0)
        self.clean()

    def test_nfa_not_revealed_before_prediction(self):
        self.advance(11)
        self.assertTrue(self.app.button(key="lsd_guided_11_calc_button").disabled)
        self.assertFalse(self.app.session_state["lsd_guided_11_decision"])
        self.app.radio(key="lsd_guided_11_predict").set_value("REJECT").run()
        self.app.button(key="lsd_guided_11_calc_button").click().run()
        self.assertTrue(self.app.session_state["lsd_guided_11_calc"])
        self.assertTrue(self.app.button(key="lsd_next").disabled)
        self.app.button(key="lsd_guided_11_decision_button").click().run()
        self.assertTrue(self.app.session_state["lsd_guided_11_decision"])
        self.assertFalse(self.app.button(key="lsd_next").disabled)
        self.clean()

    def test_lab_replay_and_manual_seed(self):
        self.to_lab()
        self.app.button(key="lsd_lab_start").click().run()
        self.app.button(key="lsd_lab_replay_next").click().run()
        self.assertEqual(self.app.session_state["lsd_lab_replay"], 1)
        self.app.button(key="lsd_lab_replay_complete").click().run()
        self.assertGreater(self.app.session_state["lsd_lab_replay"], 1)
        self.app.radio(key="lsd_lab_seed_mode").set_value("Manual exploration").run()
        self.assertFalse(self.app.session_state["lsd_lab_started"])
        self.assertEqual(self.app.session_state["lsd_lab_replay"], 0)
        self.app.button(key="lsd_lab_start").click().run()
        self.clean()

    def test_manual_pixel_change_clears_old_replay(self):
        self.to_lab()
        self.app.radio(key="lsd_lab_seed_mode").set_value("Manual exploration").run()
        self.app.button(key="lsd_lab_start").click().run()
        self.app.button(key="lsd_lab_replay_next").click().run()
        self.assertEqual(self.app.session_state["lsd_lab_replay"], 1)
        self.app.selectbox(key="lsd_lab_pixel_x").set_value(7).run()
        self.assertFalse(self.app.session_state["lsd_lab_started"])
        self.assertEqual(self.app.session_state["lsd_lab_replay"], 0)
        self.clean()

    def test_lab_parameter_invalidation(self):
        self.to_lab()
        self.app.button(key="lsd_lab_start").click().run()
        self.app.button(key="lsd_lab_replay_next").click().run()
        self.app.slider(key="lsd_lab_tau").set_value(30.).run()
        self.assertFalse(self.app.session_state["lsd_lab_started"])
        self.assertEqual(self.app.session_state["lsd_lab_replay"], 0)
        self.app.slider(key="lsd_lab_min").set_value(.2).run()
        self.assertEqual(self.app.session_state["lsd_lab_candidate"], 0)
        self.app.selectbox(key="lsd_lab_preset").set_value(PRESETS[2]).run()
        self.assertEqual(self.app.session_state["lsd_lab_preset"], PRESETS[2])
        self.clean()

    def test_lab_candidate_epsilon_and_reset(self):
        self.to_lab("B · Segment & NFA")
        self.app.selectbox(key="lsd_lab_candidate").set_value(1).run()
        self.assertEqual(self.app.session_state["lsd_lab_candidate"], 1)
        self.app.radio(key="lsd_lab_nfa_predict").set_value("REJECT").run()
        self.app.button(key="lsd_lab_nfa_calc_button").click().run()
        self.app.button(key="lsd_lab_nfa_decision_button").click().run()
        self.assertTrue(self.app.session_state["lsd_lab_nfa_decision"])
        self.app.slider(key="lsd_lab_epsilon").set_value(2.).run()
        self.assertFalse(self.app.session_state["lsd_lab_nfa_decision"])
        self.app.button(key="lsd_lab_reset").click().run()
        self.assertEqual(self.app.session_state["lsd_lab_preset"], PRESETS[0])
        self.assertEqual(self.app.session_state["lsd_lab_tau"], 22.5)
        self.clean()

    def test_presentation_and_return_to_previous_module(self):
        self.app.toggle(key="lsd_presentation").set_value(True).run()
        self.advance(3)
        self.to_lab("B · Segment & NFA")
        self.clean()
        self.app.radio(key="chapter3_module").set_value("Hough Circle Transform").run()
        self.clean()
        self.app.radio(key="chapter3_module").set_value("LSD Line Segment Detector").run()
        self.assertEqual(self.app.session_state["lsd_guided_step"], 3)
        self.clean()


if __name__ == "__main__":
    unittest.main()
