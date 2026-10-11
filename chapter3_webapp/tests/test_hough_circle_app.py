"""Streamlit state, teaching gates and module-isolation checks."""

import unittest
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from utils.hough_circle import PRESETS, candidate_center, make_grid, preset_dataset

APP = Path(__file__).resolve().parents[1] / "app.py"


class HoughCircleAppTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(APP), default_timeout=60).run()
        self.app.radio(key="chapter3_module").set_value("Hough Circle Transform").run()
        self.clean()

    def clean(self):
        self.assertEqual([e.message for e in self.app.exception], [])
        self.assertEqual([e.message for e in self.app.error], [])

    def advance(self, target):
        while self.app.session_state["circle_guided_step"] < target:
            step = self.app.session_state["circle_guided_step"]
            if step in (4, 11):
                self.app.radio(key=f"circle_guided_predict_{step}").set_value("2" if step == 4 else "Có").run()
                self.app.button(key=f"circle_reveal_{step}").click().run()
            self.app.button(key="circle_next").click().run()
            self.clean()

    def to_lab_vote(self):
        self.app.radio(key="circle_mode").set_value("B · Experiment Lab").run()
        self.app.radio(key="circle_lab_workspace").set_value("B · Circle Voting").run()
        self.clean()

    def test_navigation_and_existing_modules(self):
        for name in ("RANSAC", "Hough Transform", "Sobel Convolution", "Canny Edge Detector"):
            self.app.radio(key="chapter3_module").set_value(name).run()
            self.clean()
        self.app.radio(key="chapter3_module").set_value("Hough Circle Transform").run()
        self.assertEqual(self.app.session_state["circle_guided_step"], 1)

    def test_guided_all_steps_gates_previous_reset(self):
        self.advance(4)
        self.assertTrue(self.app.button(key="circle_next").disabled)
        self.advance(11)
        self.assertTrue(self.app.button(key="circle_next").disabled)
        self.advance(14)
        self.assertTrue(self.app.button(key="circle_next").disabled)
        self.app.button(key="circle_previous").click().run()
        self.assertEqual(self.app.session_state["circle_guided_step"], 13)
        self.app.button(key="circle_reset").click().run()
        self.assertEqual(self.app.session_state["circle_guided_step"], 1)
        self.clean()

    def test_candidate_angle_radius_synchronization(self):
        self.advance(3)
        self.app.slider(key="circle_guided_phi_control").set_value(30.).run()
        point = preset_dataset(PRESETS[0]).points[0].xy
        expected = candidate_center(point, 2, 30)
        self.assertAlmostEqual(np.linalg.norm(expected-point), 2)
        self.advance(7)
        self.app.slider(key="circle_guided_radius_control").set_value(3.).run()
        self.assertEqual(self.app.session_state["circle_guided_radius"], 3.)
        self.clean()

    def test_guided_incremental_vote_rerun_and_batch(self):
        self.advance(9)
        self.app.button(key="circle_guided_vote_next").click().run()
        self.assertEqual(int(self.app.session_state["circle_guided_votes"].counts.sum()), 1)
        self.app.run()
        self.assertEqual(int(self.app.session_state["circle_guided_votes"].counts.sum()), 1)
        self.app.button(key="circle_guided_vote_point").click().run()
        self.assertGreater(int(self.app.session_state["circle_guided_votes"].counts.sum()), 1)
        self.app.button(key="circle_guided_vote_all").click().run()
        self.assertEqual(int(self.app.session_state["circle_guided_votes"].counts.sum()), 216)
        self.app.button(key="circle_guided_vote_reset").click().run()
        self.assertEqual(int(self.app.session_state["circle_guided_votes"].counts.sum()), 0)
        self.clean()

    def test_lab_presets_and_custom_point_ids(self):
        self.app.radio(key="circle_mode").set_value("B · Experiment Lab").run()
        self.app.selectbox(key="circle_lab_preset").set_value(PRESETS[3]).run()
        self.assertEqual(len(self.app.session_state["circle_lab_dataset"].points), 10)
        self.app.number_input(key="circle_add_x").set_value(2.25).run()
        self.app.number_input(key="circle_add_y").set_value(-.75).run()
        self.app.button(key="circle_add_point").click().run()
        dataset = self.app.session_state["circle_lab_dataset"]
        self.assertEqual(len(dataset.points), 11)
        self.assertFalse(dataset.points[-1].reference)
        self.app.button(key="circle_remove_point").click().run()
        self.assertEqual(len(self.app.session_state["circle_lab_dataset"].points), 10)
        self.app.selectbox(key="circle_lab_preset").set_value(PRESETS[4]).run()
        self.assertEqual(self.app.session_state["circle_lab_polarity"], "Outward")
        self.clean()

    def test_lab_votes_modes_and_parameter_invalidation(self):
        self.to_lab_vote()
        self.app.button(key="circle_vote_next").click().run()
        self.assertEqual(int(self.app.session_state["circle_lab_votes"].counts.sum()), 1)
        self.app.button(key="circle_vote_previous").click().run()
        self.assertEqual(int(self.app.session_state["circle_lab_votes"].counts.sum()), 0)
        self.app.button(key="circle_vote_point").click().run()
        self.assertGreater(int(self.app.session_state["circle_lab_votes"].counts.sum()), 1)
        self.app.button(key="circle_vote_all").click().run()
        self.assertEqual(int(self.app.session_state["circle_lab_votes"].counts.sum()), 216)
        self.app.slider(key="circle_lab_radius_control").set_value(1.5).run()
        self.assertEqual(int(self.app.session_state["circle_lab_votes"].counts.sum()), 0)
        self.app.button(key="circle_vote_all").click().run()
        self.app.select_slider(key="circle_lab_center_width").set_value(1.).run()
        self.assertEqual(int(self.app.session_state["circle_lab_votes"].counts.sum()), 0)
        self.app.radio(key="circle_lab_voting").set_value("Known gradient").run()
        self.app.button(key="circle_vote_all").click().run()
        self.assertEqual(int(self.app.session_state["circle_lab_votes"].counts.sum()), 6)
        self.app.radio(key="circle_lab_voting").set_value("Unknown polarity").run()
        self.assertEqual(int(self.app.session_state["circle_lab_votes"].counts.sum()), 0)
        self.clean()

    def test_unknown_radius_layer_and_presentation(self):
        self.to_lab_vote()
        self.app.radio(key="circle_lab_search").set_value("Unknown radius").run()
        self.assertGreater(self.app.session_state["circle_lab_votes"].counts.shape[0], 1)
        self.app.select_slider(key="circle_lab_layer").set_value(2).run()
        self.app.button(key="circle_vote_all").click().run()
        self.assertGreater(int(self.app.session_state["circle_lab_votes"].counts.sum()), 0)
        self.app.toggle(key="circle_presentation").set_value(True).run()
        self.assertTrue(self.app.session_state["circle_presentation"])
        self.clean()

    def test_lab_wrong_radius_no_intersection(self):
        self.app.radio(key="circle_mode").set_value("B · Experiment Lab").run()
        self.app.slider(key="circle_lab_visible").set_value(2).run()
        self.app.slider(key="circle_lab_radius_control").set_value(.5).run()
        self.assertEqual(self.app.session_state["circle_lab_radius"], .5)
        self.clean()

    def test_custom_zero_gradient_no_vote(self):
        self.app.radio(key="circle_mode").set_value("B · Experiment Lab").run()
        self.app.toggle(key="circle_custom_gradient").set_value(True).run()
        self.app.button(key="circle_add_point").click().run()
        point = self.app.session_state["circle_lab_dataset"].points[-1]
        self.assertEqual(point.gradient, (0., 0.))
        self.app.radio(key="circle_lab_workspace").set_value("B · Circle Voting").run()
        self.app.radio(key="circle_lab_voting").set_value("Known gradient").run()
        self.app.button(key="circle_vote_point").click().run()
        self.assertEqual(int(self.app.session_state["circle_lab_votes"].counts.sum()), 0)
        self.clean()


if __name__ == "__main__":
    unittest.main()
