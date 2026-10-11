import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"


class StreamlitSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = AppTest.from_file(str(APP), default_timeout=30).run()
        self.assert_clean()

    def assert_clean(self) -> None:
        self.assertEqual(list(self.app.exception), [])
        self.assertEqual(list(self.app.error), [])

    def test_guided_navigation_and_reveal(self) -> None:
        self.assertEqual(self.app.session_state["guided_step"], 1)
        next_button = next(button for button in self.app.button if button.label == "Next →")
        next_button.click().run()
        self.assert_clean()
        self.assertEqual(self.app.session_state["guided_step"], 2)
        self.assertTrue(next(button for button in self.app.button if button.label == "Next →").disabled)
        self.app.radio(key="guided_prediction").set_value(
            "Không, sample có thể chứa outlier"
        ).run()
        self.assert_clean()
        self.assertFalse(next(button for button in self.app.button if button.label == "Next →").disabled)
        next(button for button in self.app.button if button.label == "Next →").click().run()
        next(button for button in self.app.button if button.label == "Next →").click().run()
        next(button for button in self.app.button if button.label == "Next →").click().run()
        self.assertEqual(self.app.session_state["guided_step"], 5)
        self.assertTrue(next(button for button in self.app.button if button.label == "Next →").disabled)
        self.assertTrue(next(button for button in self.app.button if button.label == "Reveal phân loại").disabled)
        self.app.radio(key="guided_membership_prediction").set_value("Ít").run()
        self.assert_clean()
        self.assertFalse(next(button for button in self.app.button if button.label == "Reveal phân loại").disabled)
        next(button for button in self.app.button if button.label == "Reveal phân loại").click().run()
        self.assert_clean()
        self.assertTrue(self.app.session_state["guided_reveal"])
        self.assertFalse(next(button for button in self.app.button if button.label == "Next →").disabled)

    def test_experiment_one_and_multiple_iterations(self) -> None:
        self.app.radio(key="app_mode").set_value("B · Experiment Lab").run()
        self.assert_clean()
        self.app.radio(key="lab_prediction").set_value("Thấp").run()
        self.assert_clean()
        next(button for button in self.app.button if button.label == "Chạy 1 iteration").click().run()
        self.assert_clean()
        self.assertEqual(self.app.session_state["lab_completed"], 1)
        batch = self.app.slider(key="lab_batch").value
        next(button for button in self.app.button if button.label.startswith("Chạy thêm")).click().run()
        self.assert_clean()
        self.assertEqual(self.app.session_state["lab_completed"], 1 + batch)
        self.app.checkbox(key="lab_compare_reveal").set_value(True).run()
        self.assert_clean()
        next(button for button in self.app.button if button.label == "Reset experiment").click().run()
        self.assert_clean()
        self.assertEqual(self.app.session_state["lab_completed"], 0)

    def test_loading_presets_updates_the_active_dataset(self) -> None:
        self.app.radio(key="app_mode").set_value("B · Experiment Lab").run()
        self.app.selectbox(key="lab_preset").set_value("2 · MANY OUTLIERS").run()
        next(button for button in self.app.button if button.label == "Nạp preset").click().run()
        self.assert_clean()
        self.assertEqual(len(self.app.session_state["lab_points"]), 60)
        self.assertEqual(self.app.session_state["lab_active_config"]["outliers"], 40)
        self.assertEqual(self.app.session_state["lab_completed"], 0)

        self.app.selectbox(key="lab_preset").set_value("3 · WRONG EPSILON").run()
        next(button for button in self.app.button if button.label == "Nạp ε quá lớn").click().run()
        self.assert_clean()
        self.assertEqual(len(self.app.session_state["lab_points"]), 42)
        self.assertEqual(self.app.session_state["lab_epsilon"], 0.90)

    def test_tls_rms_delta_marks_reduction_as_positive(self) -> None:
        self.app.session_state["guided_step"] = 8
        self.app.run()
        self.assert_clean()
        metric = next(item for item in self.app.metric if item.label == "TLS refit RMS ⟂")
        self.assertTrue(metric.delta.startswith("-"))
        self.assertEqual(metric.color, 1)  # Streamlit's green semantic delta.

    def test_presentation_mode_and_real_lab_stage_navigation(self) -> None:
        self.app.toggle(key="presentation_mode").set_value(True).run()
        self.assert_clean()
        self.assertTrue(self.app.session_state["presentation_mode"])
        self.app.radio(key="app_mode").set_value("B · Experiment Lab").run()
        self.app.radio(key="lab_prediction").set_value("Thấp").run()
        next(button for button in self.app.button if button.label == "Chạy 1 iteration").click().run()
        self.assert_clean()
        self.assertEqual(self.app.session_state["lab_stage"], "Sample")
        for expected in ("Fit", "Measure", "Consensus", "Update Best"):
            next(button for button in self.app.button if button.label == "Pha tiếp →").click().run()
            self.assert_clean()
            self.assertEqual(self.app.session_state["lab_stage"], expected)
        next(button for button in self.app.button if button.label == "← Pha trước").click().run()
        self.assertEqual(self.app.session_state["lab_stage"], "Consensus")
        next(button for button in self.app.button if button.label == "Reset experiment").click().run()
        self.assertEqual(self.app.session_state["lab_stage"], "Sample")


if __name__ == "__main__":
    unittest.main()
