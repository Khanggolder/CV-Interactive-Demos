import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def button(app, label):
    return next(item for item in app.button if item.label == label)


class HarrisInteractions(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(APP), default_timeout=60).run()
        self.clean()

    def clean(self):
        self.assertEqual(list(self.app.exception), [])
        self.assertEqual(list(self.app.error), [])

    def test_guided_navigation_and_reset(self):
        button(self.app, "Next →").click().run()
        self.assertEqual(self.app.session_state["chapter2_step"], 2)
        self.assertTrue(button(self.app, "Next →").disabled)
        self.app.radio(key="chapter2_predict_2").set_value("Patch C").run()
        button(self.app, "Reveal kết quả").click().run()
        self.assertFalse(button(self.app, "Next →").disabled)
        button(self.app, "Next →").click().run()
        self.assertEqual(self.app.session_state["chapter2_step"], 3)
        button(self.app, "← Previous").click().run()
        button(self.app, "Reset").click().run()
        self.assertEqual(self.app.session_state["chapter2_step"], 1)
        self.assertNotIn("chapter2_reveal_2", self.app.session_state)
        self.clean()

    def test_all_steps_and_gates(self):
        choices = {2: "Patch C", 4: "Cả x và y", 8: "Góc", 9: "Góc",
                   11: "Điểm theo thứ tự hàng/cột nhỏ nhất", 12: "min(λ₁,λ₂)"}
        for current in range(1, 14):
            if current in choices:
                self.assertTrue(button(self.app, "Next →").disabled)
                self.app.radio(key=f"chapter2_predict_{current}").set_value(choices[current]).run()
                button(self.app, "Reveal kết quả").click().run()
                if current == 11:
                    self.app.slider(key="chapter2_nms_candidate").set_value(2).run()
                    self.assertEqual(self.app.session_state["chapter2_nms_candidate"], 2)
            button(self.app, "Next →").click().run()
            self.assertEqual(self.app.session_state["chapter2_step"], current+1)
            self.clean()
        self.assertTrue(button(self.app, "Reveal kết quả").disabled)
        self.app.radio(key="chapter2_predict_14").set_value("Không").run()
        button(self.app, "Reveal kết quả").click().run()
        self.clean()

    def test_lab_controls_reset_and_presentation(self):
        self.app.radio(key="chapter2_mode").set_value("Experiment Lab").run()
        self.app.selectbox(key="chapter2_lab_preset").set_value("Cạnh ngang").run()
        self.app.radio(key="chapter2_lab_algorithm").set_value("Shi-Tomasi").run()
        self.app.toggle(key="chapter2_presentation").set_value(True).run()
        self.assertEqual(self.app.session_state["chapter2_lab_algorithm"], "Shi-Tomasi")
        self.assertTrue(self.app.session_state["chapter2_presentation"])
        button(self.app, "Reset Lab").click().run()
        self.assertEqual(self.app.session_state["chapter2_lab_preset"], "Góc chữ L")
        self.clean()


if __name__ == "__main__":
    unittest.main()
