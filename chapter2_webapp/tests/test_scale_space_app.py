import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def button(app,label):
    return next(b for b in app.button if b.label == label)


class ScaleSpaceInteractions(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(APP),default_timeout=90).run()
        self.app.selectbox(key="chapter2_module").set_value("Gaussian Scale Space, LoG & DoG").run()
        self.clean()

    def clean(self):
        self.assertEqual(list(self.app.exception),[])
        self.assertEqual(list(self.app.error),[])

    def test_module_state_isolation_and_reset(self):
        button(self.app,"Next →").disabled
        self.assertTrue(button(self.app,"Next →").disabled)
        self.app.radio(key="scale_space_predict_1").set_value("Không").run()
        button(self.app,"Reveal kết quả").click().run()
        button(self.app,"Next →").click().run()
        self.assertEqual(self.app.session_state["scale_space_step"],2)
        self.app.selectbox(key="chapter2_module").set_value("Harris & Shi-Tomasi Corner Detection").run()
        self.assertEqual(self.app.session_state["chapter2_step"],1)
        self.app.selectbox(key="chapter2_module").set_value("Gaussian Scale Space, LoG & DoG").run()
        self.assertEqual(self.app.session_state["scale_space_step"],2)
        button(self.app,"Reset").click().run()
        self.assertEqual(self.app.session_state["scale_space_step"],1)
        self.assertNotIn("scale_space_reveal_1",self.app.session_state)
        self.clean()

    def test_all_guided_steps_and_prediction_gates(self):
        answers = {1:"Không",3:"Nhỏ",7:"Blob nhỏ",8:"Âm",9:"Không",11:"Không",15:"Không"}
        for current in range(1,15):
            if current in answers:
                self.assertTrue(button(self.app,"Next →").disabled)
                self.app.radio(key=f"scale_space_predict_{current}").set_value(answers[current]).run()
                button(self.app,"Reveal kết quả").click().run()
            if current == 13:
                self.assertTrue(button(self.app,"Next →").disabled)
                self.app.radio(key="scale_space_guided_inspect_prediction").set_value("NOT EXTREMUM").run()
                button(self.app,"Bắt đầu").click().run()
                button(self.app,"So sánh →").click().run()
                self.assertEqual(self.app.session_state["scale_space_guided_inspect_seen"],1)
                button(self.app,"← So sánh").click().run()
                self.assertEqual(self.app.session_state["scale_space_guided_inspect_seen"],0)
                button(self.app,"Chạy còn lại").click().run()
                self.assertEqual(self.app.session_state["scale_space_guided_inspect_seen"],26)
                button(self.app,"Reveal quyết định").click().run()
            button(self.app,"Next →").click().run()
            self.assertEqual(self.app.session_state["scale_space_step"],current+1)
            self.clean()
        self.assertTrue(button(self.app,"Reveal kết quả").disabled)
        self.app.radio(key="scale_space_predict_15").set_value("Không").run()
        button(self.app,"Reveal kết quả").click().run()
        self.clean()

    def test_labs_presentation_invalidation(self):
        self.app.radio(key="scale_space_mode").set_value("Experiment Lab").run()
        self.app.toggle(key="scale_space_presentation").set_value(True).run()
        self.app.selectbox(key="scale_space_lab_b_preset").set_value("Blob tối").run()
        self.app.slider(key="scale_space_lab_b_intervals").set_value(4).run()
        self.app.radio(key="scale_space_lab_inspect_prediction").set_value("MAX").run()
        button(self.app,"Bắt đầu").click().run()
        button(self.app,"So sánh →").click().run()
        self.assertEqual(self.app.session_state["scale_space_lab_inspect_seen"],1)
        self.app.slider(key="scale_space_lab_b_base_sigma").set_value(1.8).run()
        self.assertEqual(self.app.session_state["scale_space_lab_inspect_seen"],0)
        self.assertFalse(self.app.session_state["scale_space_lab_inspect_started"])
        self.app.selectbox(key="scale_space_lab_b_preset").set_value("Một blob sáng").run()
        self.assertEqual(self.app.session_state["scale_space_lab_inspect_seen"],0)
        self.assertFalse(self.app.session_state["scale_space_lab_inspect_started"])
        button(self.app,"Reset Lab B").click().run()
        self.assertEqual(self.app.session_state["scale_space_lab_b_preset"],"Ba blob sáng")
        self.assertTrue(self.app.session_state["scale_space_presentation"])
        self.clean()


if __name__ == "__main__":
    unittest.main()
