import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP=Path(__file__).resolve().parents[1]/"app.py"


def button(app,label):
    return next(b for b in app.button if b.label==label)


class SiftInteractions(unittest.TestCase):
    def setUp(self):
        self.app=AppTest.from_file(str(APP),default_timeout=120).run()
        self.app.selectbox(key="chapter2_module").set_value("SIFT Interactive Tutorial").run()
        self.clean()

    def clean(self):
        self.assertEqual(list(self.app.exception),[])
        self.assertEqual(list(self.app.error),[])

    def test_guided_all_steps_and_gates(self):
        answers={3:"Không",5:"Không",6:"Không",7:"Không",10:"Có",
                 14:"Có",16:"Không",17:"Không",18:"Không"}
        for current in range(1,19):
            self.assertEqual(self.app.session_state["sift_step"],current)
            if current in answers:
                self.assertTrue(button(self.app,"Next →").disabled)
                self.app.radio(key=f"sift_predict_{current}").set_value(answers[current]).run()
                button(self.app,"Reveal kết quả").click().run()
                self.clean()
                if current<18: self.assertFalse(button(self.app,"Next →").disabled)
            if current<18:
                button(self.app,"Next →").click().run()
                self.clean()
        self.assertTrue(button(self.app,"Next →").disabled)

    def test_state_isolation_reset_and_presentation(self):
        self.app.toggle(key="sift_presentation").set_value(True).run()
        button(self.app,"Next →").click().run()
        button(self.app,"Next →").click().run()
        self.app.radio(key="sift_predict_3").set_value("Không").run()
        button(self.app,"Reveal kết quả").click().run()
        self.assertTrue(self.app.session_state["sift_reveal_3"])
        self.app.selectbox(key="chapter2_module").set_value("Harris & Shi-Tomasi Corner Detection").run()
        self.assertEqual(self.app.session_state["chapter2_step"],1)
        self.app.selectbox(key="chapter2_module").set_value("Gaussian Scale Space, LoG & DoG").run()
        self.assertEqual(self.app.session_state["scale_space_step"],1)
        self.app.selectbox(key="chapter2_module").set_value("SIFT Interactive Tutorial").run()
        self.assertEqual(self.app.session_state["sift_step"],3)
        button(self.app,"Reset").click().run()
        self.assertEqual(self.app.session_state["sift_step"],1)
        self.assertNotIn("sift_reveal_3",self.app.session_state)
        self.assertTrue(self.app.session_state["sift_presentation"])
        self.clean()

    def test_labs_parameter_invalidation_and_reset(self):
        self.app.radio(key="sift_mode").set_value("Experiment Lab").run()
        self.clean()
        self.app.radio(key="sift_lab_a_guess").set_value("KEEP").run()
        button(self.app,"Reveal Taylor và quyết định").click().run()
        self.assertTrue(self.app.session_state["sift_lab_a_reveal"])
        self.app.slider(key="sift_lab_a_contrast").set_value(.012).run()
        self.assertNotIn("sift_lab_a_reveal",self.app.session_state)
        self.app.selectbox(key="sift_lab_b_point").set_value(1).run()
        self.app.selectbox(key="sift_lab_b_cell_row").set_value(2).run()
        self.app.selectbox(key="sift_lab_b_cell_col").set_value(3).run()
        self.assertEqual(self.app.session_state["sift_lab_b_cell_row"],2)
        button(self.app,"Tính lại ảnh quay").click().run()
        self.assertEqual(self.app.session_state["sift_lab_b_rotation_ready"],20)
        self.app.selectbox(key="sift_lab_b_preset").set_value("Blob đa thang").run()
        self.assertEqual(self.app.session_state["sift_lab_b_cell_row"],0)
        self.assertNotIn("sift_lab_b_rotation_ready",self.app.session_state)
        button(self.app,"Tính keypoints và descriptors A/B").click().run()
        self.assertTrue(self.app.session_state["sift_lab_c_computed"])
        self.app.slider(key="sift_lab_c_angle").set_value(25).run()
        self.assertNotIn("sift_lab_c_computed",self.app.session_state)
        self.clean()
        button(self.app,"Reset Lab A").click().run()
        self.clean()


if __name__=="__main__": unittest.main()
