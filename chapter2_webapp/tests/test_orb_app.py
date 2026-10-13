import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP=Path(__file__).resolve().parents[1]/"app.py"


def button(app,label):return next(b for b in app.button if b.label==label)


class OrbInteractions(unittest.TestCase):
    def setUp(self):
        self.app=AppTest.from_file(str(APP),default_timeout=120).run()
        self.app.selectbox(key="chapter2_module").set_value("FAST, BRIEF & ORB").run()
        self.clean()

    def clean(self):
        self.assertEqual(list(self.app.exception),[])
        self.assertEqual(list(self.app.error),[])

    def test_guided_all_steps_and_gates(self):
        answers={4:"CORNER",5:"Không",6:"Có",10:"Có",12:"Có",16:"Có",17:"Không",18:"Không"}
        for step in range(1,19):
            self.assertEqual(self.app.session_state["orb_step"],step)
            self.clean()
            if step in answers:
                self.assertTrue(button(self.app,"Next →").disabled)
                self.app.radio(key=f"orb_guess_{step}").set_value(answers[step]).run()
                button(self.app,"Reveal kết quả").click().run()
                self.assertTrue(self.app.session_state[f"orb_reveal_{step}"])
                self.clean()
            if step<18:button(self.app,"Next →").click().run()
        self.assertTrue(button(self.app,"Next →").disabled)

    def test_labs_invalidation_and_reset(self):
        self.app.radio(key="orb_mode").set_value("Experiment Lab").run()
        self.clean()
        self.app.radio(key="orb_a_guess").set_value("CORNER").run()
        button(self.app,"Run pretest + full test / Reveal").click().run()
        self.assertTrue(self.app.session_state["orb_a_reveal"])
        self.app.slider(key="orb_a_threshold").set_value(25).run()
        self.assertNotIn("orb_a_reveal",self.app.session_state)
        self.assertIsNone(self.app.session_state["orb_a_guess"])
        self.app.slider(key="orb_a_arc").set_value(9).run()
        self.app.selectbox(key="orb_a_source").set_value("Preset 129×129").run()
        self.app.slider(key="orb_a_level").set_value(2).run()
        self.clean()
        button(self.app,"Compute 256-bit descriptor").click().run()
        self.assertIn("orb_b_computed",self.app.session_state)
        self.app.toggle(key="orb_b_steer").set_value(False).run()
        self.assertNotIn("orb_b_computed",self.app.session_state)
        button(self.app,"Calculate A/B descriptors").click().run()
        self.assertIn("orb_c_calculated",self.app.session_state)
        self.app.slider(key="orb_c_angle").set_value(25).run()
        self.assertNotIn("orb_c_calculated",self.app.session_state)
        self.clean()
        button(self.app,"Reset Lab A").click().run()
        self.clean()

    def test_presentation_reset_and_isolation(self):
        self.app.toggle(key="orb_presentation").set_value(True).run()
        button(self.app,"Next →").click().run()
        self.app.selectbox(key="chapter2_module").set_value("Harris & Shi-Tomasi Corner Detection").run()
        self.assertEqual(self.app.session_state["chapter2_step"],1)
        self.app.selectbox(key="chapter2_module").set_value("Gaussian Scale Space, LoG & DoG").run()
        self.assertEqual(self.app.session_state["scale_space_step"],1)
        self.app.selectbox(key="chapter2_module").set_value("SIFT Interactive Tutorial").run()
        self.assertEqual(self.app.session_state["sift_step"],1)
        self.app.selectbox(key="chapter2_module").set_value("FAST, BRIEF & ORB").run()
        self.assertEqual(self.app.session_state["orb_step"],2)
        button(self.app,"Reset").click().run()
        self.assertEqual(self.app.session_state["orb_step"],1)
        self.assertTrue(self.app.session_state["orb_presentation"])
        self.clean()


if __name__=="__main__":unittest.main()
