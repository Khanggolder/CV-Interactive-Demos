"""Cross-module integration regressions; numerical engines are not modified here."""
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP=Path(__file__).resolve().parents[1]/"app.py"
MODULES=(
    ("Harris & Shi-Tomasi Corner Detection","chapter2_step","chapter2_presentation"),
    ("Gaussian Scale Space, LoG & DoG","scale_space_step","scale_space_presentation"),
    ("SIFT Interactive Tutorial","sift_step","sift_presentation"),
    ("FAST, BRIEF & ORB","orb_step","orb_presentation"),
    ("Feature Matching Challenge","feature_challenge_step","feature_challenge_presentation"),
)


def button(app,label):
    return next(item for item in app.button if item.label==label)


class FiveModuleIntegration(unittest.TestCase):
    def test_navigation_presentation_and_reset_are_isolated(self):
        app=AppTest.from_file(str(APP),default_timeout=120).run()
        self.assertEqual(len(app.selectbox(key="chapter2_module").options),5)
        for name,step,presentation in MODULES:
            app.selectbox(key="chapter2_module").set_value(name).run()
            self.assertEqual(list(app.exception),[])
            app.toggle(key=presentation).set_value(True).run()
            button(app,"Next →").click().run()
            self.assertEqual(app.session_state[step],2)
        for name,step,presentation in MODULES:
            app.selectbox(key="chapter2_module").set_value(name).run()
            self.assertEqual(app.session_state[step],2)
            self.assertTrue(app.session_state[presentation],name)
            self.assertEqual(list(app.exception),[])
        for name,step,presentation in MODULES:
            app.selectbox(key="chapter2_module").set_value(name).run()
            button(app,"Reset").click().run()
            self.assertEqual(app.session_state[step],1)
            self.assertTrue(app.session_state[presentation])
            self.assertEqual(list(app.exception),[])
        self.assertTrue(all(app.session_state[step]==1 for _,step,_ in MODULES))


if __name__=="__main__":unittest.main()
