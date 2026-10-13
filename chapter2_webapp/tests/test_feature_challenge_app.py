"""Streamlit state/gate regressions for the applied module."""
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest
from tutorials.feature_challenge import _comparison_rows

APP=Path(__file__).resolve().parents[1]/'app.py'


def button(app,label):
    return next(b for b in app.button if b.label==label)


class FeatureChallengeApp(unittest.TestCase):
    def setUp(self):
        self.app=AppTest.from_file(str(APP),default_timeout=120).run()
        self.app.selectbox(key='chapter2_module').set_value('Feature Matching Challenge').run()
        self.clean()

    def clean(self):
        self.assertEqual(list(self.app.exception),[])
        self.assertEqual(list(self.app.error),[])

    def test_fourteen_steps_and_prediction_gates(self):
        answers={3:('feature_challenge_g_gate_3_guess','SIFT'),
                 9:('feature_challenge_g_gate_9_guess','Có'),
                 10:('feature_challenge_g_gate_10_guess','Có'),
                 12:('feature_challenge_g_gate_12_guess','SIFT'),
                 13:('feature_challenge_g_gate_13_guess','Không chắc')}
        for step in range(1,15):
            self.assertEqual(self.app.session_state['feature_challenge_step'],step)
            self.clean()
            if step in answers:
                self.assertTrue(button(self.app,'Next →').disabled)
                key,value=answers[step]
                self.app.radio(key=key).set_value(value).run()
                button(self.app,'Reveal kết quả').click().run()
                self.clean()
                self.assertFalse(button(self.app,'Next →').disabled)
            if step==14:
                self.assertNotIn('feature_challenge_g_final_reveal',self.app.session_state)
                self.app.text_area(key='feature_challenge_g_reason').set_value('Precision và coverage phải cùng được xét.').run()
                button(self.app,'Ghi nhận lập luận').click().run()
                self.assertTrue(self.app.session_state['feature_challenge_g_final_reveal'])
            if step<14:button(self.app,'Next →').click().run()
        self.assertTrue(button(self.app,'Next →').disabled)

    def test_lab_challenge_presentation_and_reset(self):
        self.app.toggle(key='feature_challenge_presentation').set_value(True).run()
        self.app.radio(key='feature_challenge_mode').set_value('Experiment Lab').run()
        self.clean()
        button(self.app,'Run comparison').click().run()
        self.clean()
        self.assertIn('feature_challenge_c_results',self.app.session_state)
        self.assertNotIn('Save snapshot',[b.label for b in self.app.button])
        button(self.app,'Reveal geometry metrics').click().run()
        self.clean()
        self.assertIn('Save snapshot',[b.label for b in self.app.button])
        self.app.selectbox(key='feature_challenge_c_filter_1').set_value('Đúng/sai hình học').run()
        self.clean()
        self.assertEqual(self.app.selectbox(key='feature_challenge_c_filter_1').value,'Đúng/sai hình học')
        self.app.slider(key='feature_challenge_c_angle').set_value(15.).run()
        self.clean()
        self.assertIn('feature_challenge_c_results',self.app.session_state)
        self.app.radio(key='feature_challenge_mode').set_value('Classroom Challenge').run()
        self.clean()
        self.assertNotIn('feature_challenge_class_results',self.app.session_state)
        self.app.radio(key='feature_challenge_class_prediction').set_value('SIFT').run()
        button(self.app,'Lock prediction').click().run()
        button(self.app,'Run scenario').click().run()
        self.clean()
        self.assertIn('feature_challenge_class_results',self.app.session_state)
        button(self.app,'Reveal measured quality').click().run()
        self.clean()
        button(self.app,'Reset Challenge').click().run()
        self.clean()
        self.assertNotIn('feature_challenge_class_results',self.app.session_state)

    def test_module_isolation(self):
        self.app.selectbox(key='chapter2_module').set_value('Harris & Shi-Tomasi Corner Detection').run()
        self.app.selectbox(key='chapter2_module').set_value('Gaussian Scale Space, LoG & DoG').run()
        self.app.selectbox(key='chapter2_module').set_value('SIFT Interactive Tutorial').run()
        self.app.selectbox(key='chapter2_module').set_value('FAST, BRIEF & ORB').run()
        self.app.selectbox(key='chapter2_module').set_value('Feature Matching Challenge').run()
        self.clean()
        self.assertEqual(self.app.session_state['feature_challenge_step'],1)

    def test_comparison_rule_and_mobile_cards_share_results(self):
        self.app.radio(key='feature_challenge_mode').set_value('Experiment Lab').run()
        button(self.app,'Run comparison').click().run()
        self.assertFalse(any(m.value.startswith('<div class="feature-result-card"') for m in self.app.markdown))
        button(self.app,'Reveal geometry metrics').click().run()
        self.clean()
        results=self.app.session_state['feature_challenge_c_results']
        rows=_comparison_rows(results)
        self.assertEqual([r['Rule'] for r in rows],['Ratio','Ratio','Cross-check'])
        markup=' '.join(m.value for m in self.app.markdown)
        for row in rows:
            self.assertIn(row['Pipeline'],markup)
            self.assertIn(f"Precision: {row['Precision']}",markup)
            self.assertIn(f"Coverage: {row['Coverage']}",markup)
        self.assertTrue(any('chính sách matching' in str(w.proto).lower() for w in self.app.warning),
                        repr([w.proto for w in self.app.warning]))
        self.app.selectbox(key='feature_challenge_c_rule').set_value('Ratio').run()
        self.clean()
        self.assertFalse(any(m.value.startswith('<div class="feature-result-card"') for m in self.app.markdown))
        button(self.app,'Run comparison').click().run()
        button(self.app,'Reveal geometry metrics').click().run()
        self.assertEqual({r['Rule'] for r in _comparison_rows(
            self.app.session_state['feature_challenge_c_results'])},{'Ratio'})

    def test_lab_c_tolerance_reclassifies_without_rematching(self):
        self.app.radio(key='feature_challenge_mode').set_value('Experiment Lab').run()
        button(self.app,'Run comparison').click().run()
        button(self.app,'Reveal geometry metrics').click().run()
        old=self.app.session_state['feature_challenge_c_results']
        old_decisions=[[(d.query,d.train,d.distance,d.accepted) for d in r.decisions]
                       for r in old]
        self.app.slider(key='feature_challenge_c_tolerance').set_value(1.).run()
        self.clean()
        updated=self.app.session_state['feature_challenge_c_results']
        self.assertEqual(old_decisions,
                         [[(d.query,d.train,d.distance,d.accepted) for d in r.decisions]
                          for r in updated])
        self.assertTrue(all(r.pair.config.tolerance==1. for r in updated))
        self.assertFalse(self.app.session_state['feature_challenge_c_reveal'])
        self.assertNotIn('Save snapshot',[b.label for b in self.app.button])
        self.assertTrue(any('phân loại đúng/sai đã tính lại' in str(i.proto) for i in self.app.info))


if __name__=='__main__':unittest.main()
