import unittest
from types import SimpleNamespace

import numpy as np

from utils.orb_visualization import matching_view

from utils.orb import (CIRCLE, Keypoint, apply_fast_nms, bit_entropy,
    build_oriented_brief_descriptor, build_orb_image_pyramid,
    calculate_hamming_distance, calculate_intensity_moments,
    classify_fast_ring, detect_fast_candidates, detect_fast_corner,
    evaluate_fast_early_test, fast_score_map, generate_brief_test_pattern,
    get_fast_circle_offsets, map_pyramid_coordinates, match_binary_descriptors,
    microscope_fixtures, orb_pipeline, pack_binary_descriptor,
    pattern_learning_demo, quantize_orientation,
    rank_fast_candidates_with_harris, sample_brief_pair_intensities,
    steer_brief_pairs, teaching_presets, transform_scene,
    unpack_binary_descriptor)


class MatchingRevealView(unittest.TestCase):
    def test_unrevealed_plot_hides_rule_and_geometry(self):
        image=np.zeros((9,9),dtype=np.uint8)
        point=SimpleNamespace(x_original=4.,y_original=4.)
        accepted=SimpleNamespace(query=0,train=0,accepted=True,correct_geometry=False,
                                 geometric_error=7.5,distance=39)
        rejected=SimpleNamespace(query=0,train=0,accepted=False,correct_geometry=True,
                                 geometric_error=.5,distance=50)
        before=matching_view(image,image,[point],[point],[accepted,rejected],reveal=False)
        after=matching_view(image,image,[point],[point],[accepted,rejected],reveal=True)
        self.assertEqual(len(before.data),3)  # image plus both neutral candidates
        self.assertEqual(len(after.data),2)   # only rule-accepted candidate
        self.assertTrue(all(trace.line.color=="#61758d" for trace in before.data[1:]))
        self.assertNotIn("geometry error",before.to_json())
        self.assertNotIn("đúng hình học",before.to_json())
        self.assertIn("geometry error",after.to_json())


class FastNumbers(unittest.TestCase):
    def setUp(self):self.f=microscope_fixtures()

    def test_exact_clockwise_circle(self):
        self.assertEqual(len(CIRCLE),16)
        self.assertEqual(get_fast_circle_offsets(),CIRCLE)
        self.assertEqual([CIRCLE[i] for i in (0,4,8,12)],
                         [(-3,0),(0,3),(3,0),(0,-3)])
        self.assertEqual(len(set(CIRCLE)),16)

    def test_strict_classification_and_equality(self):
        values=np.array([120,80,121,79]+[100]*12)
        self.assertEqual(classify_fast_ring(values,100,20)[:4],
                         ("SIMILAR","SIMILAR","BRIGHT","DARK"))

    def test_full_arc_and_wrap(self):
        for name in ("Cung 12","Wrap 16→1"):
            result=detect_fast_corner(self.f[name],7,7)
            self.assertTrue(result.accepted)
            self.assertTrue(result.early_pass)
            self.assertEqual(len(result.arc),12)
            self.assertEqual(result.score,110)
        self.assertIn(15,detect_fast_corner(self.f["Wrap 16→1"],7,7).arc)
        self.assertIn(0,detect_fast_corner(self.f["Wrap 16→1"],7,7).arc)

    def test_pretest_is_necessary_not_sufficient(self):
        survivor=detect_fast_corner(self.f["Pretest sống, cung rớt"],7,7)
        self.assertTrue(survivor.early_pass)
        self.assertFalse(survivor.accepted)
        self.assertEqual(survivor.score,0)
        self.assertFalse(detect_fast_corner(self.f["Early reject"],7,7).early_pass)
        with self.assertRaises(ValueError):evaluate_fast_early_test(survivor.labels,9)

    def test_flat_edge_scattered_and_dark(self):
        for name in ("Vùng phẳng","Cạnh thẳng"):
            self.assertFalse(detect_fast_corner(self.f[name],7,7).accepted)
        scattered=np.full((15,15),100.)
        for i in range(0,16,2):
            dr,dc=CIRCLE[i];scattered[7+dr,7+dc]=220
        self.assertFalse(detect_fast_corner(scattered,7,7).accepted)
        dark=255-self.f["Cung 12"]
        self.assertTrue(detect_fast_corner(dark,7,7).accepted)
        self.assertEqual(detect_fast_corner(dark,7,7).polarity,"DARK")

    def test_no_pretest_false_negative_exhaustive_patterns(self):
        rng=np.random.default_rng(2026)
        for _ in range(2000):
            ring=rng.choice((0,100,220),16)
            a=np.full((7,7),100.)
            for i,(dr,dc) in enumerate(CIRCLE):a[3+dr,3+dc]=ring[i]
            result=detect_fast_corner(a,3,3)
            self.assertFalse(result.accepted and not result.early_pass)

    def test_score_threshold_agreement_and_boundary(self):
        a=self.f["Cung 12"]
        result=detect_fast_corner(a,7,7,threshold=20)
        self.assertEqual(result.score,min(result.margins))
        self.assertEqual(result.score,fast_score_map(a)[7,7])
        self.assertTrue(result.accepted)
        self.assertFalse(detect_fast_corner(a,7,7,threshold=110).accepted)
        self.assertFalse(detect_fast_corner(a,2,7).valid)
        mask,scores=detect_fast_candidates(a,20)
        self.assertTrue(mask[7,7])
        self.assertFalse(mask[0,0])

    def test_nms_tie_and_harris_separation(self):
        scores=np.zeros((11,11));scores[5,5]=scores[5,6]=5
        eligible=scores>0
        kept=apply_fast_nms(scores,eligible)
        self.assertTrue(kept[5,5]);self.assertFalse(kept[5,6])
        a=self.f["Cung 12"]
        mask,full=detect_fast_candidates(a,20)
        selected,response=rank_fast_candidates_with_harris(a,apply_fast_nms(full,mask),full)
        self.assertTrue(all(mask[r,c] for r,c,_,_ in selected))
        self.assertEqual(response.shape,a.shape)


class PyramidAndOrientation(unittest.TestCase):
    def test_pyramid_dimensions_and_mapping(self):
        a=np.zeros((129,129))
        levels=build_orb_image_pyramid(a,levels=8)
        self.assertEqual(len(levels),8)
        self.assertEqual(levels[0].image.shape,a.shape)
        self.assertLess(levels[7].image.shape[0],levels[1].image.shape[0])
        self.assertAlmostEqual(levels[1].nominal_scale,1.2)
        x,y=map_pyramid_coordinates(levels[1],10,20)
        self.assertAlmostEqual(x,(20.5)*129/levels[1].image.shape[1]-.5)
        self.assertAlmostEqual(y,(10.5)*129/levels[1].image.shape[0]-.5)

    def test_moments_and_atan2(self):
        a=np.zeros((31,31));a[15,16]=10;a[16,15]=20
        m=calculate_intensity_moments(a,15,15,radius=2)
        self.assertEqual((m.m00,m.m10,m.m01),(30,10,20))
        self.assertAlmostEqual(m.cx,1/3)
        self.assertAlmostEqual(m.cy,2/3)
        self.assertAlmostEqual(m.angle,np.degrees(np.arctan2(20,10)))
        a[:]=0;a[14,14]=5
        self.assertAlmostEqual(calculate_intensity_moments(a,15,15,2).angle,225)

    def test_undefined_and_border(self):
        a=np.full((31,31),100.)
        self.assertIsNone(calculate_intensity_moments(a,15,15,4).angle)
        with self.assertRaises(ValueError):calculate_intensity_moments(a,2,2,4)

    def test_rotation_transform_and_detection(self):
        a=teaching_presets()["Cấu trúc bất đối xứng"]
        b,matrix=transform_scene(a,20)
        self.assertEqual(a.shape,b.shape)
        self.assertEqual(matrix.shape,(2,3))
        _,points,_=orb_pipeline(b)
        self.assertTrue(points)


class BinaryNumbers(unittest.TestCase):
    def test_pattern_deterministic_and_bounded(self):
        a=generate_brief_test_pattern()
        self.assertEqual(a.shape,(256,4))
        np.testing.assert_array_equal(a,generate_brief_test_pattern())
        for pair in a.reshape(-1,2):self.assertLessEqual(np.linalg.norm(pair),7)

    def test_entropy_and_learning(self):
        self.assertEqual(bit_entropy(0),0)
        self.assertEqual(bit_entropy(1),0)
        self.assertAlmostEqual(bit_entropy(.5),1)
        candidate,bits,entropy,selected=pattern_learning_demo()
        self.assertEqual(bits.shape,(40,80))
        self.assertGreaterEqual(len(selected),8)
        self.assertTrue(np.all(entropy[selected]>.5))

    def test_steering_and_quantization(self):
        p=np.array([[1.,0.,0.,1.]])
        moved,angle=steer_brief_pairs(p,90,quantized=False)
        np.testing.assert_allclose(moved,[[0,1,-1,0]],atol=1e-12)
        self.assertEqual(angle,90)
        self.assertEqual(quantize_orientation(359),0)
        self.assertEqual(quantize_orientation(7),12)

    def test_actual_sampling_and_border(self):
        a=np.arange(49,dtype=float).reshape(7,7)
        sample=sample_brief_pair_intensities(a,3,3,[-1,0,1,0])
        self.assertEqual(sample[-2:],(23,25))
        self.assertIsNone(sample_brief_pair_intensities(a,3,3,[-9,0,1,0]))

    def test_256_bit_pack_roundtrip_and_determinism(self):
        a=teaching_presets()["Cấu trúc bất đối xứng"]
        _,points,_=orb_pipeline(a)
        self.assertTrue(points)
        p=points[0]
        level=build_orb_image_pyramid(a)[p.level]
        d=build_oriented_brief_descriptor(level.image,p.row,p.col,p.angle)
        self.assertEqual(d.bits.shape,(256,))
        self.assertEqual(len(d.packed),32)
        np.testing.assert_array_equal(unpack_binary_descriptor(d.packed),d.bits)
        self.assertEqual(d.packed,p.descriptor)
        self.assertEqual(d.packed,pack_binary_descriptor(d.bits))
        self.assertEqual(len(d.samples),256)
        self.assertEqual(d.bits[0],int(d.samples[0][4]<d.samples[0][5]))

    def test_invalid_descriptor_and_equal_intensity(self):
        a=np.full((25,25),100.)
        bad=build_oriented_brief_descriptor(a,12,12,None)
        self.assertIsNone(bad.packed)
        edge=build_oriented_brief_descriptor(a,2,2,0)
        self.assertIsNone(edge.packed)
        center=build_oriented_brief_descriptor(a,12,12,0)
        self.assertTrue(np.all(center.bits==0))
        with self.assertRaises(ValueError):pack_binary_descriptor(np.zeros(255))

    def test_predictable_real_brief_bits(self):
        ramp=np.tile(np.arange(25,dtype=float)*8,(25,1))
        pattern=np.array([[-3,0,3,0],[3,0,-3,0],[-2,-1,2,1],[2,1,-2,-1],
                          [-1,2,1,-2],[1,-2,-1,2],[-4,0,4,0],[4,0,-4,0]],float)
        descriptor=build_oriented_brief_descriptor(ramp,12,12,0,pattern,steering=False)
        np.testing.assert_array_equal(descriptor.bits,[1,0,1,0,1,0,1,0])
        self.assertIsNone(descriptor.packed)  # eight teaching bits are not padded to 256

    def test_exact_hamming_bounds(self):
        zero=bytes(32);ones=bytes([255]*32)
        self.assertEqual(calculate_hamming_distance(zero,zero),0)
        self.assertEqual(calculate_hamming_distance(zero,ones),256)
        self.assertEqual(calculate_hamming_distance(bytes([0b10110000])+bytes(31),
                                                    bytes([0b11000100])+bytes(31)),4)
        self.assertEqual(calculate_hamming_distance(ones,zero),256)

    def test_matching_rules_and_geometry_separate(self):
        descriptor=lambda byte:bytes([byte])+bytes(31)
        make=lambda x,d:Keypoint(0,10,int(x),float(x),10.,30.,0.,0.,descriptor(d))
        a=[make(10,0),make(20,255)]
        b=[make(90,0),make(20,255)]
        transform=np.array([[1,0,0],[0,1,0]],float)
        matches=match_binary_descriptors(a,b,method="nearest",transform=transform)
        self.assertTrue(matches[0].accepted)
        self.assertFalse(matches[0].correct_geometry)
        self.assertTrue(matches[1].correct_geometry)
        self.assertTrue(all(m.accepted for m in match_binary_descriptors(a,b,"cross-check")))
        self.assertTrue(all(m.accepted for m in match_binary_descriptors(a,b,"ratio")))
        missing=[Keypoint(0,0,0,0.,0.,0.,0.,None,None)]
        self.assertFalse(match_binary_descriptors(missing,b)[0].accepted)


if __name__=="__main__":unittest.main()
