"""Independent geometry, descriptor, matching and integration fixtures."""
import unittest
from dataclasses import replace

import cv2
import numpy as np

from utils import harris, orb, sift
from utils.feature_challenge import (Config, Detection, Decision, Feature, METHODS,
    PRESETS, SCENES, Pair, Result, build_transformed_image_pair,
    compute_matching_metrics, count_unique_keypoints, detect_pair,
    evaluate_match_geometry, generate_challenge_scene, generate_experiment_snapshot,
    map_points_with_affine, match_features, normalize_patch_descriptor, run_method)


def detection(method, features):
    return Detection(method, len(features), len(features), len(features), tuple(features), 0.)


def feature(x, y, vector, i=0):
    return Feature(float(x),float(y),vector,{},True,i)


class PairFixtures(unittest.TestCase):
    def test_six_deterministic_scenes(self):
        self.assertEqual(len(SCENES),6)
        for scene in SCENES:
            np.testing.assert_array_equal(generate_challenge_scene(scene),generate_challenge_scene(scene))
            self.assertGreater(np.std(generate_challenge_scene(scene)),.05)

    def test_affine_forward_inverse_and_pixel_center(self):
        config=Config(angle=0,scale=1,dx=7,dy=-4)
        pair=build_transformed_image_pair(config)
        np.testing.assert_allclose(map_points_with_affine([[5,6]],pair.matrix),[[12,2]],atol=1e-9)
        inv=cv2.invertAffineTransform(pair.matrix)
        np.testing.assert_allclose(map_points_with_affine([[12,2]],inv),[[5,6]],atol=1e-9)
        self.assertAlmostEqual(pair.b[60,70],pair.a[64,63],places=8)

    def test_mask_exact_support_and_padding(self):
        pair=build_transformed_image_pair(Config(angle=0,scale=1,dx=8,dy=0))
        self.assertFalse(pair.valid_b[:,0:8].any())
        self.assertTrue(pair.valid_b[:,8:].all())

    def test_noise_brightness_blur(self):
        cfg=Config(noise=.05,brightness=.1,contrast=.8,blur=1.1,seed=41)
        p=build_transformed_image_pair(cfg)
        np.testing.assert_array_equal(p.b,build_transformed_image_pair(cfg).b)
        self.assertFalse(np.array_equal(p.b,build_transformed_image_pair(replace(cfg,seed=42)).b))
        self.assertFalse(np.array_equal(p.b,build_transformed_image_pair(replace(cfg,noise=0)).b))

    def test_invalid_bounds(self):
        with self.assertRaises(ValueError):build_transformed_image_pair(Config(scale=0))
        with self.assertRaises(ValueError):map_points_with_affine([[1,2]],np.eye(3))


class DescriptorFixtures(unittest.TestCase):
    def test_harris_patch_normalization(self):
        x=np.arange(121,dtype=float).reshape(11,11)
        d=normalize_patch_descriptor(x)
        self.assertEqual(d.shape,(121,))
        self.assertAlmostEqual(float(np.linalg.norm(d)),1,places=6)
        self.assertAlmostEqual(float(d.mean()),0,places=6)
        np.testing.assert_allclose(d,normalize_patch_descriptor(2*x+10),atol=1e-6)
        np.testing.assert_array_equal(normalize_patch_descriptor(np.ones((11,11))),np.zeros(121))

    def test_harris_engine_reuse_and_border(self):
        pair=build_transformed_image_pair(Config())
        a,b=detect_pair(pair,METHODS[0])
        self.assertGreater(a.available_locations,0)
        response=harris.pipeline(pair.a).harris
        for f in a.features:
            self.assertAlmostEqual(f.metadata['response'],response[int(f.y),int(f.x)])
        self.assertTrue(all(f.descriptor is None or f.descriptor.shape==(121,) for f in a.features))
        self.assertTrue(all(f.x>=5 and f.y>=5 for f in a.features))

    def test_sift_orientations_not_unique_locations(self):
        pair=build_transformed_image_pair(Config())
        a,b=detect_pair(pair,METHODS[1])
        self.assertGreater(a.orientation_records,a.retained_locations)
        self.assertTrue(all(f.descriptor is None or f.descriptor.shape==(128,) for f in a.features))
        self.assertTrue(all('octave' in f.metadata and 'sigma' in f.metadata for f in a.features))
        self.assertEqual(count_unique_keypoints([(2,3),(2.1,3.1),(9,8)]),2)
        self.assertTrue(sift.apply_lowe_ratio_test(np.array([.1,.3]),.8).accepted)
        self.assertFalse(sift.apply_lowe_ratio_test(np.array([.1,.11]),.8).accepted)

    def test_orb_pyramid_hamming_and_metadata(self):
        pair=build_transformed_image_pair(Config())
        a,b=detect_pair(pair,METHODS[2])
        self.assertTrue(all(f.descriptor is None or len(f.descriptor)==32 for f in a.features))
        self.assertTrue(all('level' in f.metadata and 'angle' in f.metadata for f in a.features))
        self.assertEqual(orb.calculate_hamming_distance(bytes(32),bytes([255])*32),256)


class MatchingFixtures(unittest.TestCase):
    def test_euclidean_nearest_ratio_and_crosscheck(self):
        a=detection(METHODS[0],[feature(0,0,np.array([0.,0.])),feature(1,0,np.array([.9,0.]),1)])
        b=detection(METHODS[0],[feature(0,0,np.array([.1,0.])),feature(2,0,np.array([2.,0.]),1)])
        self.assertAlmostEqual(match_features(a,b,'Nearest')[0].distance,.1)
        self.assertTrue(match_features(a,b,'Ratio',.8)[0].accepted)
        self.assertFalse(match_features(a,b,'Cross-check')[1].accepted)

    def test_hamming_ratio_and_crosscheck(self):
        zeros=bytes(32);ones=bytes([255])*32
        a=detection(METHODS[2],[feature(0,0,zeros),feature(1,0,ones,1)])
        b=detection(METHODS[2],[feature(0,0,zeros),feature(1,0,ones,1)])
        self.assertEqual([d.distance for d in match_features(a,b,'Nearest')],[0,0])
        self.assertTrue(all(d.accepted for d in match_features(a,b,'Cross-check')))
        self.assertTrue(all(d.accepted for d in match_features(a,b,'Ratio')))

    def test_invalid_descriptor_is_not_candidate(self):
        a=detection(METHODS[0],[feature(0,0,None),feature(2,2,np.array([1.] ),1)])
        b=detection(METHODS[0],[feature(3,2,None)])
        self.assertEqual(len(match_features(a,b)),1)
        self.assertIsNone(match_features(a,b)[0].train)

    def test_geometry_posthoc_only(self):
        cfg=Config(angle=0,scale=1,dx=5,dy=-2)
        pair=build_transformed_image_pair(cfg)
        a=detection(METHODS[0],[feature(20,20,np.array([0.]))])
        b=detection(METHODS[0],[feature(25,18,np.array([0.])),feature(40,40,np.array([2.]),1)])
        original=match_features(a,b,'Nearest')
        good=evaluate_match_geometry(pair,a,b,original,4.)
        self.assertEqual(original[0].train,good[0].train)
        self.assertTrue(good[0].correct)
        swapped=detection(METHODS[0],[feature(40,40,np.array([0.]))])
        wrong=evaluate_match_geometry(pair,a,swapped,match_features(a,swapped,'Nearest'),4.)
        self.assertFalse(wrong[0].correct)
        self.assertGreater(wrong[0].error,4)
        self.assertTrue(evaluate_match_geometry(pair,a,swapped,match_features(a,swapped,'Nearest'),100.)[0].correct)

    def test_metrics_denominators_and_empty(self):
        pair=build_transformed_image_pair(Config())
        a=detection(METHODS[0],[feature(0,0,np.array([0.])),feature(1,1,np.array([1.]),1)])
        b=detection(METHODS[0],[feature(0,0,np.array([0.]))])
        decisions=(Decision(0,0,None,0,None,True,'nearest',0.,True),
                   Decision(1,0,None,1,None,True,'nearest',8.,False))
        result=Result(METHODS[0],pair,a,b,decisions,'Nearest',0)
        m=compute_matching_metrics(result)
        self.assertEqual(m['precision'],.5)
        self.assertEqual(m['coverage'],.5)
        self.assertEqual(m['median_error'],4.)
        self.assertAlmostEqual(m['p90_error'],7.2)
        self.assertNotIn('recall',m)
        no=compute_matching_metrics(replace(result,decisions=()))
        self.assertIsNone(no['precision'])
        self.assertIsNone(no['median_error'])
        self.assertEqual(no['accepted'],0)

    def test_zero_overlap_no_invented_matches(self):
        pair=build_transformed_image_pair(Config(dx=20,dy=20,angle=45,scale=.55))
        a,b=detect_pair(pair,METHODS[0])
        self.assertTrue(all(f.descriptor is None or f.eligible for f in a.features))
        decisions=match_features(a,b)
        self.assertTrue(all(d.train is None or d.query<len(a.features) for d in decisions))

    def test_snapshot_preserves_full_config(self):
        cfg=Config(scene=SCENES[1],seed=991,rule='Nearest')
        result=run_method(build_transformed_image_pair(cfg),METHODS[0])
        shot=generate_experiment_snapshot([result],'Harris','Observed ambiguity')
        self.assertEqual(shot['config']['seed'],991)
        self.assertEqual(shot['metrics'][METHODS[0]]['accepted'],compute_matching_metrics(result)['accepted'])

    def test_same_pair_budget_and_determinism(self):
        pair=build_transformed_image_pair(Config(budget=10))
        runs=[run_method(pair,m) for m in METHODS]
        self.assertTrue(all(r.pair is pair for r in runs))
        self.assertTrue(all(r.a.retained_locations<=10 for r in runs))
        again=run_method(pair,METHODS[0])
        self.assertEqual([(d.query,d.train,d.accepted) for d in runs[0].decisions],
                         [(d.query,d.train,d.accepted) for d in again.decisions])


if __name__=='__main__':unittest.main()
