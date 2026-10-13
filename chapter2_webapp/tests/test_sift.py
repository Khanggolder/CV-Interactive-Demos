import unittest
from dataclasses import replace

import numpy as np
import cv2

from utils.scale_space import Octave, build_pyramid, detect_extrema
from utils.sift import (Keypoint, apply_lowe_ratio_test, assign_keypoint_orientations,
    build_sift_descriptor, calculate_dog_derivatives, calculate_orientation_histogram,
    calculate_spatial_hessian, compute_descriptor_distances, descriptor_index,
    detect_dog_candidates, edge_decision, evaluate_geometric_correspondence,
    filter_keypoints, gradient_field, match_descriptors, normalize_sift_descriptor,
    pipeline, refine_keypoint, sample_oriented_descriptor_patch,
    teaching_presets, transform_scene)


def quadratic_stack(cx=4.25,cy=4.2,cs=2.1):
    yy,xx=np.indices((11,11))
    return tuple(2.-(xx-cx)**2-.7*(yy-cy)**2-.9*(s-cs)**2
                 +.12*(xx-cx)*(yy-cy)+.06*(xx-cx)*(s-cs)
                 for s in range(5))


def fake_octave(stack):
    sigmas=tuple(1.6*2**(i/3) for i in range(6))
    return Octave(0,tuple(np.zeros_like(stack[0]) for _ in range(6)),stack,
                  sigmas,sigmas,tuple(0. for _ in range(6)))


def point(layer=2,row=4,col=4):
    return Keypoint(0,layer,row,col,0.,"MAX",float(row),float(col),float(layer),
                    float(row),float(col),1.6,1.6,0.)


class SiftNumericalTests(unittest.TestCase):
    def test_pyramid_reuse_and_coordinates(self):
        image=teaching_presets()["Blob đa thang"]
        pyramid=build_pyramid(image)
        found=detect_dog_candidates(pyramid)
        self.assertEqual(len(found),sum(len(detect_extrema(o.dogs)) for o in pyramid))
        for p in found:
            self.assertEqual(p.row_global,p.row*2**p.octave)
            self.assertEqual(p.col_global,p.col*2**p.octave)
            self.assertAlmostEqual(p.sigma_global,p.sigma_local*2**p.octave)
            self.assertEqual(p.dog_value,pyramid[p.octave].dogs[p.layer][p.row,p.col])

    def test_taylor_analytic_derivatives_and_symmetry(self):
        t=calculate_dog_derivatives(quadratic_stack(),2,4,4)
        expected_h=np.array([[-2,.12,.06],[.12,-1.4,0],[.06,0,-1.8]])
        np.testing.assert_allclose(t.hessian,expected_h,atol=1e-12)
        np.testing.assert_allclose(t.hessian,t.hessian.T)
        expected_delta=np.array([.25,.2,.1])
        np.testing.assert_allclose(-np.linalg.solve(t.hessian,t.gradient),expected_delta,atol=1e-12)
        np.testing.assert_allclose(calculate_spatial_hessian(quadratic_stack(),2,4,4),expected_h[:2,:2])

    def test_refined_quadratic_and_response(self):
        stack=quadratic_stack()
        kp=refine_keypoint(point(),fake_octave(stack))
        self.assertEqual(kp.status,"refined")
        np.testing.assert_allclose([kp.col_local,kp.row_local,kp.layer_local],[4.25,4.2,2.1],atol=1e-12)
        self.assertAlmostEqual(kp.response,2.,places=12)
        self.assertAlmostEqual(kp.sigma_local,1.6*2**(2.1/3),places=12)

    def test_relocation_and_failures(self):
        stack=quadratic_stack(cx=4.8)
        moved=refine_keypoint(point(),fake_octave(stack))
        self.assertEqual(moved.status,"refined")
        self.assertEqual(moved.col,5)
        self.assertAlmostEqual(moved.col_local,4.8)
        singular=tuple(np.zeros((11,11)) for _ in range(5))
        self.assertEqual(refine_keypoint(point(),fake_octave(singular)).status,"invalid")
        self.assertEqual(refine_keypoint(point(row=1),fake_octave(quadratic_stack(cy=-2))).status,"invalid")
        with self.assertRaises(ValueError): calculate_dog_derivatives(stack,0,4,4)

    def test_contrast_boundary_and_counts(self):
        stack=quadratic_stack()
        octave=fake_octave(stack)
        candidate=point()
        kept=filter_keypoints([candidate],(octave,),contrast=2.0)
        self.assertEqual(kept[0].status,"kept")
        rejected=filter_keypoints([candidate],(octave,),contrast=2.00001)
        self.assertEqual(rejected[0].status,"low contrast")
        _,raw,filtered,_=pipeline(teaching_presets()["Cảnh nhiễu"],with_descriptors=False)
        self.assertEqual(len(raw),len(filtered))
        self.assertTrue(any(p.status=="low contrast" for p in filtered))
        self.assertTrue(set(p.status for p in filtered)<= {"invalid","low contrast","edge","kept"})

    def test_edge_geometry_and_boundary(self):
        self.assertTrue(edge_decision(np.diag([2.,2.]))[0])
        self.assertFalse(edge_decision(np.diag([10.,1.]))[0])
        self.assertAlmostEqual(edge_decision(np.diag([10.,1.]))[3],12.1)
        self.assertFalse(edge_decision(np.diag([1.,0.]))[0])
        self.assertFalse(edge_decision(np.diag([1.,-1.]))[0])
        self.assertFalse(edge_decision(np.array([[np.nan,0],[0,1.]]))[0])

    def test_gradient_orientation_and_weighting(self):
        yy,xx=np.indices((65,65)); ramp=xx/64
        gx,gy=gradient_field(ramp)
        self.assertAlmostEqual(gx[32,32],1/64)
        self.assertAlmostEqual(gy[32,32],0)
        sigmas=tuple(1.6*2**(i/3) for i in range(6))
        octave=Octave(0,tuple(ramp for _ in range(6)),tuple(ramp for _ in range(5)),
                      sigmas,sigmas,tuple(0. for _ in range(6)))
        kp=replace(point(row=32,col=32),row_local=32.,col_local=32.,sigma_local=2.,status="kept")
        info=calculate_orientation_histogram(kp,octave)
        self.assertEqual(info.histogram.shape,(36,))
        self.assertEqual(info.peaks[0],0.)
        center=next(item for item in info.contributions if item[:2]==(32,32))
        far=next(item for item in info.contributions if item[:2]==(32,35))
        self.assertEqual(center[4],1.)
        self.assertLess(far[4],center[4])
        self.assertEqual(len(assign_keypoint_orientations(kp,octave)),len(info.peaks))

    def test_descriptor_layout_trilinear_and_normalization(self):
        yy,xx=np.indices((65,65)); image=(xx+yy)/128
        sigmas=tuple(1.6*2**(i/3) for i in range(6))
        octave=Octave(0,tuple(image for _ in range(6)),tuple(image for _ in range(5)),
                      sigmas,sigmas,tuple(0. for _ in range(6)))
        kp=replace(point(row=32,col=32),row_local=32.,col_local=32.,sigma_local=2.,orientation=10.)
        descriptor=build_sift_descriptor(kp,octave)
        self.assertEqual(descriptor.cells.shape,(4,4,8))
        self.assertEqual(descriptor.raw.shape,(128,))
        self.assertTrue(np.isfinite(descriptor.final).all())
        np.testing.assert_allclose(descriptor.cells.reshape(128),descriptor.raw)
        self.assertEqual(descriptor_index(2,3,7),95)
        self.assertGreater(np.count_nonzero(descriptor.cells[:,:,0]),0)
        self.assertGreater(np.count_nonzero(descriptor.cells[:,:,1]),0)
        self.assertAlmostEqual(np.linalg.norm(descriptor.first_l2),1.)
        self.assertTrue(np.all(descriptor.clipped<=.2))
        self.assertAlmostEqual(np.linalg.norm(descriptor.final),1.)
        patch,valid=sample_oriented_descriptor_patch(kp,octave)
        self.assertEqual(patch.shape,valid.shape)
        self.assertTrue(valid.all())
        rotated=build_sift_descriptor(replace(kp,orientation=45.),octave)
        self.assertFalse(np.allclose(rotated.raw,descriptor.raw))

    def test_zero_descriptor(self):
        a,b,c=normalize_sift_descriptor(np.zeros(128))
        self.assertFalse(a.any() or b.any() or c.any())
        with self.assertRaises(ValueError): descriptor_index(4,0,0)

    def test_orientation_rotation_and_descriptor_frame(self):
        yy,xx=np.indices((65,65))
        source=xx/64
        turned=cv2.rotate(source,cv2.ROTATE_90_CLOCKWISE)
        sigmas=tuple(1.6*2**(i/3) for i in range(6))
        make=lambda a:Octave(0,tuple(a for _ in range(6)),tuple(a for _ in range(5)),
                             sigmas,sigmas,tuple(0. for _ in range(6)))
        kp=replace(point(row=32,col=32),row_local=32.,col_local=32.,sigma_local=2.,status="kept")
        before=calculate_orientation_histogram(kp,make(source))
        after=calculate_orientation_histogram(kp,make(turned))
        self.assertEqual(before.peaks,(0.,))
        self.assertEqual(after.peaks,(90.,))
        a=build_sift_descriptor(replace(kp,orientation=before.peaks[0]),make(source))
        b=build_sift_descriptor(replace(kp,orientation=after.peaks[0]),make(turned))
        np.testing.assert_allclose(a.final,b.final,atol=1e-12)

    def test_secondary_orientations_are_records(self):
        image=teaching_presets()["Blob đa thang"]
        pyramid,raw,filtered,described=pipeline(image)
        kept=[p for p in filtered if p.status=="kept"]
        self.assertGreater(len(described),len(kept))
        key=(described[0].octave,described[0].row,described[0].col)
        same=[p for p in described if (p.octave,p.row,p.col)==key]
        self.assertGreater(len({p.orientation for p in same}),1)
        self.assertTrue(all(p.descriptor is not None for p in same))

    def test_boundary_descriptor_marks_missing_samples(self):
        yy,xx=np.indices((65,65)); image=(xx+yy)/128
        sigmas=tuple(1.6*2**(i/3) for i in range(6))
        octave=Octave(0,tuple(image for _ in range(6)),tuple(image for _ in range(5)),
                      sigmas,sigmas,tuple(0. for _ in range(6)))
        kp=replace(point(row=3,col=3),row_local=3.,col_local=3.,sigma_local=4.,orientation=30.)
        patch,valid=sample_oriented_descriptor_patch(kp,octave)
        self.assertFalse(valid.all())
        self.assertTrue(np.all(patch[~valid]==0))
        descriptor=build_sift_descriptor(kp,octave)
        self.assertTrue(np.isfinite(descriptor.final).all())
        self.assertGreater(len(descriptor.samples),0)

    def test_distance_ratio_and_geometric_truth(self):
        q=np.zeros(128); t=np.array([np.ones(128)*.1,np.ones(128)*.2])
        d=compute_descriptor_distances(q,t)
        np.testing.assert_allclose(d,[np.sqrt(128)*.1,np.sqrt(128)*.2])
        self.assertTrue(apply_lowe_ratio_test([.2,.5],.8).accepted)
        self.assertFalse(apply_lowe_ratio_test([.4,.5],.8).accepted)
        self.assertFalse(apply_lowe_ratio_test([.2,.25],.8).accepted)
        self.assertFalse(apply_lowe_ratio_test([0,0],.8).accepted)
        self.assertIsNone(apply_lowe_ratio_test([.2],.8).ratio)
        a=replace(point(),col_global=4.,row_global=5.)
        b=replace(point(),col_global=7.,row_global=3.)
        error,ok=evaluate_geometric_correspondence(a,b,np.array([[1,0,3],[0,1,-2]]))
        self.assertEqual(error,0.)
        self.assertTrue(ok)

    def test_end_to_end_real_matches(self):
        image=teaching_presets()["Cảnh biến đổi affine"]
        other,matrix=transform_scene(image,20,1.)
        _,_,_,a=pipeline(image)
        _,_,_,b=pipeline(other)
        matches=match_descriptors(a,b)
        self.assertEqual(len(matches),len(a))
        self.assertGreater(sum(m.accepted for m in matches),0)
        self.assertGreater(sum(not m.accepted for m in matches),0)
        selected=next(m for m in matches if m.accepted)
        self.assertEqual(selected.ratio,selected.d1/selected.d2)
        error,_=evaluate_geometric_correspondence(a[selected.query],b[selected.nearest],matrix)
        self.assertTrue(np.isfinite(error))


if __name__=="__main__": unittest.main()
