"""Physical and notebook integration checks for isolated generations 08/09."""
from pathlib import Path
import sys, tempfile, unittest
from itertools import product
import numpy as np
import torch
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT/'MRzero_generations_08_09'),str(ROOT/'MRzero_validation'),str(ROOT/'Experiment')]
from voxel_3d import prepare_voxel_inputs, FiniteVoxel3DACDCModel
from reconstruction_08 import FrameDFTReconstruction
from prepare_acdc_inputs import centered_fft
import continuous_reconstruction as fourier
from inplane_geometry import FrameReconstruction, inplane_matrix, rotate_acquisition_k
from test_experiment_voxel import voxel_fixture
from test_acdc_gpu import event_response
from test_echo_shim_inputs import fixture
from subspin_sampling import _load_source

def model_fixture(coherence=False):
    exp,cal=voxel_fixture(children=1)
    meta=exp['inplane_voxel']
    shape=exp['m0_proxy'].shape
    meta['axes']=['native row','native column','slice normal']
    meta['edge_vectors_m']=np.vstack((meta['edge_vectors_m'],[0.,0.]))
    rng=np.random.default_rng(771)
    meta['b0_gradient_Hz_per_pixel']=np.concatenate((meta['b0_gradient_Hz_per_pixel'],np.zeros(shape+(1,))),axis=-1)
    meta['field_gradient_Hz_per_A_per_pixel']=np.concatenate((meta['field_gradient_Hz_per_A_per_pixel'],rng.normal(size=shape+(1,32))*300),axis=-2)
    corners=np.array(list(product((-.5,.5),repeat=3)))
    meta['field_at_corners_Hz_per_A']=cal['acdc_maps'][...,None,:]+np.einsum('va,pzac->pzvc',corners,meta['field_gradient_Hz_per_A_per_pixel'])
    if coherence:
        exp['multi_echo_coherence']={'width_Hz':np.full(shape,50.),'dispersion_kind':np.ones(shape,dtype=int)}
    return exp,cal

def model(exp,cal):
    return FiniteVoxel3DACDCModel(exp,cal,profile='cpu',precision='float64',sample_chunk=5,
        reconstruction_factory=lambda a,d,t:FrameDFTReconstruction(a,d,t,sample_chunk=5))

class Generation08Tests(unittest.TestCase):
    def test_direct_sum_inverse_convention_and_adjoint(self):
        n=5; fov=.13
        k=(np.arange(n)-n//2)/fov
        kx,ky=np.meshgrid(k,k)
        acq={'kx_mInv':kx,'ky_mInv':ky,'FOV_m':np.array([fov,fov])}
        frame=inplane_matrix(37,True,False)
        rotated=rotate_acquisition_k(acq,matrix=frame)
        fast=FrameReconstruction(rotated,'cpu',torch.float64,frame_matrix=frame)
        direct=FrameDFTReconstruction(rotated,'cpu',torch.float64,frame_matrix=frame,sample_chunk=4)
        rng=np.random.default_rng(770)
        signal=torch.as_tensor(rng.normal(size=(2,n,n))+1j*rng.normal(size=(2,n,n)))
        image=torch.as_tensor(rng.normal(size=(2,n,n))+1j*rng.normal(size=(2,n,n)))
        torch.testing.assert_close(direct.image(signal),fast.image(signal),rtol=1e-13,atol=1e-13)
        torch.testing.assert_close(torch.vdot(direct.image(signal).flatten(),image.flatten()),torch.vdot(signal.flatten(),direct.adjoint(image).flatten()),rtol=1e-13,atol=1e-13)
        self.assertEqual(direct.base.method,'matrix_dft')

    def test_force_direct_switch_prevents_fft_in_plain_reader_operator(self):
        n=4;fov=.1
        k=(np.arange(n)-n//2)/fov
        kx,ky=np.meshgrid(k,k)
        acq={'kx_mInv':kx,'ky_mInv':ky,'FOV_m':np.array([fov,fov])}
        signal=torch.arange(n*n).reshape(n,n).to(torch.complex128)
        expected=fourier.ContinuousReconstruction(acq,'cpu',torch.float64).image(signal)
        with patch.object(fourier,'FORCE_DIRECT_DFT',True), patch.object(torch.fft,'ifft2',side_effect=AssertionError('IFFT forbidden')), patch.object(torch.fft,'fft2',side_effect=AssertionError('FFT forbidden')):
            operator=fourier.ContinuousReconstruction(acq,'cpu',torch.float64,sample_chunk=3)
            self.assertEqual(operator.method,'direct_chunked')
            actual=operator.image(signal)
            torch.testing.assert_close(actual,expected,rtol=1e-13,atol=1e-13)
            operator.adjoint(actual)

    def test_raw_reader_direct_transforms_match_odd_even_centering(self):
        rng=np.random.default_rng(775)
        for shape,axes in [((4,3,6),(0,2)),((5,2,7),(0,2)),((2,3,5,4),(2,3)),((2,3,5,4),(-1,-2)),((3,2,7),(2,))]:
            value=rng.normal(size=shape)+1j*rng.normal(size=shape)
            for inverse in (False,True):
                with patch.object(fourier,'FORCE_DIRECT_DFT',False):
                    expected=centered_fft(value,axes,inverse)
                with patch.object(fourier,'FORCE_DIRECT_DFT',True), patch.object(np.fft,'fftn',side_effect=AssertionError('FFT forbidden')), patch.object(np.fft,'ifftn',side_effect=AssertionError('IFFT forbidden')), patch.object(torch.fft,'fft2',side_effect=AssertionError('FFT forbidden')), patch.object(torch.fft,'ifft2',side_effect=AssertionError('IFFT forbidden')):
                    actual=centered_fft(value,axes,inverse)
                    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
                    restored=centered_fft(actual,axes,not inverse)
                    np.testing.assert_allclose(restored,value,rtol=1e-12,atol=1e-12)

    def test_three_axis_gradient_and_quadrature(self):
        for coherence in (False,True):
            exp,cal=model_fixture(coherence)
            child=model(exp,cal)
            rng=np.random.default_rng(773)
            current=rng.normal(size=(32,3))*.04
            direction=rng.normal(size=current.shape);direction/=np.linalg.norm(direction)
            _,gradient=child.objective_and_grad(current)
            eps=1e-5
            fd=(child.objective_and_grad(current+eps*direction)[0]-child.objective_and_grad(current-eps*direction)[0])/(2*eps)
            np.testing.assert_allclose(np.sum(gradient*direction),fd,rtol=5e-6,atol=2e-9)
            check=child.validate_voxel_quadrature(current,sample_count=5,point_count=7)
            self.assertTrue(check['passed'],check)
            self.assertEqual(child.B_quality.shape[0],16*int(child.quality_mask.sum()))

    def test_signal_matches_displaced_spins_in_three_axes(self):
        exp,cal=model_fixture()
        child=model(exp,cal)
        rng=np.random.default_rng(774)
        current=rng.normal(size=(32,3))*.04
        acq=exp['acquisition'];meta=exp['inplane_voxel']
        active=exp['field']['simulation_mask']&(exp['m0_proxy']>0)
        nodes,weights=np.polynomial.legendre.leggauss(16)
        offsets=np.stack(np.meshgrid(nodes/2,nodes/2,nodes/2,indexing='ij'),axis=-1).reshape(-1,3)
        qweights=np.prod(np.stack(np.meshgrid(weights/2,weights/2,weights/2,indexing='ij'),axis=-1),axis=-1).ravel()
        elapsed=np.asarray(acq['time_s'])-np.asarray(acq['rf_time_s'])
        _,integrated=event_response(current,acq['line_start_s'],acq['line_end_s'],acq['time_s'],exp['current_response']['tau_s'])
        _,before=event_response(current,acq['line_start_s'],acq['line_end_s'],acq['rf_time_s'],exp['current_response']['tau_s'])
        integrated-=before
        position=np.stack((exp['geometry']['X_m'][active],exp['geometry']['Y_m'][active]),axis=-1)[:,None,:]+offsets@meta['edge_vectors_m']
        local_b0=exp['field']['b0Hz'][active,None]+meta['b0_gradient_Hz_per_pixel'][active]@offsets.T
        local_maps=cal['acdc_maps'][active,None,:]+np.einsum('va,pac->pvc',offsets,meta['field_gradient_Hz_per_A_per_pixel'][active])
        amplitude=exp['m0_proxy'][active,None]
        rate=1/exp['relaxation']['T2star_s'][active,None]
        for b0_enabled in (False,True):
            expected=np.empty(elapsed.size,complex)
            for j,t in enumerate(elapsed.ravel()):
                cycles=position@np.array([acq['kx_mInv'].ravel()[j],acq['ky_mInv'].ravel()[j]])+local_maps@integrated[j]
                if b0_enabled:cycles+=local_b0*t
                expected[j]=np.sum(amplitude*qweights*np.exp(-(t-exp['amplitude_reference_time_s'])*rate)*np.exp(-2j*np.pi*cycles))*np.exp(1j*acq['phase_offset_rad'].ravel()[j])
            actual=child.forward(current,b0_enabled=b0_enabled)[1]
            np.testing.assert_allclose(actual,expected.reshape(elapsed.shape),rtol=6e-5,atol=6e-6)

    def test_through_plane_corner_limit(self):
        exp,cal=model_fixture()
        cal['acdc_maps'][:]=1e-6
        meta=exp['inplane_voxel'];meta['field_gradient_Hz_per_A_per_pixel'][:]=0
        meta['field_gradient_Hz_per_A_per_pixel'][...,2,0]=100.
        corners=np.array(list(product((-.5,.5),repeat=3)))
        meta['field_at_corners_Hz_per_A']=cal['acdc_maps'][...,None,:]+np.einsum('va,pzac->pzvc',corners,meta['field_gradient_Hz_per_A_per_pixel'])
        child=model(exp,cal)
        current=np.zeros((32,3));current[0]=1
        self.assertGreater(float((child.B_quality@child.tensor(current)).abs().max()),49.)
        self.assertFalse(child.hardware.diagnostics(current,30.,10.)['checks']['command_field'])
        projected=child.project_currents(current,30.,10.)
        self.assertTrue(child.hardware.diagnostics(projected,30.,10.)['all_limits_met'])

    def test_preparation_uses_physical_full_thickness_and_one_center(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs,fit,source,path,settings=fixture(tmp)
            got=prepare_voxel_inputs(inputs,fit,path,settings,alignment_angle_deg=31,frame_angle_deg=47)
            exp=got['expanded_experiment'];meta=exp['inplane_voxel']
            self.assertEqual(exp['m0_proxy'].shape[1],1)
            np.testing.assert_array_equal(exp['geometry']['Z_m'],0.)
            np.testing.assert_array_equal(meta['b0_gradient_Hz_per_pixel'][...,2],0.)
            np.testing.assert_allclose(meta['field_gradient_Hz_per_A_per_pixel'][...,2,:],np.broadcast_to(.002*200*np.arange(1,33),meta['field_gradient_Hz_per_A_per_pixel'][...,2,:].shape),atol=1e-12)
            corners=np.array(list(product((-.5,.5),repeat=3)))
            xyz=got['xyz_m'][...,None,:]+corners@meta['patient_edge_vectors_m']
            expected=(3+xyz@[100.,170.,200.])[...,None]*np.arange(1,33)
            np.testing.assert_allclose(meta['field_at_corners_Hz_per_A'],expected,atol=2e-12)
            self.assertEqual(got['report']['inplane_voxel_field_limit_corners_per_child'],8)
            np.testing.assert_allclose(exp['m0_proxy'].sum(),fit['first_echo_amplitude'].ravel()[got['parent_flat_index']].sum()/4)
            _load_source.cache_clear()

    def test_preparation_filters_parents_when_full_corner_not_covered(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs,fit,source,path,settings=fixture(tmp)
            original=prepare_voxel_inputs(inputs,fit,path,settings)
            source['mask'][4,4,3]=False
            np.savez(path,**source)
            _load_source.cache_clear()
            filtered=prepare_voxel_inputs(inputs,fit,path,settings)
            self.assertLess(len(filtered['parent_flat_index']),len(original['parent_flat_index']))
            n=len(filtered['parent_flat_index'])
            exp=filtered['expanded_experiment']
            self.assertEqual(exp['subspins']['weights'].shape,(n,1))
            self.assertEqual(exp['geometry']['X_m'].shape,(n,1))
            self.assertEqual(filtered['expanded_calibration']['acdc_maps'].shape,(n,1,32))
            self.assertEqual(filtered['parent_valid_mask'].sum(),n)
            self.assertGreater(filtered['report']['excluded_ACDC_amplitude_fraction_of_baseline'],0.)
            _load_source.cache_clear()


from types import SimpleNamespace
import json
from first_simulation_report import first_simulation_report
from native_b0_model import NativeB0ACDCModel

class ReportTest(unittest.TestCase):
    def make_inputs(self):
        phantom=SimpleNamespace(PD=torch.tensor([1.,2.]),B0=torch.tensor([-2.,3.]),T1=torch.tensor([np.inf,np.inf]),T2=torch.tensor([.1,.2]),T2dash=torch.tensor([np.inf,np.inf]),D=torch.zeros(2),B1=torch.ones(2))
        class Model:
            runtime_info={'native_precision':'float32','mrzero_version':'test'}
            time_s=np.array([[.01,.02],[.03,.04]])
            rf_time_s=np.zeros((2,2))
            TE_s=.025
            amplitude_reference_time_s=.005
            device=torch.device('cpu')
            dtype=torch.float64
            _rate=torch.tensor([10.,5.])
            _amplitude=torch.tensor([1.,2.])
            support=np.array([[True],[True]])
            _native_phantoms={}
            def _phantom(self, b0):
                self._native_phantoms[b0]=phantom
                return phantom
        prepared={'weights':np.ones((2,1)),'native_X_m':np.zeros((2,2)),'parent_valid_mask':np.array([[1,0],[0,1]],bool),'native_B0_Hz':np.array([[-2.,0.],[0.,3.]]),'native_measured_B0_Hz':np.array([[-2.,0.],[0.,3.]]),'report':{'slice_thickness_mm':5.}}
        inputs={'acquisition':{'TE_s':.025,'TR_s':1.,'FOV_m':np.array([.2,.2]),'matrix_size':np.array([2,2])},'scan_metadata':{'TE_s':.025,'TR_s':1.,'header_flip_deg':15.,'image_shape':[2,2],'FOV_m':[.2,.2],'slice_thickness_mm':5.}}
        return Model(),prepared,inputs,phantom
    def test_json_snapshot_without_cache_change(self):
        model,prep,inputs,phantom=self.make_inputs()
        report,rows=first_simulation_report(model,prep,inputs,np.ones((2,2)),np.ones((2,2))*2,np.ones((2,2),complex),'08')
        json.dumps(report,allow_nan=False)
        self.assertEqual(model._native_phantoms,{})
        self.assertEqual(report['native_parameters']['T1']['positive_infinity_count'],2)
        self.assertEqual(report['baseline_vs_scan_magnitude']['gain'],.5)
        self.assertEqual(report['geometry']['z_samples_per_parent'],1)
        self.assertTrue(all(len(row)==3 for row in rows))
    def test_gen09_prefers_actual_native_phantom(self):
        model,prep,inputs,phantom=self.make_inputs()
        model.first_native_phantom=phantom
        model.first_native_simulation={'voxel_size_m':np.array([.001,.001,.005]),'b0_event_min_Hz':-2.}
        def wrong_builder(*args): raise AssertionError('must not use inherited point phantom')
        model._phantom=wrong_builder
        report,rows=first_simulation_report(model,prep,inputs,np.ones((2,2)),np.ones((2,2)),np.ones((2,2),complex),'09')
        self.assertEqual(report['geometry']['native_voxel_size_m'],[.001,.001,.005])
        json.dumps(report,allow_nan=False)



class NativeGeneration09Test(unittest.TestCase):
    def test_existing_optimizer_runs_fresh_native_graphs(self):
        from experiment_optimization import optimize_mrzero_image
        experiment, calibration = model_fixture(coherence=True)
        edges = experiment['inplane_voxel']['edge_vectors_m'][:2]
        experiment['inplane_voxel']['voxel_edge_vectors_m'] = np.vstack((
            np.column_stack((edges, np.zeros(2))), [0., 0., .005]))
        native = NativeB0ACDCModel(experiment, calibration, profile='cpu',
                                  precision='float64', sample_chunk=5)
        count = native._dynamic_calls
        solution = optimize_mrzero_image(native, max_iterations=2,
            total_current_limit_A=30., field_limit_Hz=1000., use_cuda_graph=False,
            field_guidance_weight=.01, field_guidance_scale_Hz=100.)
        self.assertGreater(native._dynamic_calls, count)
        self.assertTrue(np.isfinite(solution['image']).all())
        self.assertLessEqual(solution['optimization_info']['objective'],
                             solution['initial_objective'] + 1e-12)

    def test_native_dynamic_signal_gradient_and_chunk_parity(self):
        e,c = voxel_fixture(children=1)
        meta=e['inplane_voxel']
        edges2=meta['edge_vectors_m']
        meta['edge_vectors_m']=np.vstack((edges2,[0.,0.]))
        meta['voxel_edge_vectors_m']=np.vstack((np.column_stack((edges2,[0.,0.])),[0.,0.,.005]))
        meta['b0_gradient_Hz_per_pixel']=np.concatenate((meta['b0_gradient_Hz_per_pixel'],np.zeros(e['m0_proxy'].shape+(1,))),axis=-1)
        rng=np.random.default_rng(221)
        grad=np.concatenate((meta['field_gradient_Hz_per_A_per_pixel'],rng.normal(size=e['m0_proxy'].shape+(1,32))*200),axis=-2)
        meta['field_gradient_Hz_per_A_per_pixel']=grad
        meta['field_at_corners_Hz_per_A']=c['acdc_maps'][...,None,:]+np.einsum('va,pzac->pzvc',np.array(list(product((-.5,.5),repeat=3))),grad)
        meta['axes']=['native row','native column','slice normal']
        meta['through_plane_B0']='synthetic zero slope'
        width=np.full(e['m0_proxy'].shape,51.)
        kind=(np.arange(width.size).reshape(width.shape)%3).astype(np.int8)
        e['multi_echo_coherence']={'width_Hz':width,'dispersion_kind':kind}
        opts=dict(profile='cpu',precision='float64',sample_chunk=5)
        a=FiniteVoxel3DACDCModel(e,c,**opts)
        b=NativeB0ACDCModel(e,c,**opts)
        b.forward(np.zeros((32,3)))
        assert b.first_native_simulation['B0_enabled'] and b.first_native_simulation['zero_currents']
        currents=rng.normal(size=(32,3))*.04
        for b0 in (True,False):
         for response in (True,False):
          ia,sa=a.forward(currents,b0_enabled=b0,response_enabled=response)
          ib,sb=b.forward(currents,b0_enabled=b0,response_enabled=response)
          err=np.max(abs(sa-sb));rel=np.linalg.norm(sa-sb)/np.linalg.norm(sa)
          print('signal parity B0',b0,'response',response,'max',err,'relative',rel,flush=True)
          np.testing.assert_allclose(sb,sa,rtol=2e-5,atol=2e-6)
        v,g=b.objective_and_grad(currents)
        va,ga=a.objective_and_grad(currents)
        print('objective',v,va,'gradient rel',np.linalg.norm(g-ga)/np.linalg.norm(ga),flush=True)
        np.testing.assert_allclose(v,va,rtol=2e-5,atol=2e-6)
        np.testing.assert_allclose(g,ga,rtol=2e-4,atol=3e-6)
        direction=rng.normal(size=currents.shape);direction/=np.linalg.norm(direction)
        eps=.002
        vp=b._objective_tensor(b._check_currents(currents+eps*direction),gradient=False)[0]
        vm=b._objective_tensor(b._check_currents(currents-eps*direction),gradient=False)[0]
        finite=float((vp-vm)/(2*eps));exact=float(np.sum(g*direction))
        print('directional grad native',exact,'finite difference',finite,flush=True)
        np.testing.assert_allclose(exact,finite,rtol=.003,atol=2e-5)
        _,s0=b.forward(currents)
        b.sample_chunk=15
        _,s1=b.forward(currents)
        np.testing.assert_allclose(s0,s1,rtol=3e-5,atol=3e-6)
        print('chunk parity passed; native calls',b._dynamic_calls,'first',b.first_native_simulation,flush=True)

        # The local executor without extension arguments retains the upstream result.
        from native_pdg import execute_graph as local_execute
        mr0=b._mr0
        rep=mr0.Repetition.zero(3)
        rep.pulse.angle=torch.tensor(np.pi/2,dtype=torch.float32)
        rep.pulse.phase=torch.tensor(np.pi/2,dtype=torch.float32)
        rep.pulse.usage=mr0.PulseUsage.EXCIT
        rep.pulse.selective=False
        rep.event_time[:]=torch.tensor([.001,.001,.001])
        rep.gradm[:]=torch.tensor([[3.,2.,1.],[-1.,4.,-.5],[-1.,-1.,0.]])
        rep.adc_usage[:]=1
        seq=mr0.Sequence([rep],normalized_grads=False)
        phantom=b._dynamic_phantom(True)
        expected=mr0.execute_graph(mr0.compute_graph(seq,phantom,16,0.),seq,phantom,print_progress=False,min_emitted_signal=0.,min_latent_signal=0.)
        actual=local_execute(mr0.compute_graph(seq,phantom,16,0.),seq,phantom,print_progress=False,min_emitted_signal=0.,min_latent_signal=0.)
        np.testing.assert_array_equal(actual.numpy(),expected.numpy())
        print('standard native executor exact parity passed',flush=True)

if __name__ == "__main__":
    unittest.main()
