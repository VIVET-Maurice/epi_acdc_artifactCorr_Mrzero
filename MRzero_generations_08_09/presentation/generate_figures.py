"""Scientific figures and matched reconstruction controls for the selected deck.

All patient figures use measured arrays or freshly simulated arrays. This module
never edits production notebooks or physics. Historical legacy arrays are labeled
as archived, and alternate reconstruction/aperture/phantom tests are explicit.
"""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo
import argparse,csv,json,pickle,sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
PROJECT=PACKAGE.parent
sys.path.insert(0,str(PACKAGE))
from prepare_data import prepare_context,safe,sha256
import continuous_reconstruction as fourier
from inplane_geometry import FrameReconstruction,rotate_acquisition_k
from reconstruction_08 import FrameDFTReconstruction
from measured_image_target import configure_measured_gre_target
from twin_geometry import compare_magnitude

plt.rcParams.update({'font.size':10,'axes.titlesize':11,'svg.fonttype':'none','figure.facecolor':'white'})

@contextmanager
def reconstruction_mode(*,direct=False,matrix=False):
    prior=(fourier.FORCE_DIRECT_DFT,fourier.MATRIX_DFT_ENABLED)
    fourier.FORCE_DIRECT_DFT,fourier.MATRIX_DFT_ENABLED=direct,matrix
    try:yield
    finally:fourier.FORCE_DIRECT_DFT,fourier.MATRIX_DFT_ENABLED=prior

def relative(actual,reference,mask=None):
    a,b=np.asarray(actual),np.asarray(reference)
    if mask is not None:a,b=a[mask],b[mask]
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),np.finfo(float).tiny))

def texture(image,mask):
    a=np.abs(image);m=np.asarray(mask,bool);mu=float(a[m].mean())
    delta=np.r_[(np.diff(a,axis=0)[m[1:]&m[:-1]]),
                (np.diff(a,axis=1)[m[:,1:]&m[:,:-1]])]
    return {'mask_pixels':int(m.sum()),'CV':float(a[m].std()/mu),
            'adjacent_RMS_over_mean':float(np.sqrt(np.mean(delta**2))/mu)}

def image(ax,data,X,Y,title,*,vmin=0,vmax=1,cmap='gray',mask=None):
    artist=ax.pcolormesh(np.asarray(X)*1000,np.asarray(Y)*1000,np.asarray(data),
        shading='auto',cmap=cmap,vmin=vmin,vmax=vmax,rasterized=True)
    ax.set(title=title,xlabel='Common-frame X [mm]',ylabel='Common-frame Y [mm]',aspect='equal')
    if mask is not None and np.any(mask) and not np.all(mask):
        ax.contour(X*1000,Y*1000,mask.astype(float),levels=[.5],colors=['#D97706'],linewidths=.6)
    return artist

class Figures:
    def __init__(self,context):
        self.context=context;self.figures={};self.metrics=[]
        self.folder=HERE/'figures';self.data=HERE/'data'
        self.folder.mkdir(exist_ok=True);self.data.mkdir(exist_ok=True)
    def save(self,fig,name,slides,description,**metadata):
        files={}
        for ext in ('png','svg','pdf'):
            target=self.folder/f'{name}.{ext}'
            fig.savefig(target,dpi=300,bbox_inches='tight',facecolor='white')
            files[ext]=str(target.resolve())
        plt.close(fig)
        self.figures[name]={'slides':slides,'description':description,'files':files,
            'metadata':safe(metadata),'source_inputs':self.context['source_paths'],
            'source_input_sha256':self.context['input_sha256'],
            'git_sha':self.context['provenance']['git_sha'],
            'code_sha256':self.context['provenance']['code_sha256'],
            'device':'cpu','dtype':'float64; MRzero native float32',
            'timestamp':datetime.now(ZoneInfo('Europe/Paris')).isoformat()}
        return files
    def number(self,experiment,condition,metric,value,units='',**extra):
        if value is not None and np.isscalar(value):self.metrics.append(dict(
            experiment=experiment,condition=condition,metric=metric,value=float(value),units=units,**extra))

def reconstruction_controls(context,signal,measured):
    acq=context['prepared']['expanded_experiment']['acquisition'];O=context['frame_matrix']
    with reconstruction_mode():auto=FrameReconstruction(acq,'cpu',torch.float64,frame_matrix=O,sample_chunk=256)
    with reconstruction_mode(direct=True):direct=FrameReconstruction(acq,'cpu',torch.float64,frame_matrix=O,sample_chunk=256)
    with reconstruction_mode():matrix=FrameDFTReconstruction(acq,'cpu',torch.float64,frame_matrix=O,sample_chunk=256)
    native=rotate_acquisition_k(acq,matrix=np.asarray(O).T)
    ny,nx=auto.image_size;fx,fy=np.asarray(native['FOV_m'])
    kx,ky=np.meshgrid((np.arange(nx)-nx//2)/fx,(np.arange(ny)-ny//2)/fy)
    ideal=rotate_acquisition_k(dict(native,kx_mInv=kx,ky_mInv=ky),matrix=O)
    with reconstruction_mode():grid=FrameReconstruction(ideal,'cpu',torch.float64,frame_matrix=O)
    reconstructed={name:operator.numpy_image(signal) for name,operator in
        [('auto',auto),('direct_actual_k',direct),('production_08V2_matrix',matrix),('forced_grid_ifft',grid)]}
    mask=np.ones((ny,nx),bool)
    primary=compare_magnitude(measured,reconstructed['auto'],mask)
    gain=primary['gain'];rows=[]
    pattern_mask=binary_erosion(np.asarray(context['prepared']['parent_valid_mask'],bool),iterations=3)
    for name,candidate in reconstructed.items():
        row={'operator':name,'complex_relative_image_error':relative(candidate,reconstructed['auto']),
             'magnitude_difference_NRMSE_percent':100*relative(np.abs(candidate),np.abs(reconstructed['auto'])),
             'OFF_NRMSE_percent':compare_magnitude(measured,candidate,mask,gain=gain)['NRMSE_percent'],
             'pattern':texture(candidate,pattern_mask)}
        rows.append(row)
    # Exact Cartesian sanity: asymmetric complex object, no scanner data/physics.
    rng=np.random.default_rng(20261111);n=16;fov=.2
    xx,yy=np.meshgrid((np.arange(n)-n//2)/fov,(np.arange(n)-n//2)/fov)
    synthetic=dict(kx_mInv=xx,ky_mInv=yy,FOV_m=np.array([fov,fov]))
    known=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
    A=np.exp(-2j*np.pi*(xx.ravel()[:,None]*((np.indices((n,n))[1].ravel()-n//2)*fov/n)[None,:]
                       +yy.ravel()[:,None]*((np.indices((n,n))[0].ravel()-n//2)*fov/n)[None,:]))
    synthetic_signal=(A@known.ravel()).reshape(n,n)
    with reconstruction_mode():fft=fourier.ContinuousReconstruction(synthetic,'cpu',torch.float64)
    with reconstruction_mode(direct=True):idft=fourier.ContinuousReconstruction(synthetic,'cpu',torch.float64)
    sfft,sidft=fft.numpy_image(synthetic_signal),idft.numpy_image(synthetic_signal)
    sanity={'size':[n,n],'seed':20261111,'FFT_vs_direct_relative_L2':relative(sfft,sidft),
            'FFT_vs_known_relative_L2':relative(sfft,known),'direct_vs_known_relative_L2':relative(sidft,known)}
    np.savez_compressed(HERE/'data/reconstruction_controls.npz',kdata=signal,measured_off=measured,
        **reconstructed,X_m=auto.X_m,Y_m=auto.Y_m,mask=mask,shared_gain=gain,pattern_mask=pattern_mask,
        synthetic_known=known,synthetic_kdata=synthetic_signal,synthetic_fft=sfft,synthetic_direct=sidft)
    result={'rows':rows,'shared_OFF_gain':gain,'primary_mask':'full EPI FOV, all finite pixels',
        'mask_pixels':int(mask.sum()),'sanity':sanity,
        'pattern_metric_mask':'Retained parent support eroded three iterations with scipy default connectivity; same fixed mask for all reconstructions. Object structure is not removed by these metrics.',
        'operators':{'auto':auto.report,'direct_actual_k':direct.report,
                     'production_08V2_matrix':matrix.report,'forced_grid_ifft':grid.report},
        'interpretation':'Automatic actual-coordinate and direct-adjoint comparison uses EXACTLY the same complex k-space. Production 08V2 is the current matrix inverse, a distinct operator. Forced nominal-grid IFFT is a separate diagnostic.'}
    (HERE/'data/reconstruction_controls.json').write_text(json.dumps(safe(result),indent=2))
    return result,reconstructed,auto

def main_figures(context,registry):
    eq=np.load(HERE/'data/equivalence.npz')
    inputs=context['inputs'];measured=np.asarray(inputs['scan'])
    signal=eq['baseline_kdata08'];baseline=eq['baseline_image08']
    comparisons,images,operator=reconstruction_controls(context,signal,measured)
    X,Y=operator.X_m,operator.Y_m
    mask=np.ones(measured.shape,bool);off=compare_magnitude(measured,baseline,mask)
    g=off['gain'];scale=float(measured.max());residual=(g*np.abs(baseline)-measured)/scale
    np.savez_compressed(HERE/'data/off_fidelity.npz',measured_off=measured,simulated_off=baseline,
        simulated_off_scaled=g*np.abs(baseline),residual=residual,mask=mask,gain=g,scale=scale,X_m=X,Y_m=Y)
    fig,axes=plt.subplots(1,3,figsize=(12,3.25),layout='constrained')
    panels=[measured/scale,g*np.abs(baseline)/scale,residual]
    vmax=max(1.,float(panels[1].max()));rmax=float(np.abs(residual).max())
    for ax,a,title,signed in zip(axes,panels,['Measured EPI OFF','Fresh Gen08 simulated OFF',f'Simulation − measured\nNRMSE {off["NRMSE_percent"]:.3f}%'],[False,False,True]):
        im=image(ax,a,X,Y,title,vmin=-rmax if signed else 0,vmax=rmax if signed else vmax,cmap='RdBu_r' if signed else 'gray')
        fig.colorbar(im,ax=ax,label='Magnitude / measured OFF peak' if not signed else 'Signed error / OFF peak',shrink=.8)
    registry.save(fig,'slide03_off_validation',[3], 'Fresh uncalibrated OFF prediction versus independently measured EPI.',
        mask='full FOV, 16384 finite pixels',gain=g,scale=scale,NRMSE_percent=off['NRMSE_percent'],baseline_calibration='before nuisance calibration')
    # Registered GRE target only resamples measured data; no forward model is introduced.
    proxy=SimpleNamespace(reconstruction=operator,device=torch.device('cpu'),dtype=torch.float64,
        complex_dtype=torch.complex128,runtime_info={})
    target=configure_measured_gre_target(proxy,inputs['gre'],baseline_image=baseline,
        measured_baseline_image=measured,frame_matrix=context['frame_matrix'],threshold_fraction=0.)
    gre=target['measured_reference_magnitude'];cover=target['coverage_mask']
    exp_gain=target['report']['experimental_baseline_to_GRE_gain']
    common=float(gre.max());sim=g*exp_gain*np.abs(baseline)/common;obs=exp_gain*measured/common
    gre=gre/common;vmax=max(float(gre.max()),float(sim.max()),float(obs.max()))
    diff=sim-obs;rmax=float(np.abs(diff).max())
    fig,axes=plt.subplots(1,4,figsize=(13,3.0),layout='constrained')
    for ax,a,title,signed in zip(axes,[np.where(cover,gre,np.nan),obs,sim,diff],
        ['Measured GRE\nreference','Measured EPI\nOFF','Fresh Gen08\nsimulated OFF','Simulated − measured\nEPI OFF'],[False,False,False,True]):
        im=image(ax,a,X,Y,title,vmin=-rmax if signed else 0,vmax=rmax if signed else vmax,cmap='RdBu_r' if signed else 'gray')
        fig.colorbar(im,ax=ax,label='Magnitude / GRE peak' if not signed else 'Error / GRE peak',shrink=.7)
    registry.save(fig,'slide10_gre_off_comparison',[10], 'GRE and EPI comparison, one shared GRE normalization and fixed OFF gains.',
        mask='OFF NRMSE full FOV; GRE missing coverage shown blank',OFF_gain=g,EPI_to_GRE_gain=exp_gain,
        NRMSE_percent=off['NRMSE_percent'],GRE_coverage_pixels=int(cover.sum()),all_grayscale_limits=[0,vmax])
    # Main reconstruction test: no new forward simulation or per-image gain fit.
    fig,axes=plt.subplots(1,4,figsize=(13,3.05),layout='constrained')
    delta=g*(np.abs(images['direct_actual_k'])-np.abs(images['auto']))/scale
    vmax=max(float(g*np.abs(images[name]).max()/scale) for name in
             ('auto','direct_actual_k','production_08V2_matrix'))
    dmax=max(float(np.abs(delta).max()),1e-16)
    for ax,a,title,signed in zip(axes,[g*np.abs(images['auto'])/scale,g*np.abs(images['direct_actual_k'])/scale,
        g*np.abs(images['production_08V2_matrix'])/scale,delta],
        ['08 automatic\n'+comparisons['operators']['auto']['method'],
         'Pulseq-k direct\nadjoint IDFT','Current 08 V2:\nmatrix inverse','Direct − automatic\nmagnitude'],[False,False,False,True]):
        im=image(ax,a,X,Y,title,vmin=-dmax if signed else 0,vmax=dmax if signed else vmax,cmap='RdBu_r' if signed else 'gray')
        fig.colorbar(im,ax=ax,label='Magnitude / OFF peak' if not signed else 'Difference / OFF peak',shrink=.7)
    registry.save(fig,'slide08_reconstruction_control',[8], 'Same complex k-space; automatic actual-k, direct adjoint, and current matrix inverse.',
        mask='full FOV',gain=g,comparisons=comparisons['rows'],synthetic_sanity=comparisons['sanity'])
    fig,axes=plt.subplots(1,4,figsize=(13,3.05),layout='constrained')
    matched=[('Measured EPI\nOFF',measured/scale),('08 automatic\nrow-separable',g*np.abs(baseline)/scale),
             ('08 V2\nmatrix inverse',g*np.abs(images['production_08V2_matrix'])/scale),
             ('09 local\nnative executor',g*np.abs(eq['baseline_image09'])/scale)]
    shared=max(float(a.max()) for _,a in matched)
    for ax,(title,a) in zip(axes,matched):
        im=image(ax,a,X,Y,title,vmax=shared);fig.colorbar(im,ax=ax,label='Magnitude / OFF peak',shrink=.7)
    nrmse09=compare_magnitude(measured,eq['baseline_image09'],mask,gain=g)['NRMSE_percent']
    registry.save(fig,'matched_generations_off',[3,10,11], 'Same prepared object and zero currents; three generation images share one fixed 08 reporting gain.',
        mask='full finite FOV',gain=g,OFF_NRMSE08_percent=off['NRMSE_percent'],OFF_NRMSE09_percent=nrmse09)
    registry.number('scanner_fidelity','fresh09_uncalibrated','OFF_NRMSE_percent',nrmse09,'%',mask_pixels=mask.size,gain=g)
    # Forced-grid IFFT is distinct from the existing actual-coordinate automatic operator.
    fig,axes=plt.subplots(1,3,figsize=(11,3.05),layout='constrained')
    common_max=max(float(np.abs(images[name]).max()) for name in ('forced_grid_ifft','direct_actual_k'))
    for index,(ax,a,title) in enumerate(zip(axes,[np.abs(images['forced_grid_ifft']),np.abs(images['direct_actual_k']),
        np.abs(images['forced_grid_ifft']-images['direct_actual_k'])],
        ['Forced Cartesian IFFT (diagnostic)','Pulseq-coordinate direct adjoint','Absolute complex difference'])):
        im=image(ax,a,X,Y,title,vmax=common_max if index<2 else float(np.max(a)))
        fig.colorbar(im,ax=ax,label='Simulation units',shrink=.7)
    registry.save(fig,'reconstruction_forced_grid',[8,13], 'Forced-grid mismatch control; no gain fitting or trajectory rounding in production.',comparisons=comparisons['rows'])
    for row in comparisons['rows']:
        for key in ('complex_relative_image_error','magnitude_difference_NRMSE_percent','OFF_NRMSE_percent'):
            registry.number('reconstruction',row['operator'],key,row[key],'%' if 'percent' in key else 'relative L2',mask_pixels=mask.size,gain=g)
    registry.number('scanner_fidelity','fresh08_uncalibrated','OFF_NRMSE_percent',off['NRMSE_percent'],'%',mask_pixels=mask.size,gain=g)
    registry.number('reconstruction','exact_cartesian_sanity','FFT_vs_direct_relative_L2',comparisons['sanity']['FFT_vs_direct_relative_L2'],'relative L2')
    return {'off':off,'reconstruction':comparisons,'GRE_target_report':target['report'],
        'GRE_coverage_pixels':int(cover.sum()),'OFF_mask_pixels':mask.size,'comparison_stage':'before EPI nuisance calibration; no image registration'},X,Y

def geometry_figures(context,registry):
    acq=context['prepared']['expanded_experiment']['acquisition'];O=context['frame_matrix']
    t=np.asarray(acq['time_s']);rf=np.asarray(acq['rf_time_s']);kx=np.asarray(acq['kx_mInv']);ky=np.asarray(acq['ky_mInv'])
    chronological=np.argsort(t.ravel());ny,nx=t.shape;fov=acq['FOV_m']
    native=rotate_acquisition_k(acq,matrix=O.T)
    idealx,idealy=np.meshgrid((np.arange(nx)-nx//2)/fov[0],(np.arange(ny)-ny//2)/fov[1])
    actual_error=np.hypot(native['kx_mInv']-idealx,native['ky_mInv']-idealy)
    dt=np.diff(t.ravel()[chronological]);row_polarity=np.sign(t[:,-1]-t[:,0])
    fig,axes=plt.subplots(1,3,figsize=(12,3.2),layout='constrained')
    dots=axes[0].scatter(kx.ravel()[chronological],ky.ravel()[chronological],c=(t-rf).ravel()[chronological]*1000,s=.8,rasterized=True)
    fig.colorbar(dots,ax=axes[0],label='Time after RF [ms]',shrink=.7)
    axes[0].set(xlabel=r'$k_x$ [cycles/m]',ylabel=r'$k_y$ [cycles/m]',title='Pulseq k at imaging ADC centers',aspect='equal')
    for row in (0,1,64,65,126,127):axes[1].plot(np.arange(nx),(t[row]-rf[row])*1000,label=f'row {row}')
    axes[1].set(xlabel='Stored readout index',ylabel='Time after RF [ms]',title='Odd/even reversal retains ADC time');axes[1].legend(fontsize=7,ncol=2)
    axes[2].plot(np.arange(ny),np.max(actual_error,axis=1));axes[2].set(xlabel='Stored phase row',ylabel='Maximum coordinate error [cycles/m]',title='Pulseq versus forced Cartesian grid')
    report={'ADC_samples':t.size,'ADC_elapsed_min_ms':float((t-rf).min()*1000),'ADC_elapsed_max_ms':float((t-rf).max()*1000),
        'sequence_TE_ms':acq['TE_s']*1000,'sequence_TR_ms':acq['TR_s']*1000,'dwell_us':acq['dwell_s']*1e6,
        'max_actual_minus_nominal_k_mInv':float(actual_error.max()),'row_time_directions':np.unique(row_polarity).tolist(),
        'chronological_dwell_min_us':float(dt.min()*1e6),'chronological_dwell_median_us':float(np.median(dt)*1e6),
        'unique_imaging_ADC_times':len(np.unique(t)),'RF_histories':len(np.unique(rf)),
        'frame_matrix':O.tolist(),'FOV_m':list(fov),'slice_thickness_mm':context['prepared']['report']['slice_thickness_mm'],
        'trajectory_source':'Nominal gradient trajectory calculated from Pulseq at ADC centers; no independently measured gradient trajectory is available.'}
    np.savez_compressed(HERE/'data/trajectory_timing.npz',kx_mInv=kx,ky_mInv=ky,time_s=t,rf_time_s=rf,
        chronological_order=chronological,actual_minus_nominal_k_mInv=actual_error,frame_matrix=O)
    registry.save(fig,'slide13_trajectory_timing',[4,13], 'Actual k-space and sorted/reversed ADC times; no physical samples discarded.',**report)
    (HERE/'data/trajectory_timing.json').write_text(json.dumps(safe(report),indent=2))
    return report

def controlled_figures(context,registry):
    eq=np.load(HERE/'data/equivalence.npz');report=json.loads((HERE/'data/equivalence.json').read_text())
    rf=np.asarray(context['prepared']['expanded_experiment']['acquisition']['rf_time_s'])
    X,Y=eq['X_m'],eq['Y_m'];t=(eq['time_s']-rf).ravel()*1000;order=eq['chronological_ADC_order']
    # Magnitude of complex image residual, not a difference of magnitudes.
    residual=np.abs(eq['dynamic_image09']-eq['dynamic_image08']);peak=float(np.abs(eq['dynamic_image08']).max())
    fig,axes=plt.subplots(1,2,figsize=(6.1,2.7),layout='constrained')
    im=image(axes[0],residual/peak,X,Y,'Absolute complex image residual\n09 − 08, nonzero currents',vmax=float(residual.max()/peak))
    fig.colorbar(im,ax=axes[0],label='Residual / image peak',shrink=.7)
    sr=np.abs(eq['dynamic_signal09']-eq['dynamic_signal08']).ravel()
    axes[1].plot(t[order],sr[order],linewidth=.5);axes[1].set(xlabel='Time after RF [ms]',ylabel='Absolute complex signal residual [a.u.]',title='All 16,384 ADC samples')
    registry.save(fig,'slide11_signal_equivalence',[11], 'Matched 08/09 complex residuals with identical dynamic currents and actual-k reconstruction.',metrics=report)
    for row in report['rows']:
        registry.number('implementation_equivalence',row['case'],'complex_signal_relative_L2',row['signal_complex_relative_L2'],'relative L2')
        registry.number('implementation_equivalence',row['case'],'complex_image_relative_L2',row['image_complex_relative_L2'],'relative L2')
    uniform=np.load(HERE/'data/uniform_phantom.npz');ur=json.loads((HERE/'data/uniform_phantom.json').read_text())
    a,b=np.abs(uniform['B0_zero_image']),np.abs(uniform['measured_B0_image']);vmax=max(float(a.max()),float(b.max()));delta=b-a
    fig,axes=plt.subplots(1,3,figsize=(11.5,3.0),layout='constrained')
    for ax,v,title,signed in zip(axes,[a,b,delta],['Uniform phantom: B0 = 0','Same phantom: measured B0','Magnitude difference'],[False,False,True]):
        im=image(ax,v,uniform['X_m'],uniform['Y_m'],title,vmin=-float(abs(delta).max()) if signed else 0,
            vmax=float(abs(delta).max()) if signed else vmax,cmap='RdBu_r' if signed else 'gray',mask=uniform['interior_mask'])
        fig.colorbar(im,ax=ax,label='Shared B0=0 interior normalization',shrink=.7)
    registry.save(fig,'uniform_phantom_b0_control',[2,8,14], 'Controlled homogeneous phantom isolates B0; production object is unchanged.',metrics=ur)
    for case in ('B0_zero','measured_B0'):
        for key,value in ur[case].items():registry.number('uniform_phantom',case,key,value,'%' if 'percent' in key else '')
    aperture=None
    if (HERE/'data/adc_window.npz').exists():
        ad=np.load(HERE/'data/adc_window.npz');aperture=json.loads((HERE/'data/adc_window.json').read_text())
        a=np.abs(ad['center_native_image']);b=np.abs(ad['window5_image']);s=float(a.max());delta=(b-a)/s;vmax=max(float(a.max()/s),float(b.max()/s))
        fig,axes=plt.subplots(1,3,figsize=(11,3.0),layout='constrained')
        for ax,v,title,signed in zip(axes,[a/s,b/s,delta],['Production: center-of-dwell sample','Hypothetical rectangular 5 μs aperture','Window − center magnitude'],[False,False,True]):
            im=image(ax,v,ad['X_m'],ad['Y_m'],title,vmin=-float(abs(delta).max()) if signed else 0,
                vmax=float(abs(delta).max()) if signed else vmax,cmap='RdBu_r' if signed else 'gray')
            fig.colorbar(im,ax=ax,label='Center-sampled peak normalization',shrink=.7)
        registry.save(fig,'slide12_adc_window',[12], 'Separate rectangular-aperture diagnostic; scanner receive filter remains unknown.',metrics=aperture)
        for row in aperture['quadrature_rows']:
            for key in ('complex_signal_relative_change','magnitude_image_NRMSE_percent'):
                registry.number('ADC_window',f'Gauss order {row["quadrature_order"]}',key,row[key],'%' if 'percent' in key else 'relative L2')
    return report,ur,aperture

def sampling_controls_figure(context,registry):
    """Full-width two-row slide figure; keep raw controls readable on one backup."""
    u=np.load(HERE/'data/uniform_phantom.npz');a=np.load(HERE/'data/adc_window.npz')
    uz,ub=np.abs(u['B0_zero_image']),np.abs(u['measured_B0_image'])
    scale=float(np.abs(a['center_native_image']).max())
    center,window=np.abs(a['center_native_image'])/scale,np.abs(a['window5_image'])/scale
    fig,axes=plt.subplots(2,3,figsize=(12,4.5),layout='constrained',sharex=True,sharey=True)
    for row,(left,right,titles,normalization) in enumerate([
        (uz,ub,['Uniform: B0 = 0','Uniform: measured B0','B0 magnitude difference'],'Shared B0=0 interior scale'),
        (center,window,['ADC center','Rectangular 5 μs window','Window − center'],'Magnitude / center peak')]):
        delta=right-left;maximum=max(float(left.max()),float(right.max()));dmax=float(abs(delta).max())
        for col,value in enumerate((left,right,delta)):
            signed=col==2
            artist=image(axes[row,col],value,u['X_m'],u['Y_m'],titles[col],
                vmin=-dmax if signed else 0,vmax=dmax if signed else maximum,
                cmap='RdBu_r' if signed else 'gray',mask=u['interior_mask'] if row==0 else None)
            axes[row,col].set_xlabel('X [mm]' if row==1 else '')
            axes[row,col].set_ylabel('Y [mm]' if col==0 else '')
            axes[row,col].set_xticks([-50,0,50]);axes[row,col].set_yticks([-50,0,50])
            axes[row,col].title.set_fontsize(11);axes[row,col].tick_params(labelsize=9)
            if col==1:
                cb=fig.colorbar(artist,ax=axes[row,:2],shrink=.86,pad=.02)
                cb.set_label(normalization,fontsize=9);cb.ax.tick_params(labelsize=9)
            elif col==2:
                cb=fig.colorbar(artist,ax=axes[row,col],shrink=.86,pad=.02)
                cb.set_label('Signed magnitude difference',fontsize=9);cb.ax.tick_params(labelsize=9)
    registry.save(fig,'slide12_sampling_controls',[12], 'Raw homogeneous B0 and zero-current hypothetical aperture diagnostics; two separate controls, no production model change.',
        uniform_mask='retained support eroded three times, fixed across top row',
        normalization='top: one B0=0 interior gain; bottom: shared native center peak',
        receiver_filter='unknown; rectangular aperture is hypothetical',currents_A=0,
        B0_change='center field and known local B0 slopes changed together',
        difference_maps='signed differences of magnitude, independent row color limits')

def benchmark_figures(registry):
    report=json.loads((HERE/'data/runtime_benchmarks.json').read_text());records=report['records']
    def panel(ax,rows,title,compact=False,log=False):
        for i,row in enumerate(rows):
            value=row['median_seconds'];lo=value-row['q25_seconds'];hi=row['q75_seconds']-value
            ax.errorbar(i,value,yerr=[[lo],[hi]],fmt='o',color='#1D4E89',capsize=3)
            stagger=compact and row['operation']=='reconstruction_only_production' and i==1
            ax.annotate(f'{value:.2g}' if compact else f'{value:.3g}',(i,value),
                xytext=(0,19 if stagger else 5),textcoords='offset points',ha='center',fontsize=9 if compact else 8,
                arrowprops=dict(arrowstyle='-',lw=.5,color='#777777') if stagger else None)
        labels={'08':'08','08V2':'08 V2','09':'09','comparator_actual_k_direct_adjoint':'Adjoint',
                'comparator_forced_nominal_grid_IFFT':'FFT'}
        ax.set_xticks(np.arange(len(rows)),[labels.get(r['generation'],r['generation']) for r in rows],rotation=25,ha='right',fontsize=9)
        ax.set(ylabel='Time [s]',title=title);ax.grid(axis='y',alpha=.2)
        ax.title.set_fontsize(10 if compact else 11);ax.yaxis.label.set_size(10)
        ax.tick_params(axis='y',labelsize=9)
        ax.margins(x=.22)
        if log:ax.set_yscale('log');ax.margins(y=.35)
        else:ax.set_ylim(0,max(r['q75_seconds'] for r in rows)*1.35)
    for row in records:
        registry.number('runtime',row['generation']+'/'+row['operation'],'median',row['median_seconds'],'s')
        registry.number('runtime',row['generation']+'/'+row['operation'],'IQR',row['IQR_seconds'],'s')
    operations=list(dict.fromkeys(r['operation'] for r in records))
    fig,axes=plt.subplots(2,int(np.ceil(len(operations)/2)),figsize=(14.5,6),layout='constrained',squeeze=False)
    for ax,operation in zip(axes.ravel(),operations):
        panel(ax,[r for r in records if r['operation']==operation],operation.replace('_',' '))
    for ax in axes.ravel()[len(operations):]:ax.set_visible(False)
    registry.save(fig,'runtime_all_stages',[11], 'All measured timing stages separately reported; median and IQR.',metrics=report)
    fig,axes=plt.subplots(2,2,figsize=(6.4,4.1),layout='constrained')
    for ax,operation,title in zip(axes.ravel()[:3],
        ('cold_baseline_available','warm_forward_and_reconstruction','warm_loss_and_gradient'),
        ('Fresh model +\nfirst OFF','Warm forward +\nreconstruction','Warm loss + gradient')):
        panel(ax,[r for r in records if r['operation']==operation],title,compact=True)
    recon=[r for r in records if r['operation'].startswith('reconstruction_only')]
    panel(axes[1,1],recon,'Reconstruction only\n(log scale)',compact=True,log=True)
    registry.save(fig,'slide11_runtime_benchmarks',[11], 'Same prepared object, CPU/device/precision; median and IQR. Setup includes constructor plus first OFF; adjoint/FFT are explicit comparators, matrix inverse remains 08 V2.',metrics=report)
    return report

def coherence_figure(context,registry):
    q=np.linspace(-3,3,601);nodes,weights=np.polynomial.legendre.leggauss(64)
    numerical=(np.exp(-2j*np.pi*q[:,None]*nodes[None,:]/2)*weights[None,:]/2).sum(1)
    analytic=np.sinc(q)
    fig,axes=plt.subplots(1,2,figsize=(8,2.8),layout='constrained')
    axes[0].plot(q,analytic,label='Analytic sinc(q)');axes[0].plot(q[::20],numerical.real[::20],'.',label='64-point quadrature')
    axes[0].set(xlabel='Phase slope across full voxel [cycles]',ylabel='Signed coherence',title='Uniform box: one-axis integral');axes[0].legend(fontsize=8)
    axes[1].plot(q,analytic**3);axes[1].set(xlabel='q₁ = q₂ = q₃ [cycles]',ylabel='Signed product coherence',title='Three independent voxel axes')
    error=float(np.max(np.abs(numerical-analytic)))
    registry.save(fig,'slide07_sinc_integral_check',[7], 'Numerical integral of a uniform voxel; theory control, not patient data.',quadrature_order=64,maximum_absolute_error=error)
    np.savez_compressed(HERE/'data/coherence_integral.npz',q_cycles=q,analytic=analytic,numerical=numerical)
    return {'quadrature_order':64,'maximum_absolute_integral_error':error}

def input_map_figures(context,registry):
    """Show measured/fitted scalar inputs without treating them as receiver maps."""
    prepared=context['prepared'];exp=prepared['expanded_experiment']
    eq=np.load(HERE/'data/equivalence.npz');X,Y=eq['X_m'],eq['Y_m']
    indices=np.asarray(prepared['parent_flat_index']).ravel()
    def parent(values):
        out=np.full(X.size,np.nan);out[indices]=np.asarray(values).ravel();return out.reshape(X.shape)
    maps={'B0_Hz':parent(exp['field']['b0Hz']),
          'effective_amplitude':parent(exp['m0_proxy']),
          'residual_R2_sInv':parent(1./np.asarray(exp['relaxation']['T2star_s']))}
    fig,axes=plt.subplots(1,3,figsize=(11.5,3.0),layout='constrained')
    for ax,(name,value),title,unit in zip(axes,maps.items(),
            ['Measured B0 at retained centers','Effective fitted amplitude','Fitted residual exponential rate'],
            ['Hz','Amplitude proxy [a.u.]','s⁻¹']):
        if name=='B0_Hz':lo=-float(np.nanmax(abs(value)));hi=-lo;cmap='RdBu_r'
        else:lo=0;hi=float(np.nanmax(value[np.isfinite(value)]));cmap='gray'
        im=image(ax,value,X,Y,title,vmin=lo,vmax=hi,cmap=cmap)
        fig.colorbar(im,ax=ax,label=unit,shrink=.7)
    registry.save(fig,'input_maps',[4,14], 'Registered measured B0 and fitted scalar amplitude/rate; no isolated receiver sensitivities.',
        mask='Only retained centers shown; excluded pixels blank',amplitude='Effective RSS-derived signal proxy, not calibrated proton density')
    np.savez_compressed(HERE/'data/input_maps.npz',**maps,X_m=X,Y_m=Y)
    return {'retained_voxels':len(indices),'amplitude_is_true_PD':False}

def local_q_figures(context,registry):
    arrays=np.load(HERE/'data/local_q.npz');report=json.loads((HERE/'data/local_q.json').read_text())
    # One ADC, two current cases, identical scale per local axis.
    zero,dynamic=arrays['q_baseline'][:,1],arrays['q_dynamic'][:,1]
    fig,axes=plt.subplots(2,4,figsize=(13.5,6.2),layout='constrained')
    for row,q in enumerate((zero,dynamic)):
        for axis in range(3):
            maximum=max(float(np.nanmax(abs(zero[axis]))),float(np.nanmax(abs(dynamic[axis]))),1e-12)
            im=image(axes[row,axis],q[axis],arrays['X_m'],arrays['Y_m'],
                ('Zero currents' if row==0 else 'Diagnostic currents')+f': q{axis+1}',
                vmin=-maximum,vmax=maximum,cmap='RdBu_r')
            fig.colorbar(im,ax=axes[row,axis],label='Cycles across full voxel',shrink=.7)
        F=np.prod(np.sinc(q),axis=0)
        im=image(axes[row,3],F,arrays['X_m'],arrays['Y_m'],
            'Signed three-sinc factor',vmin=-1,vmax=1,cmap='RdBu_r')
        fig.colorbar(im,ax=axes[row,3],label='Signed coherence',shrink=.7)
    elapsed=float(report['elapsed_s'][1])*1000
    fig.suptitle(f'ADC nearest k = 0: {elapsed:.4f} ms after RF; local linear voxel slopes',fontsize=12)
    registry.save(fig,'local_q_maps',[7,14], 'Local per-voxel phase slopes and signed box coherence; deterministic currents are not optimized.',
        units='cycles across one full native voxel',sample_label=report['sample_labels'][1],elapsed_ms=elapsed,
        normal_B0_gradient='assumed zero, not measured',current_role='diagnostic only',details=report)
    return report

def optimization_replay_figure(context,registry):
    path=HERE/'data/archived_optimization_replay.json'
    if not path.exists():return None
    report=json.loads(path.read_text());data=np.load(path.with_suffix('.npz'))
    g=float(report['fixed_simulator_to_GRE_gain']);scale=float(data['measured_GRE'].max())
    panels=[data['measured_GRE']/scale,g*np.abs(data['baseline_image'])/scale,
            g*np.abs(data['archived_image'])/scale,g*np.abs(data['quantized_image'])/scale]
    maximum=max(float(np.max(a)) for a in panels)
    fig,axes=plt.subplots(1,4,figsize=(13,3.0),layout='constrained')
    for ax,a,title in zip(axes,panels,['Measured GRE target','Fresh model: zero current',
        'Saved currents: fresh offline replay','Saved quantized currents: replay']):
        im=image(ax,a,data['X_m'],data['Y_m'],title,vmax=maximum,mask=data['mask'])
        fig.colorbar(im,ax=ax,label='Shared measured GRE peak',shrink=.7)
    registry.save(fig,'archived_current_replay',[], 'Supplementary notebook evidence: saved commands replayed on fresh object; no new optimization or verified ACDC-ON scan linkage.',
        gain=g,mask_pixels=report['mask_pixels'],details=report)
    registry.number('offline_current_replay','zero','GRE_objective_NRMSE_percent',
        report['baseline_vs_measured_GRE_NRMSE_percent'],'%',mask_pixels=report['mask_pixels'],gain=g)
    for row in report['rows']:
        registry.number('offline_current_replay',row['case'],'GRE_objective_NRMSE_percent',
            row['simulated_prediction_vs_measured_GRE_NRMSE_percent'],'%',mask_pixels=report['mask_pixels'],gain=g)
    return report

def validate_controlled_results(context):
    """Reject absent or mismatched result caches before assembling any figures."""
    for name in ('equivalence','uniform_phantom','runtime_benchmarks','adc_window'):
        path=HERE/'data'/f'{name}.json'
        if not path.exists():raise FileNotFoundError(f'Missing {path}. Set RUN_EXPENSIVE=True.')
        metadata=json.loads(path.read_text());provenance=metadata['provenance']
        if provenance['input_sha256']!=context['input_sha256']:
            raise RuntimeError(f'{name} input hashes differ from current prepared data. Rerun expensive calculations.')
        if provenance['prepared_provenance']['code_sha256']!=context['provenance']['code_sha256']:
            raise RuntimeError(f'{name} source hashes differ from prepared data. Rerun expensive calculations.')
        if provenance.get('runner_sha256')!=sha256(HERE/'controlled_experiments.py'):
            raise RuntimeError(f'{name} controlled-experiment code changed. Rerun expensive calculations.')
        cache=provenance.get('context_cache')
        if cache and cache.get('sha256')!=sha256(HERE/'cache/prepared_context.pkl'):
            raise RuntimeError(f'{name} prepared-object cache differs. Rerun expensive calculations.')

def add_receiver_and_legacy(context,registry):
    from receiver_audit import generate_legacy_pattern_figure,run_receiver_audit
    legacy=generate_legacy_pattern_figure(PROJECT,HERE);receiver=run_receiver_audit(context,HERE,project_root=PROJECT)
    for name,slides,entry in [('slide02_legacy_pattern',[2],legacy),('slide04_receiver_channels',[4,6,14],receiver)]:
        registry.figures[name]={'slides':slides,'description':entry['report'].get('interpretation',name),
            'files':{Path(path).suffix.lstrip('.'):path for path in entry['artifacts']},
            'metadata':entry['report'],'git_sha':context['provenance']['git_sha'],
            'source_inputs':context['source_paths'],'source_input_sha256':context['input_sha256']}
    registry.metrics.extend(legacy['metrics'])
    registry.metrics.extend(receiver['metrics'])
    return legacy['report'],receiver['report']

def export_results(context,registry,summary):
    """Save the same tabular results and manifest from CLI or notebook."""
    summary['provenance']=context['provenance'];summary['source_paths']=context['source_paths']
    summary['configuration']=context['configuration'];summary['prepared_report']=context['prepared']['report']
    summary['scan_metadata']=context['inputs']['scan_metadata']
    summary['echo_times_s']=context['inputs']['echo_times_s']
    summary['timestamp']=datetime.now(ZoneInfo('Europe/Paris')).isoformat()
    replay=HERE/'data/archived_optimization_replay.json'
    if replay.exists():summary['optimization_replay']=json.loads(replay.read_text())
    (HERE/'presentation_summary.json').write_text(json.dumps(safe(summary),indent=2))
    with (HERE/'results_metrics.csv').open('w',newline='') as stream:
        headings=sorted({key for row in registry.metrics for key in row})
        writer=csv.DictWriter(stream,fieldnames=headings);writer.writeheader();writer.writerows(registry.metrics)
    manifest={'figures':registry.figures,'provenance':context['provenance'],
        'source_paths':context['source_paths'],'input_sha256':context['input_sha256'],
        'timestamp':summary['timestamp'],'parameters':context['configuration'],
        'data_files':{str(p.relative_to(HERE)):sha256(p) for p in (HERE/'data').glob('*') if p.is_file()},
        'presentation_code_sha256':{str(p.name):sha256(p) for p in HERE.glob('*.py')},
        'numerical_results':'results_metrics.csv','summary':'presentation_summary.json'}
    (HERE/'figures_manifest.json').write_text(json.dumps(safe(manifest),indent=2))
    print('Scientific figure generation complete:',HERE/'figures_manifest.json',flush=True)
    return summary,manifest

def generate_all(context,*,run_expensive=False):
    torch.set_num_threads(int(context.get('threads',2)))
    if run_expensive:
        from controlled_experiments import run_controlled_experiments
        run_controlled_experiments(context,HERE/'data',run_expensive=True,warmup=1,repeats=3,setup_repeats=3)
    required=('equivalence.npz','uniform_phantom.npz','runtime_benchmarks.json')
    missing=[name for name in required if not (HERE/'data'/name).exists()]
    if missing:raise FileNotFoundError(f'Missing controlled calculations {missing}. Set RUN_EXPENSIVE=True to generate them.')
    validate_controlled_results(context)
    registry=Figures(context)
    summary,_,_=main_figures(context,registry)
    summary['geometry']=geometry_figures(context,registry)
    summary['equivalence'],summary['uniform_phantom'],summary['ADC_window']=controlled_figures(context,registry)
    sampling_controls_figure(context,registry)
    summary['runtime']=benchmark_figures(registry)
    summary['coherence_integral']=coherence_figure(context,registry)
    summary['input_maps']=input_map_figures(context,registry)
    summary['local_q']=local_q_figures(context,registry)
    summary['optimization_replay']=optimization_replay_figure(context,registry)
    summary['legacy'],summary['receiver_audit']=add_receiver_and_legacy(context,registry)
    return export_results(context,registry,summary)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-expensive',action='store_true');parser.add_argument('--refresh-inputs',action='store_true')
    args=parser.parse_args();ctx=prepare_context(refresh=args.refresh_inputs)
    generate_all(ctx,run_expensive=args.run_expensive)
