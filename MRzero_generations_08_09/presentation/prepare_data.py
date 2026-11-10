"""Reproduce the selected 08 input/echo fit without optimization or exports.

Preparation executes only the import/configuration, metadata, 32-echo fit and
voxel preparation cells of the CURRENT notebook. Each executed cell is saved
verbatim alongside its hash; the presentation does not substitute another fit.
Only execution device/precision, output directory and non-scientific display
are overridden. All experiment/fit/geometry settings are retained.
"""
from __future__ import annotations
import hashlib,json,os,pickle,subprocess,time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import torch

PACKAGE=Path(__file__).resolve().parent.parent
PROJECT=PACKAGE.parent
PRESENTATION=PACKAGE/'presentation'

def safe(value):
    if isinstance(value,dict):return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe(v) for v in value]
    if isinstance(value,Path):return str(value.resolve())
    if isinstance(value,np.ndarray):return safe(value.tolist())
    if isinstance(value,np.generic):return safe(value.item())
    if isinstance(value,float) and not np.isfinite(value):return str(value)
    if isinstance(value,(str,int,float,bool)) or value is None:return value
    return str(value)

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()

def source_provenance():
    files=sorted(PACKAGE.glob('*.py'))+sorted(PACKAGE.glob('*.ipynb'))
    extra=PROJECT/'MRzero_validation/experiment_multiecho.py'
    if extra.exists():files.append(extra)
    return {'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
        'git_source_status':subprocess.check_output(['git','status','--short','--',str(PACKAGE.relative_to(PROJECT)),str(extra.relative_to(PROJECT))],cwd=PROJECT,text=True).strip(),
        'code_sha256':{str(p.resolve()):sha256(p) for p in files},
        'git_note':'Generation source files are untracked/dirty; code SHA256 hashes identify the actual implementation.'}

def prepare_context(*,refresh=False,fit_iterations=None,threads=2):
    cache=PRESENTATION/'cache';cache.mkdir(exist_ok=True,parents=True)
    destination=cache/'prepared_context.pkl'
    provenance=source_provenance()
    if destination.exists() and not refresh:
        with destination.open('rb') as f:context=pickle.load(f)
        if context['provenance']['code_sha256']!=provenance['code_sha256']:
            raise RuntimeError('Code differs from prepared cache. Rerun with refresh=True; no stale cache was used.')
        for name,path in context['source_paths'].items():
            if not Path(path).is_file() or sha256(path)!=context['input_sha256'][name]:
                raise RuntimeError(f'Input {name} differs from the prepared cache. Rerun with refresh=True.')
        return context
    torch.set_num_threads(threads);np.random.seed(0);torch.manual_seed(0)
    notebook=PACKAGE/'08_acdc_echo32_b0_s0_t2star_MRzero.ipynb'
    nb=json.loads(notebook.read_text())
    scope={'__name__':'__presentation__'};timings={};snapshots={}
    previous=Path.cwd();os.chdir(PROJECT)
    started=time.perf_counter()
    try:
        for index in (1,3,4,6,7,14):
            text=''.join(nb['cells'][index]['source'])
            snapshots[str(index)]=text
            print(f'Preparing current 08 cell {index}',flush=True)
            tick=time.perf_counter();exec(compile(text,f'{notebook}:cell{index}','exec'),scope)
            timings[f'cell_{index}_s']=time.perf_counter()-tick
            if index==1:scope['display']=lambda *a,**k:None
            if index==3:
                scope.update(GPU_PROFILE='cpu',GPU_REQUIRED=False,GPU_PRECISION='float64',
                    OUTPUT_DIR=cache/'preparation_only',RUN_OPTIMIZATION=False,
                    EXPORT_WAVEFORM=False,EXPORT_FIELD_SHIM_WAVEFORMS=False,EXPORT_ADC_WAVEFORM=False)
                if fit_iterations is not None:scope['COMPLEX_FIT_ITERATIONS']=fit_iterations
            if index==4:
                scope['display']=lambda *a,**k:None
                scope['show']=lambda fig:scope['plt'].close(fig)
    finally:os.chdir(previous)
    selected_config={name:safe(value) for name,value in scope.items() if name.isupper() and not name.startswith('_')}
    inputs=scope['inputs'];prepared=scope['prepared']
    input_files={key:Path(value).resolve() for key,value in inputs['paths'].items()}
    input_files['calibration']=scope['CALIBRATION_PATH'].resolve()
    context={'prepared':prepared,'inputs':inputs,'fit':scope['fit'],
        'legacy_echo_fit':scope['legacy_echo_fit'],'frame_matrix':scope['O'],
        'max_current_A':scope['CHANNEL_MAX_CURRENT_A'],
        'total_current_limit_A':scope['TOTAL_CURRENT_LIMIT_A'],'field_limit_Hz':scope['FIELD_LIMIT_HZ'],
        'model_options':dict(profile='cpu',precision='float64',sample_chunk=128,
            memory_fraction=.4,require_cuda=False,hardware=scope['HARDWARE']),
        'configuration':selected_config,'provenance':provenance,
        'source_paths':{k:str(v) for k,v in input_files.items()},
        'input_sha256':{key:sha256(value) for key,value in input_files.items()},
        'timings':dict(timings,preparation_total_s=time.perf_counter()-started),
        'timestamp':datetime.now(ZoneInfo('Europe/Paris')).isoformat(),
        'fit_policy':'Current notebook settings; full 60-iteration fit unless explicit override is recorded.',
        'fit_iteration_override':fit_iterations,'threads':threads}
    with destination.open('wb') as f:pickle.dump(context,f,protocol=5)
    (cache/'prepared_context_manifest.json').write_text(json.dumps(safe({k:v for k,v in context.items() if k not in ('inputs','prepared','fit','legacy_echo_fit')}),indent=2))
    (cache/'executed_preparation_cells.json').write_text(json.dumps(snapshots,indent=2))
    print('Prepared context saved:',destination,flush=True)
    return context

if __name__=='__main__':
    prepare_context(refresh=True)
