"""Exercise the seven notebook export configurations without running MRzero."""
import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from export_acdc_waveform import export_waveform, export_zero_time_copy


ROOT=Path(__file__).parent
NOTEBOOKS=sorted(ROOT.glob('0[56]*.ipynb'))


def cells(path):
    return json.loads(path.read_text())['cells']


def code_containing(path,text):
    matches=[''.join(cell['source']) for cell in cells(path)
             if cell['cell_type']=='code' and text in ''.join(cell['source'])]
    if len(matches)!=1:
        raise AssertionError(f'{path.name}: expected one cell containing {text!r}, got {len(matches)}')
    return matches[0]


class NotebookWaveformTests(unittest.TestCase):
    def test_adc_parameters_and_cell_precede_simulation_model_in_all_notebooks(self):
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name):
                parameter_source=code_containing(path,'WAVEFORM_ADC_INTERVAL_S=')
                namespace={}
                expected_names={'EXPORT_ADC_WAVEFORM','WAVEFORM_ADC_INTERVAL_S',
                                'WAVEFORM_ADC_DURATION_S'}
                for statement in ast.parse(parameter_source).body:
                    if isinstance(statement,ast.Assign) and any(
                        isinstance(target,ast.Name) and target.id in expected_names
                        for target in statement.targets):
                        exec(compile(ast.Module(body=[statement],type_ignores=[]),
                                     '<adc parameters>','exec'),namespace)
                self.assertTrue(namespace['EXPORT_ADC_WAVEFORM'])
                self.assertEqual(namespace['WAVEFORM_ADC_INTERVAL_S'],.000050)
                self.assertIsNone(namespace['WAVEFORM_ADC_DURATION_S'])
                notebook_cells=cells(path)
                sources=[''.join(cell['source']) for cell in notebook_cells]
                adc_index=next(index for index,source in enumerate(sources)
                    if 'adc_waveform_export=export_adc_waveform(' in source)
                input_index=next(index for index,source in enumerate(sources)
                    if notebook_cells[index]['cell_type']=='code' and 'inputs=read_twin_inputs(' in source)
                model_index=next(index for index,source in enumerate(sources)
                    if notebook_cells[index]['cell_type']=='code' and 'prepared=prepare_echo_shim_inputs(' in source)
                self.assertGreater(adc_index,input_index)
                self.assertLess(adc_index,model_index)

    def test_adc_export_is_independent_of_image_optimization_and_model(self):
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name):
                folder=Path('/acquisition/waveform')
                returned={'csv_file':str(folder/'waveform_adc.csv'),
                          'samples':100,'trigger_count':64,'interval_s':50e-6}
                namespace=dict(CLOSED_LOOP_RESUME=False,EXPORT_ADC_WAVEFORM=True,
                    RUN_OPTIMIZATION=False,EXPORT_WAVEFORM=False,
                    WAVEFORM_OUTPUT_DIR=folder,WAVEFORM_ADC_INTERVAL_S=50e-6,
                    WAVEFORM_ADC_DURATION_S=None,TRIGGER_CHANNEL='ext1',
                    inputs={'timing':{'verified':True},
                            'paths':{'scan_sequence':Path('/acquisition/sequences/scan.seq')}},
                    table=lambda *args:None)
                source=code_containing(path,'adc_waveform_export=export_adc_waveform(')
                with patch('adc_waveform_export.export_adc_waveform',return_value=returned) as export:
                    with redirect_stdout(io.StringIO()):
                        exec(source,namespace)
                export.assert_called_once_with(namespace['inputs']['paths']['scan_sequence'],
                    namespace['inputs']['timing'],folder,interval_s=50e-6,
                    trigger_channel='ext1',duration_s=None)
                self.assertIs(namespace['adc_waveform_export'],returned)
                self.assertNotIn('model',namespace)
                self.assertNotIn('solution',namespace)
                self.assertNotIn('torch',namespace)

    def test_adc_export_disabled_and_closed_loop_resume_create_no_file(self):
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name):
                source=code_containing(path,'adc_waveform_export=export_adc_waveform(')
                namespace={'CLOSED_LOOP_RESUME':False,'EXPORT_ADC_WAVEFORM':False}
                with patch('adc_waveform_export.export_adc_waveform') as export:
                    with redirect_stdout(io.StringIO()):
                        exec(source,namespace)
                export.assert_not_called()
                self.assertIsNone(namespace['adc_waveform_export'])
                if 'ClosedLoop' in path.name:
                    namespace={'CLOSED_LOOP_RESUME':True}
                    with patch('adc_waveform_export.export_adc_waveform') as export:
                        with redirect_stdout(io.StringIO()):
                            exec(source,namespace)
                    export.assert_not_called()
                    self.assertNotIn('adc_waveform_export',namespace)

    def test_adc_csv_actual_64_trigger_sequence_has_50us_steps_and_resets(self):
        sequence=ROOT.parent/'acquisition_024_(06)/sequences/gre_epi_64x64_fov200mm_nav3_cimax_snr_t2star40ms_line_triggers_v6.seq'
        if not sequence.is_file():
            self.skipTest('Actual 64-line-trigger sequence is unavailable')
        from adc_waveform_export import export_adc_waveform
        from read_acdc_sequence import read_sequence
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _,timing=read_sequence(sequence)
        original_sequence=sequence.read_bytes()
        with tempfile.TemporaryDirectory(prefix='notebook_adc_') as tmp:
            folder=Path(tmp)/'waveform'
            result=export_adc_waveform(sequence,timing,folder,interval_s=50e-6,
                                      trigger_channel='ext1',duration_s=None)
            exported=Path(result['csv_file'])
            self.assertEqual(exported.parent,folder)
            self.assertEqual(exported.name,'waveform_adc.csv')
            raw_lines=exported.read_text().splitlines()
            self.assertTrue(raw_lines)
            self.assertTrue(all(',' not in line and not line.startswith('#')
                                for line in raw_lines))
            times=np.loadtxt(exported,delimiter=',')
            self.assertEqual(times.ndim,1)
            self.assertEqual(times[0],0.)
            self.assertEqual(np.count_nonzero(times==0.),64)
            differences=np.diff(times)
            resets=np.flatnonzero(differences<0.)+1
            self.assertEqual(resets.size,63)
            np.testing.assert_array_equal(times[resets],0.)
            np.testing.assert_allclose(differences[differences>=0.],50e-6,
                                       rtol=0.,atol=1e-12)
            self.assertEqual(sequence.read_bytes(),original_sequence)

    def test_all_notebooks_compile_and_load_only_shared_calibration(self):
        self.assertEqual(len(NOTEBOOKS),7)
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name):
                for i,cell in enumerate(cells(path)):
                    if cell['cell_type']=='code':
                        source=''.join(cell['source'])
                        compile(source,f'{path.name}:{i}','exec')
                        self.assertNotIn('INPUT_DIR/"ACDC_3T.npz"',source)
                        if 'prepare_echo_shim_inputs(inputs,fit,' in source:
                            self.assertIn('prepare_echo_shim_inputs(inputs,fit,CALIBRATION_PATH,',source)
                source=code_containing(path,'CALIBRATION_PATH=')
                namespace={'PROJECT_ROOT':Path('/project'),'INPUT_DIR':Path('/acquisition')}
                for statement in ast.parse(source).body:
                    if isinstance(statement,ast.Assign) and any(isinstance(target,ast.Name)
                        and target.id in ('CALIBRATION_PATH','WAVEFORM_OUTPUT_DIR') for target in statement.targets):
                        exec(compile(ast.Module(body=[statement],type_ignores=[]),'<paths>','exec'),namespace)
                self.assertEqual(namespace['CALIBRATION_PATH'],Path('/project/data/ACDC_3T.npz'))
                self.assertEqual(namespace['WAVEFORM_OUTPUT_DIR'],Path('/acquisition/waveform'))

    def test_field_exports_work_without_any_image_optimization(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            for path in NOTEBOOKS:
                with self.subTest(notebook=path.name):
                    calls=[]
                    returned={'exports':{'static':{'csv_file':str(folder/'waveform_static.csv')},
                                         'dynamic':{'csv_file':str(folder/'waveform_dynamic.csv')}},
                              'report':{'checked':True}}
                    def export(model,output,settings,**options):
                        calls.append((model,output,settings,options))
                        return returned
                    namespace=dict(CLOSED_LOOP_RESUME=False,EXPORT_FIELD_SHIM_WAVEFORMS=True,
                        RUN_OPTIMIZATION=False,IMAGE_INITIALIZATION='zero',solver_ok=False,
                        solution=None,model=object(),WAVEFORM_OUTPUT_DIR=folder,
                        WAVEFORM_PARAMETERS={'channel_map':[1]},TOTAL_CURRENT_LIMIT_A=30.,
                        FIELD_LIMIT_HZ=500.,inputs={'timing':{'t':1},'paths':{'scan_sequence':'scan.seq'}},
                        CSV_TIME_ORIGIN='first_imaging_trigger',TRIGGER_CHANNEL='ext1',
                        FIELD_SHIM_RASTER_S=10e-6,FIELD_SHIM_MAX_ITERATIONS=80,
                        source_hashes={'calibration':'confirmed_hash'},summary={},
                        table=lambda *args:None,export_field_shim_waveforms=export)
                    exec(code_containing(path,'field_shim_exports=export_field_shim_waveforms('),namespace)
                    self.assertEqual(len(calls),1)
                    self.assertIs(calls[0][0],namespace['model'])
                    self.assertEqual(calls[0][1],folder)
                    self.assertEqual(calls[0][3]['raster_s'],10e-6)
                    self.assertEqual(calls[0][3]['max_iterations'],80)
                    self.assertEqual(calls[0][3]['provenance'],namespace['source_hashes'])
                    self.assertIs(namespace['summary']['field_shim_waveforms'],returned)
                    calls.clear()
                    namespace['EXPORT_FIELD_SHIM_WAVEFORMS']=False
                    with redirect_stdout(io.StringIO()):
                        exec(code_containing(path,'field_shim_exports=export_field_shim_waveforms('),namespace)
                    self.assertEqual(calls,[])

    def test_image_csv_uses_acquisition_waveform_folder(self):
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name):
                source=code_containing(path,'export=export_waveform(currents,')
                self.assertIn('WAVEFORM_PARAMETERS,WAVEFORM_OUTPUT_DIR,',source)
                self.assertNotIn('WAVEFORM_PARAMETERS,OUTPUT_DIR,',source)
                self.assertIn('start_at_first_nonzero=True',source)

    def image_export_namespace(self,folder,*,enabled=True,solver_ok=True,solution=True):
        folder.mkdir()
        output=folder/'results'
        output.mkdir()
        edges=np.array([.001,.002,.003,.004])
        calls=[]
        def export(*args,**options):
            calls.append(('timed',options))
            return export_waveform(*args,**options)
        def zero_time(source):
            calls.append(('zero_time',Path(source)))
            return export_zero_time_copy(source)
        namespace=dict(CLOSED_LOOP_RESUME=False,Path=Path,np=np,json=json,
            inputs={'paths':{'scan_sequence':'my_sequence.seq'},'timing':{'verified':True}},
            TRIGGER_CHANNEL='ext1',CSV_TIME_ORIGIN='first_imaging_trigger',model=object(),
            sequence_trigger_timing=lambda *args,**options:{'trigger_rf_s':.0005},
            waveform_edges_for_origin=lambda *args,**options:(edges,{'origin':'first_imaging_trigger'}),
            WAVEFORM_PARAMETERS=dict(channel_map=[2,1],channel_polarity=[-1,1],
                max_current_A=3.,total_current_limit_A=30.,min_dac_interval_s=8e-6),
            WAVEFORM_OUTPUT_DIR=folder/'waveform',OUTPUT_DIR=output,
            currents=np.array([[0.,.125,.25],[0.,-.5,.75]]),EXPORT_WAVEFORM=enabled,
            solution=object() if solution else None,solver_ok=solver_ok,
            summary={},json_safe=lambda value:value,
            export_waveform=export,export_zero_time_copy=zero_time)
        return namespace,calls

    def test_image_export_creates_zero_time_companion_in_all_notebooks(self):
        with tempfile.TemporaryDirectory() as tmp:
            for index,path in enumerate(NOTEBOOKS):
                with self.subTest(notebook=path.name):
                    namespace,calls=self.image_export_namespace(Path(tmp)/str(index))
                    with redirect_stdout(io.StringIO()):
                        exec(code_containing(path,'export=export_waveform(currents,'),namespace)
                    timed=namespace['csv_file']
                    zero_time=namespace['csv_zero_time_file']
                    self.assertEqual([item[0] for item in calls],['timed','zero_time'])
                    self.assertFalse(calls[0][1]['zero_time_column'])
                    self.assertTrue(calls[0][1]['start_at_first_nonzero'])
                    self.assertEqual(calls[1][1],timed)
                    self.assertEqual(zero_time,timed.with_name(timed.stem+'_zero_time.csv'))
                    self.assertEqual(zero_time.parent,namespace['WAVEFORM_OUTPUT_DIR'])
                    original_lines=timed.read_bytes().splitlines(keepends=True)
                    copied_lines=zero_time.read_bytes().splitlines(keepends=True)
                    self.assertEqual(len(original_lines),len(copied_lines))
                    for original,copied in zip(original_lines,copied_lines):
                        if original.lstrip().startswith(b'#'):
                            self.assertEqual(copied,original)
                        else:
                            self.assertEqual(copied,b'0,'+original.partition(b',')[2])
                    timed_rows=np.loadtxt(timed,delimiter=',')
                    np.testing.assert_allclose(timed_rows[:,0],np.array([0.,.001,.002]),
                                               rtol=0,atol=1e-15)
                    # The all-zero initial state is removed; mapping, polarity,
                    # useful command rows and the final zero remain unchanged.
                    np.testing.assert_array_equal(timed_rows[:,1:3],
                        np.array([[-.5,-.125],[.75,-.25],[0.,0.]]))
                    np.testing.assert_array_equal(timed_rows[:,3:],0.)
                    np.testing.assert_array_equal(namespace['csv_edges'],
                                                  np.array([.001,.002,.003,.004]))
                    timed_report=namespace['summary']['waveform_export']
                    np.testing.assert_array_equal(timed_report['state_edges_s'],namespace['csv_edges'])
                    self.assertEqual(timed_report['leading_zero_states'],1)
                    self.assertEqual(timed_report['time_origin_shift_s'],.002)
                    report=namespace['summary']['waveform_zero_time_export']
                    self.assertTrue(report['time_column_all_zero'])
                    self.assertEqual(report['source_csv_file'],str(timed))
                    self.assertEqual(report['csv_file'],str(zero_time))
                    saved=json.loads(next(namespace['OUTPUT_DIR'].glob('*_optimization.json')).read_text())
                    self.assertEqual(saved['waveform_zero_time_export'],report)

    def test_disabled_or_unverified_image_export_creates_no_companion(self):
        with tempfile.TemporaryDirectory() as tmp:
            for index,path in enumerate(NOTEBOOKS):
                for reason,options in [('disabled',{'enabled':False}),
                                       ('unverified',{'solver_ok':False}),
                                       ('no_solution',{'solution':False})]:
                    with self.subTest(notebook=path.name,reason=reason):
                        namespace,calls=self.image_export_namespace(Path(tmp)/f'{index}_{reason}',**options)
                        with redirect_stdout(io.StringIO()):
                            exec(code_containing(path,'export=export_waveform(currents,'),namespace)
                        self.assertEqual(calls,[])
                        self.assertIsNone(namespace['csv_file'])
                        self.assertIsNone(namespace['csv_zero_time_file'])
                        self.assertNotIn('waveform_zero_time_export',namespace['summary'])
                        self.assertFalse(namespace['WAVEFORM_OUTPUT_DIR'].exists())

    def test_closed_loop_resume_skips_image_and_zero_time_exports(self):
        for path in NOTEBOOKS:
            if 'ClosedLoop' not in path.name:
                continue
            with self.subTest(notebook=path.name):
                namespace={'CLOSED_LOOP_RESUME':True}
                exec(code_containing(path,'export=export_waveform(currents,'),namespace)
                self.assertEqual(set(namespace)-{'__builtins__'},{'CLOSED_LOOP_RESUME'})

    def test_previews_restore_rf_origin_after_rebased_csv_export(self):
        previews=[path for path in NOTEBOOKS if any(
            'fig,waveform_report=plot_acdc_waveform(model,currents,' in ''.join(cell['source'])
            for cell in cells(path))]
        self.assertEqual(len(previews),6)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            for path in previews:
                for origin,has_csv in [('first_imaging_trigger',True),('rf',True),
                                       ('first_imaging_trigger',False)]:
                    with self.subTest(notebook=path.name,origin=origin,has_csv=has_csv):
                        calls=[]
                        def plot(model,currents,**options):
                            calls.append(options)
                            return SimpleNamespace(savefig=lambda *args,**kwargs:None),{'checked':True}
                        namespace=dict(CLOSED_LOOP_RESUME=False,Path=Path,json=json,
                            csv_file=folder/'waveform1.csv' if has_csv else None,
                            WAVEFORM_OUTPUT_DIR=folder,OUTPUT_DIR=folder,
                            solution=object(),CSV_TIME_ORIGIN=origin,
                            trigger_report={'trigger_rf_s':.0005},model=object(),currents=object(),
                            WAVEFORM_PARAMETERS={'channel_map':[1],'channel_polarity':[1]},
                            WAVEFORM_PREVIEW_CHANNEL=None,
                            summary={'waveform_export':{'time_origin_shift_s':.002,'leading_zero_states':1}},
                            plot_acdc_waveform=plot,show=lambda fig:None,json_safe=lambda value:value)
                        with redirect_stdout(io.StringIO()):
                            exec(code_containing(path,'fig,waveform_report=plot_acdc_waveform(model,currents,'),namespace)
                        self.assertEqual(len(calls),1)
                        expected=(.0005 if origin=='first_imaging_trigger' else 0.)+(.002 if has_csv else 0.)
                        self.assertAlmostEqual(calls[0]['csv_origin_rf_s'],expected)
                        self.assertEqual(calls[0]['csv_leading_zero_states'],1 if has_csv else 0)

    def test_closed_loop_resume_skips_field_solver_and_exports(self):
        for path in NOTEBOOKS:
            if 'ClosedLoop' not in path.name:
                continue
            with self.subTest(notebook=path.name):
                namespace={'CLOSED_LOOP_RESUME':True}
                exec(code_containing(path,'field_shim_exports=export_field_shim_waveforms('),namespace)
                self.assertEqual(set(namespace)-{'__builtins__'},{'CLOSED_LOOP_RESUME'})


if __name__=='__main__':
    unittest.main()
