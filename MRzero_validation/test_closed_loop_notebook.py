"""Exercise notebook configuration and resume guards without running a scanner."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


NOTEBOOK=Path(__file__).with_name('06_ACDC_ClosedLoop_RealFeedback.ipynb')


def notebook_cells():
    return json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']


def code_containing(text):
    matches=[''.join(cell['source']) for cell in notebook_cells()
             if cell['cell_type']=='code' and text in ''.join(cell['source'])]
    if len(matches)!=1:
        raise AssertionError(f'Expected exactly one code cell containing {text!r}, got {len(matches)}')
    return matches[0]


class ClosedLoopNotebookTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root=Path(self.temporary.name)
        self.ns=dict(Path=Path,deepcopy=deepcopy,np=np,json=json,hashlib=hashlib,
            PROJECT_ROOT=root,INPUT_DIR=root/'acquisition',
            CALIBRATION_PATH=root/'data/ACDC_3T.npz',
            WAVEFORM_OUTPUT_DIR=root/'acquisition/waveform',
            GRE_EPI_SEQUENCE_NAME='custom_epi.seq',GRE_EPI_SCAN_PATH=None,
            B0_SEQUENCE_NAME='custom_multiecho.seq',B0_SCAN_PATH=None,
            REFERENCE_SEQUENCE_NAME=None,REFERENCE_SCAN_PATH=None)
        exec(code_containing('ALIGNMENT_ANGLE_DEG=180.0'),self.ns)
        exec(code_containing('RUN_CLOSED_LOOP=False\nRUN_SIMULATION_OPTIMIZATION='),self.ns)
        with redirect_stdout(io.StringIO()):
            exec(code_containing('def current_closed_loop_configuration():'),self.ns)

    def test_arbitrary_acquisition_names_and_complex_physics_are_archived(self):
        n=self.ns
        n.update(REFERENCE_SEQUENCE_NAME='gre_64x64_my_protocol.seq',
            REFERENCE_SCAN_PATH='my_new_reference_64.dat',
            B0_SCAN_PATH=Path(n['PROJECT_ROOT'])/'different_name_b0.dat',
            GRE_EPI_SCAN_PATH='different_baseline_name.dat',ECHO_SIGNAL_MODEL='complex',
            OBJECT_AMPLITUDE_SOURCE='reference',COMPLEX_MAX_WIDTH_HZ=90.)
        config=n['current_closed_loop_configuration']()['model_configuration']
        self.assertEqual(config['REFERENCE_SEQUENCE_NAME'],'gre_64x64_my_protocol.seq')
        self.assertEqual(config['REFERENCE_SCAN_PATH'],str((n['INPUT_DIR']/'my_new_reference_64.dat').resolve()))
        self.assertEqual(config['GRE_EPI_SCAN_PATH'],str((n['INPUT_DIR']/'different_baseline_name.dat').resolve()))
        self.assertEqual(config['B0_SCAN_PATH'],str((n['PROJECT_ROOT']/'different_name_b0.dat').resolve()))
        self.assertEqual(config['ECHO_SIGNAL_MODEL'],'complex')
        self.assertEqual(config['OBJECT_AMPLITUDE_SOURCE'],'reference')
        self.assertEqual(config['COMPLEX_MAX_WIDTH_HZ'],90.)

    def archive_configuration(self,physical):
        n=self.ns
        folder=n['CLOSED_LOOP_SESSION_DIR']
        folder.mkdir(parents=True)
        content=json.dumps({'configuration':{'model_configuration':physical}}).encode()
        (folder/'state_0000.json').write_bytes(content)
        pointer=json.dumps({'state_file':'state_0000.json',
            'state_sha256':hashlib.sha256(content).hexdigest()}).encode()
        (folder/'session.json').write_bytes(pointer)
        n['RUN_CLOSED_LOOP']=True
        return content,pointer

    def test_new_session_records_joint_object_and_measured_gre_objective(self):
        n=self.ns
        physical=n['current_closed_loop_model_configuration']()
        self.assertEqual(physical['OBJECT_AMPLITUDE_SOURCE'],'joint')
        self.assertEqual(physical['OBJECT_CONTRAST_SIGMA_MM'],10.)
        self.assertEqual(physical['image_optimization_target'],'measured_GRE')
        self.assertEqual(physical['image_optimization_loss'],
                         'fixed_gain_masked_relative_magnitude_MSE')
        self.assertEqual(physical['OPTIMIZATION_REFERENCE_THRESHOLD'],0.)
        self.assertEqual(physical['CALIBRATION_PATH'],str(n['CALIBRATION_PATH'].resolve()))
        self.assertEqual(n['current_closed_loop_configuration']()['waveform_output_dir'],
                         str(n['WAVEFORM_OUTPUT_DIR'].resolve()))

    def test_resume_rejects_changed_joint_contrast_and_gre_objective_roi(self):
        n=self.ns
        self.archive_configuration(n['current_closed_loop_model_configuration']())
        n['OBJECT_CONTRAST_SIGMA_MM']=15.
        with self.assertRaisesRegex(ValueError,'OBJECT_CONTRAST_SIGMA_MM'):
            n['current_closed_loop_configuration']()
        n['OBJECT_CONTRAST_SIGMA_MM']=10.
        n['OPTIMIZATION_REFERENCE_THRESHOLD']=.2
        with self.assertRaisesRegex(ValueError,'OPTIMIZATION_REFERENCE_THRESHOLD'):
            n['current_closed_loop_configuration']()

    def test_target_is_configured_without_optimization_using_the_active_mode_roi(self):
        image=np.ones((4,4))
        roi=np.eye(4,dtype=bool)
        calls=[]
        def configure(model,reference,**options):
            calls.append(options)
            return dict(reference_image=image,mask=np.ones_like(roi),
                        report={'fixed_gain':True})
        namespace=dict(np=np,CLOSED_LOOP_RESUME=False,RUN_OPTIMIZATION=False,
            model=object(),inputs={'gre':object()},baseline_image=image,
            measured_image=image,initial_image=image,O=np.eye(2),X_image=image,Y_image=image,
            OPTIMIZATION_REFERENCE_THRESHOLD=0.,
            CLOSED_LOOP_REFERENCE_THRESHOLD_FRACTION=.2,CLOSED_LOOP_ROI_MASK=roi,
            configure_measured_gre_target=configure,
            relative_image_mse=lambda *args:0.,table=lambda *args:None,
            plt=SimpleNamespace(subplots=lambda *args,**kwargs:(object(),[object()]*3)),
            image_physical=lambda *args,**kwargs:None,show=lambda *args:None)
        cell=code_containing('optimization_target=configure_measured_gre_target(')
        for closed_mode in (False,True):
            namespace['RUN_CLOSED_LOOP']=closed_mode
            with redirect_stdout(io.StringIO()):
                exec(cell,namespace)
            self.assertEqual(calls[-1]['threshold_fraction'],.2 if closed_mode else 0.)
            self.assertIs(calls[-1]['roi_mask'],roi if closed_mode else None)
            self.assertEqual(namespace['optimization_target_report'],{'fixed_gain':True})
            self.assertIsNone(namespace['solution'])

    def test_older_session_keeps_its_exact_model_snapshot_and_files(self):
        n=self.ns
        current=n['current_closed_loop_model_configuration']()
        # An earlier session has no knowledge of the new temporal/reference model.
        previous={key:current[key] for key in ('input_dir','B0_SEQUENCE_NAME',
            'B0_SCAN_PATH','ALIGNMENT_ANGLE_DEG','CALIBRATION_PARAMETERS')}
        content,pointer=self.archive_configuration(previous)
        n['ECHO_SIGNAL_MODEL']='complex'
        n['OBJECT_AMPLITUDE_SOURCE']='reference'
        frozen=n['current_closed_loop_configuration']()['model_configuration']
        self.assertEqual(frozen,previous)
        folder=n['CLOSED_LOOP_SESSION_DIR']
        self.assertEqual((folder/'state_0000.json').read_bytes(),content)
        self.assertEqual((folder/'session.json').read_bytes(),pointer)
        frozen['ALIGNMENT_ANGLE_DEG']=999
        self.assertEqual(n['current_closed_loop_configuration']()['model_configuration'],previous)
        self.assertIsNone(n['current_closed_loop_configuration']()['waveform_output_dir'])

    def test_resume_keeps_its_archived_waveform_directory(self):
        n=self.ns
        self.archive_configuration(n['current_closed_loop_model_configuration']())
        folder=n['CLOSED_LOOP_SESSION_DIR']
        state=folder/'state_0000.json'
        archived=json.loads(state.read_text())
        archived['configuration']['waveform_output_dir']=str(folder/'archived_waveform')
        content=json.dumps(archived).encode()
        state.write_bytes(content)
        (folder/'session.json').write_text(json.dumps({'state_file':state.name,
            'state_sha256':hashlib.sha256(content).hexdigest()}))
        n['WAVEFORM_OUTPUT_DIR']=folder/'a_different_directory'
        self.assertEqual(n['current_closed_loop_configuration']()['waveform_output_dir'],
                         str(folder/'archived_waveform'))

    def test_resume_rejects_changed_calibration_path(self):
        n=self.ns
        self.archive_configuration(n['current_closed_loop_model_configuration']())
        n['CALIBRATION_PATH']=n['PROJECT_ROOT']/'other_calibration.npz'
        with self.assertRaisesRegex(ValueError,'CALIBRATION_PATH'):
            n['current_closed_loop_configuration']()

    def test_resume_rejects_changed_archived_reference_and_physics(self):
        n=self.ns
        self.archive_configuration(n['current_closed_loop_model_configuration']())
        n['REFERENCE_SCAN_PATH']='another_reference.dat'
        with self.assertRaisesRegex(ValueError,'REFERENCE_SCAN_PATH'):
            n['current_closed_loop_configuration']()
        n['REFERENCE_SCAN_PATH']=None
        n['COMPLEX_MAX_WIDTH_HZ']+=1.
        with self.assertRaisesRegex(ValueError,'COMPLEX_MAX_WIDTH_HZ'):
            n['current_closed_loop_configuration']()

    def test_gpu_runtime_choices_can_change_when_resuming_frozen_physics(self):
        n=self.ns
        physical=n['current_closed_loop_model_configuration']()
        self.archive_configuration(physical)
        n.update(GPU_PROFILE='another_gpu',GPU_MEMORY_FRACTION=.25,
                 COMPLEX_FIT_BATCH_SIZE=32,RECONSTRUCTION_CHUNK=64)
        self.assertEqual(n['current_closed_loop_configuration']()['model_configuration'],physical)

    def test_resume_detects_modified_metadata(self):
        n=self.ns
        self.archive_configuration(n['current_closed_loop_model_configuration']())
        state=n['CLOSED_LOOP_SESSION_DIR']/'state_0000.json'
        state.write_text('{}')
        with self.assertRaisesRegex(ValueError,'Empreinte'):
            n['current_closed_loop_configuration']()

    def test_resume_skips_all_data_fit_gpu_and_export_cells(self):
        cells=[cell for cell in notebook_cells() if cell['cell_type']=='code'
               and ''.join(cell['source']).startswith('# Reprise fermée :')]
        self.assertGreaterEqual(len(cells),15)
        namespace={'CLOSED_LOOP_RESUME':True}
        for index,cell in enumerate(cells):
            exec(compile(''.join(cell['source']),f'closed_loop_cell_{index}','exec'),namespace)
        self.assertEqual(set(namespace)-{'__builtins__'},{'CLOSED_LOOP_RESUME'})


if __name__=='__main__':
    unittest.main()
