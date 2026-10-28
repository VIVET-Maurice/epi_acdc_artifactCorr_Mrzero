"""Run notebook settings, initialization and CSV export with disabled outputs."""

import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'Experiment'))
from acdc_gpu_field import solve_field_gpu
from export_acdc_waveform import (
    export_waveform, export_zero_time_copy, validate_waveform_settings,
)
from experiment_optimization import optimize_mrzero_image
from test_experiment_realistic import small_model


NOTEBOOK = Path(__file__).with_name('06_acdc_echo32_b0_s0_t2star_MRzero.ipynb')


def source_containing(text):
    sources = [''.join(cell['source']) for cell in json.loads(NOTEBOOK.read_text())['cells']
               if cell['cell_type'] == 'code' and text in ''.join(cell['source'])]
    if len(sources) != 1:
        raise AssertionError(f'Expected one notebook cell containing {text!r}.')
    return sources[0]


def parameter_namespace(disabled, namespace=None):
    """Emulate editing only the user-facing list and the physical output map."""
    tree = ast.parse(source_containing('DISABLED_CHANNELS='))
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        names = [target.id for target in statement.targets if isinstance(target, ast.Name)]
        if 'DISABLED_CHANNELS' in names:
            statement.value = ast.parse(repr(disabled), mode='eval').body
        if 'WAVEFORM_PARAMETERS' in names and isinstance(statement.value, ast.Dict):
            for index, key in enumerate(statement.value.keys):
                if isinstance(key, ast.Constant) and key.value == 'channel_map':
                    statement.value.values[index] = ast.parse(
                        repr([2, 1] + list(range(3, 33))), mode='eval').body
    namespace = {} if namespace is None else namespace
    namespace.update(np=np, validate_waveform_settings=validate_waveform_settings)
    exec(compile(ast.fix_missing_locations(tree), '<notebook parameters>', 'exec'), namespace)
    return namespace


class DisabledChannelNotebookTests(unittest.TestCase):
    def test_removing_disabled_channels_reenables_their_limits_on_cell_rerun(self):
        namespace = parameter_namespace([2, 8])
        np.testing.assert_array_equal(namespace['CHANNEL_MAX_CURRENT_A'][[0, 7]], 0.)
        parameter_namespace([], namespace)
        np.testing.assert_array_equal(namespace['CHANNEL_MAX_CURRENT_A'], 3.)
        self.assertEqual(namespace['WAVEFORM_PARAMETERS']['disabled_channels'], [])

    def test_settings_reach_static_initialization_image_optimization_and_csv(self):
        for disabled in ([2, 8], list(range(1, 33))):
            with self.subTest(disabled=disabled), redirect_stdout(io.StringIO()):
                namespace = parameter_namespace(disabled)
                experiment, _, model = small_model(
                    max_current_A=namespace['CHANNEL_MAX_CURRENT_A'])
                mask = np.isin(namespace['WAVEFORM_PARAMETERS']['channel_map'], disabled)
                namespace.update(model=model, experiment=experiment,
                    zero_currents=np.zeros((model.nc, model.nstates)),
                    solve_field_gpu=solve_field_gpu, table=lambda *args: None)
                exec(source_containing('static_fit=solve_field_gpu('), namespace)
                # The static solve itself must exclude these channels, before
                # the notebook's subsequent hardware projection is applied.
                np.testing.assert_array_equal(namespace['static_fit']['currents_A'][mask], 0.)
                np.testing.assert_array_equal(namespace['initial_currents'][mask], 0.)
                if not mask.all():
                    self.assertGreater(np.abs(namespace['initial_currents'][~mask]).max(), 0.)
                snapshots = []
                solution = optimize_mrzero_image(model, namespace['initial_currents'],
                    max_iterations=2, callback=snapshots.append, use_cuda_graph=False)
                for snapshot in snapshots:
                    np.testing.assert_array_equal(snapshot['currents_A'][mask], 0.)
                with tempfile.TemporaryDirectory() as tmp:
                    namespace.update(Path=Path, json=json, OUTPUT_DIR=Path(tmp),
                        WAVEFORM_OUTPUT_DIR=Path(tmp)/'waveform',
                        inputs={'paths': {'scan_sequence': 'fixture.seq'}, 'timing': {}},
                        sequence_trigger_timing=lambda *args, **kwargs: {'trigger_rf_s': 0.},
                        waveform_edges_for_origin=lambda *args, **kwargs: (
                            np.r_[model.line_start_s, model.line_end_s[-1]], {}),
                        summary={}, solution=solution, solver_ok=True,
                        currents=solution['currents_A'], json_safe=lambda value: value,
                        export_waveform=export_waveform, export_zero_time_copy=export_zero_time_copy)
                    exec(source_containing('export=export_waveform(currents,'), namespace)
                    for key in ('csv_file', 'csv_zero_time_file'):
                        rows = np.loadtxt(namespace[key], delimiter=',', ndmin=2)
                        # Numeric column N is physical output N, after time.
                        np.testing.assert_array_equal(rows[:, disabled], 0.)
                        if not mask.all():
                            self.assertGreater(np.abs(rows[:, 1:]).max(), 0.)
                    settings = namespace['summary']['waveform_export']['settings']
                    self.assertEqual(settings['disabled_channels'], disabled)


if __name__ == '__main__':
    unittest.main()
