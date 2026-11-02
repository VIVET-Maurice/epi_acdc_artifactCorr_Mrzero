"""Spiral ADC ordering, navigator separation and actual-trigger regression tests."""

from copy import deepcopy
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Experiment"))

from read_spiral_sequence import read_spiral_sequence
from spiral_inputs import _spiral_twix_data


ROOT = Path(__file__).resolve().parent.parent
SPIRAL_SEQUENCE = ROOT / "acquisition_032_(07)/sequences/gre_spiral_128x128_fov200mm_nav3_cimax_turn_triggers_v1.seq"


class RawBlock:
    def __init__(self, values, *, image=True, navigator=False, reflected=False, segment=0):
        self.data = np.asarray(values, complex)
        self.image = image
        self.navigator = navigator
        self.reflected = reflected
        for counter in ("cAve", "cSlc", "cPar", "cEco", "cPhs", "cRep", "cLin", "cSeg"):
            setattr(self, counter, 0)
        self.cSet = segment
        self.mdh = SimpleNamespace(CutOff=SimpleNamespace(Pre=0, Post=0),
            SliceData=SimpleNamespace(SlicePos=SimpleNamespace(Sag=0., Cor=3., Tra=-12.),
                                     Quaternion=[2 ** -.5, 0., 0., 2 ** -.5]))

    def is_image_scan(self):
        return self.image

    def is_flag_set(self, name):
        return {"PHASCOR": self.navigator, "REFLECT": self.reflected}.get(name, False)


def raw_fixture():
    acquisition = dict(sequence_kind="spiral", kx_mInv=np.zeros((1, 8)),
        navigator_count=3, navigator_sample_count=[4, 4, 4],
        navigator_dwell_s=[6.5e-6] * 3, dwell_s=5e-6, max_adc_segment_length=3,
        image_shape=[8, 8], FOV_m=[.2, .2], TE_s=.0084425, TR_s=.15544)
    values = np.vstack((np.arange(8) + 1j * np.arange(8)[::-1], 2 * np.arange(8) + 10))
    segments = [RawBlock(values[:, :3], segment=0), RawBlock(values[:, 3:6], segment=1),
                RawBlock(values[:, 6:], segment=2)]
    nav = [RawBlock(np.ones((2, 4)) * (index + 20), image=False, navigator=True,
                    reflected=index == 1) for index in range(3)]
    # A SYNCDATA-like packet carries no ADC data and must not enter the sample stream.
    sync = RawBlock(np.empty((0, 0)), image=False)
    header = dict(alTE=[8443], alTR=[155440], sRXSPEC={"alDwellTime": [6500]},
        sKSpace={"lBaseResolution": 8, "lPhaseEncodingLines": 8}, adFlipAngleDegree=[15.],
        sSliceArray={"asSlice": [dict(dReadoutFOV=200., dPhaseFOV=200., dThickness=5.)]})
    return dict(mdb=[*nav, segments[0], sync, *segments[1:]], hdr={"MeasYaps": header}), acquisition, values


class SpiralRawTests(unittest.TestCase):
    def test_transport_segments_are_concatenated_without_cartesian_sorting(self):
        scan, acquisition, expected = raw_fixture()
        metadata, diagnostics = _spiral_twix_data(scan, acquisition)
        np.testing.assert_array_equal(diagnostics["kspace"][0], expected)
        self.assertEqual(metadata["imaging_adc_segment_samples"], [3, 3, 2])
        self.assertEqual(diagnostics["kspace"].shape, (1, 2, 8))
        self.assertEqual(diagnostics["navigator_kspace"].shape, (3, 2, 4))
        self.assertFalse(diagnostics["navigator_phase_correction_applied"])

    def test_imaging_dwell_is_not_overwritten_by_first_navigator_header(self):
        scan, acquisition, _ = raw_fixture()
        metadata, _ = _spiral_twix_data(scan, acquisition)
        self.assertEqual(metadata["dwell_s"], 5e-6)
        self.assertAlmostEqual(metadata["header_first_adc_dwell_s"], 6.5e-6)
        np.testing.assert_allclose(metadata["encoding_origin_m"], [.003, 0.], atol=1e-12)

    def test_samples_are_never_padded_or_cropped(self):
        scan, acquisition, _ = raw_fixture()
        scan["mdb"][-1].data = scan["mdb"][-1].data[:, :-1]
        with self.assertRaisesRegex(ValueError, "7 imaging samples; Pulseq requires 8"):
            _spiral_twix_data(scan, acquisition)

    def test_shuffled_or_repeated_segment_counters_are_rejected(self):
        for last_counter in (0, 1, 4):
            scan, acquisition, _ = raw_fixture()
            scan["mdb"][-1].cSet = last_counter
            with self.assertRaisesRegex(ValueError, "cSet"):
                _spiral_twix_data(scan, acquisition)

    def test_reflected_imaging_segment_is_not_silently_reversed(self):
        scan, acquisition, _ = raw_fixture()
        scan["mdb"][-1].reflected = True
        with self.assertRaisesRegex(ValueError, "REFLECT"):
            _spiral_twix_data(scan, acquisition)

    def test_different_slice_between_segments_is_rejected(self):
        scan, acquisition, _ = raw_fixture()
        scan["mdb"][-1].mdh.SliceData.SlicePos.Cor += 1.
        with self.assertRaisesRegex(ValueError, "geometry changes"):
            _spiral_twix_data(scan, acquisition)

    def test_second_repetition_cannot_be_confused_with_transport_segments(self):
        scan, acquisition, _ = raw_fixture()
        scan["mdb"][-1].cRep = 1
        with self.assertRaisesRegex(ValueError, "cRep"):
            _spiral_twix_data(scan, acquisition)

    def test_wrong_reference_matrix_or_fov_is_rejected(self):
        scan, acquisition, _ = raw_fixture()
        scan["hdr"]["MeasYaps"]["sKSpace"]["lBaseResolution"] = 16
        with self.assertRaisesRegex(ValueError, "matrix"):
            _spiral_twix_data(scan, acquisition)
        scan, acquisition, _ = raw_fixture()
        scan["hdr"]["MeasYaps"]["sSliceArray"]["asSlice"][0]["dReadoutFOV"] = 250.
        with self.assertRaisesRegex(ValueError, "FOV"):
            _spiral_twix_data(scan, acquisition)


@unittest.skipUnless(SPIRAL_SEQUENCE.is_file(), "Measured spiral Pulseq fixture is not installed.")
class ActualSpiralSequenceTests(unittest.TestCase):
    def test_samples_and_states_have_independent_dimensions(self):
        acquisition, timing = read_spiral_sequence(SPIRAL_SEQUENCE)
        self.assertEqual(acquisition["kx_mInv"].shape, (1, 29084))
        self.assertEqual(acquisition["image_shape"], [128, 128])
        self.assertEqual(acquisition["line_start_s"].shape, (65, 1))
        self.assertEqual(timing["output_count"], 65)
        self.assertTrue(timing["output_pulses_verified"])
        np.testing.assert_array_equal(np.unique(acquisition["line_index"]), np.arange(1, 66))
        self.assertTrue(np.all(np.diff(acquisition["time_s"][0]) > 0))

    def test_actual_output_times_not_equal_sample_groups_define_turns(self):
        acquisition, timing = read_spiral_sequence(SPIRAL_SEQUENCE)
        np.testing.assert_allclose(timing["state_start_s"][:3], [.00843, .01116, .01338], atol=1e-12)
        state = acquisition["line_index"][0] - 1
        starts, ends = acquisition["line_start_s"][:, 0], acquisition["line_end_s"][:, 0]
        times = acquisition["time_s"][0]
        self.assertTrue(np.all(times >= starts[state]))
        self.assertTrue(np.all(times < ends[state]))
        counts = np.bincount(state)
        self.assertEqual(counts[0], 544)
        self.assertNotEqual(counts[0], counts[1])
        self.assertEqual(timing["navigator_count"], 3)

    def test_inconsistent_turn_count_and_adc_definitions_fail(self):
        source = SPIRAL_SEQUENCE.read_text()
        for old, new in (("SpiralTurns 65", "SpiralTurns 64"),
                         ("SpiralADCSamples 29084", "SpiralADCSamples 29080"),
                         ("ImagingTurnOutputCount 65", "ImagingTurnOutputCount 64")):
            with self.subTest(definition=old), TemporaryDirectory() as folder:
                path = Path(folder) / "wrong.seq"
                path.write_text(source.replace(old, new))
                with self.assertRaises(ValueError):
                    read_spiral_sequence(path)


if __name__ == "__main__":
    unittest.main()
