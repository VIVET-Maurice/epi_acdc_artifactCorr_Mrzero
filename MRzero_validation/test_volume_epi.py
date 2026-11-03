"""Independent complex 3D image and partition-specific navigator fixtures."""

from copy import deepcopy
import ctypes
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from volume_epi import iter_epi3d_receivers, read_epi3d_sequence, _ordered_receiver


def fftc(values, axes, inverse=False):
    transform = np.fft.ifftn if inverse else np.fft.fftn
    return np.fft.fftshift(transform(np.fft.ifftshift(values, axes=axes), axes=axes), axes=axes)


class Block:
    def __init__(self, data, line, partition, flags, nx, ny, nz):
        self.data = data
        self.cLin, self.cPar = line, partition
        self.flags = set(flags)
        self.cAve = self.cSlc = self.cEco = self.cPhs = self.cRep = self.cSet = self.cSeg = 0
        self.mdh = SimpleNamespace(SamplesInScan=nx, UsedChannels=len(data),
                                   CenterCol=nx//2, CenterLin=ny//2, CenterPar=nz//2)

    def is_image_scan(self):
        return 'PHASCOR' not in self.flags

    def is_flag_set(self, flag):
        return flag in self.flags


def fixture():
    nx, ny, nz, receivers = 12, 8, 4, 2
    rng = np.random.default_rng(711)
    image = rng.normal(size=(receivers, nx, ny, nz)) + 1j*rng.normal(size=(receivers, nx, ny, nz))
    kspace = fftc(image, (1, 2, 3))
    blocks, corrections = [], np.empty((receivers, nz, nx))
    x = np.linspace(-1, 1, nx)
    for partition in range(nz):
        delta = np.array([.10+.04*partition+.06*(receiver+1)*x for receiver in range(receivers)])
        corrections[:, partition] = delta
        positive_base = (1.2+.2*np.cos(x))[None, :] * np.exp(1j*.13*x)[None, :]
        evolution = (.31+.04*x)[None, :]
        nav_hybrid = np.stack((np.repeat(positive_base*np.exp(-1j*evolution), receivers, axis=0),
                               np.repeat(positive_base, receivers, axis=0)*np.exp(1j*delta),
                               np.repeat(positive_base*np.exp(1j*evolution), receivers, axis=0)))
        nav_kspace = fftc(nav_hybrid, (2,))
        for nav_index in range(3):
            negative = nav_index == 1
            data = nav_kspace[nav_index, :, ::-1] if negative else nav_kspace[nav_index]
            # Navigator line counters are intentionally not useful for indexing.
            blocks.append(Block(data, 127 if partition else 0, partition,
                                ['PHASCOR']+(['REFLECT'] if negative else []), nx, ny, nz))
        shot = kspace[:, :, :, partition].transpose(0, 2, 1)
        hybrid = fftc(shot, (2,), inverse=True)
        hybrid[:, ::2] *= np.exp(1j*delta)[:, None, :]
        corrupted = fftc(hybrid, (2,))
        for line in [3, 0, 6, 1, 5, 2, 7, 4]:
            negative = line % 2 == 0
            data = corrupted[:, line, ::-1] if negative else corrupted[:, line]
            blocks.append(Block(data, line, partition, ['REFLECT'] if negative else [], nx, ny, nz))
    header = dict(sKSpace=dict(lBaseResolution=nx, lPhaseEncodingLines=ny, lPartitions=nz, ucDimension=4))
    return dict(mdb=blocks, hdr=dict(MeasYaps=header)), image, corrections


def make_sequence(path, *, different_last_shot=False, missing_trigger=False):
    """Write a small actual 3D EPI sequence with analytic Cartesian moments."""
    import pypulseq as pp
    nx, ny, nz, fov, dwell, tr = 8, 4, 4, .2, 20e-6, .01
    sequence = pp.Sequence()
    positive = pp.make_trapezoid('x', flat_area=nx/fov, flat_time=nx*dwell, rise_time=.0001, fall_time=.0001)
    negative = deepcopy(positive)
    negative.amplitude *= -1
    negative.area *= -1
    negative.flat_area *= -1
    pre = pp.make_trapezoid('x', area=-positive.area/2-.5/fov, duration=.001)
    line_duration = pp.calc_duration(positive)
    blip = pp.make_trapezoid('y', area=1/fov, duration=.0002)
    trigger = pp.make_digital_output_pulse('ext1', duration=10e-6)
    imaging_start = .0004+.001+3*line_duration+.001
    center_line = ny//2
    # Row zero is negative; row two is also negative. kx=0 is its fourth
    # physical ADC sample (nx/2 - 1), independent of grid sorting.
    echo_time = imaging_start-.0002+center_line*(line_duration+.0002)+.0001+(nx/2-.5)*dwell
    for repetition in range(2+nz):
        rf_phase = repetition*.23
        rf = pp.make_block_pulse(np.deg2rad(20.), duration=.0004, use='excitation', phase_offset=rf_phase)
        sequence.add_block(rf)
        if repetition < 2:
            sequence.add_block(pp.make_delay(tr-.0004))
            continue
        partition = repetition-2
        sequence.add_block(pre)
        for nav in range(3):
            gx = negative if nav == 1 else positive
            adc = pp.make_adc(nx, dwell=dwell, delay=gx.rise_time, phase_offset=rf_phase)
            sequence.add_block(gx, adc)
        sequence.add_block(pp.make_trapezoid('y', area=-ny/2/fov, duration=.001),
                           pp.make_trapezoid('z', area=(partition-nz/2)/fov, duration=.001))
        for line in range(ny):
            gx = negative if line % 2 == 0 else positive
            adc = pp.make_adc(nx, dwell=dwell, delay=gx.rise_time, phase_offset=rf_phase)
            extras = [] if missing_trigger and partition == nz-1 and line == ny-1 else [trigger]
            sequence.add_block(gx, adc, *extras)
            if line != ny-1:
                sequence.add_block(blip)
            if different_last_shot and partition == nz-1 and line == 0:
                sequence.add_block(pp.make_delay(.0001))
        elapsed = imaging_start+ny*line_duration+(ny-1)*.0002
        if different_last_shot and partition == nz-1:
            elapsed += .0001
        sequence.add_block(pp.make_delay(tr-elapsed))
    for key, value in dict(Name='synthetic_3d_epi_fixture', AcquisitionDimension=3,
                            Matrix=[nx, ny, nz], FOV=[fov]*3, NavigatorEchoes=3,
                            TR=tr, TE=echo_time).items():
        sequence.set_definition(key, value)
    sequence.write(str(path))
    return path


class VolumeEPITests(unittest.TestCase):
    def test_actual_pulseq_3d_sequence_retains_rf_adc_kz_and_zero_gaps(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = make_sequence(Path(temporary)/'three_d.seq')
            acquisition, timing = read_epi3d_sequence(path)
        self.assertEqual((acquisition['nx'], acquisition['ny'], acquisition['nz']), (8, 4, 4))
        np.testing.assert_allclose(acquisition['kz_partition_mInv'], [-10., -5., 0., 5.], atol=1e-3)
        self.assertEqual(timing['dummy_rf_count'], 2)
        self.assertEqual(timing['trigger_count'], 16)
        self.assertEqual(timing['waveform_event_state_index'], [0, 1, 2, 3, -1]*4)
        np.testing.assert_allclose(np.diff(timing['active_rf_center_sequence_s']), .01, atol=1e-12)
        self.assertLess(timing['shared_partition_checks']['ADC timing'], 1e-12)
        starts = np.asarray(timing['global_state_start_sequence_s']).reshape(4, 4)
        ends = np.asarray(timing['global_state_end_sequence_s']).reshape(4, 4)
        self.assertTrue(np.all(ends[:-1, -1] < starts[1:, 0]))
        np.testing.assert_array_equal(acquisition['rf_time_s'], np.zeros((4, 8)))
        self.assertTrue(np.all(np.diff(acquisition['time_s'][0]) < 0))
        json.dumps(timing, allow_nan=False)

    def test_actual_sequence_rejects_partition_dependent_timing_and_missing_trigger(self):
        with tempfile.TemporaryDirectory() as temporary:
            for options in ({'different_last_shot': True}, {'missing_trigger': True}):
                with self.subTest(options=options):
                    path = make_sequence(Path(temporary)/'invalid.seq', **options)
                    with self.assertRaises(ValueError):
                        read_epi3d_sequence(path)

    def test_full_complex_volume_and_each_partitions_navigator_phase_are_recovered(self):
        scan, expected, corrections = fixture()
        results = list(iter_epi3d_receivers('in_memory.dat', scan=scan))
        self.assertEqual([item[0] for item in results], [0, 1])
        for receiver, actual, report in results:
            np.testing.assert_allclose(actual, expected[receiver], rtol=2e-13, atol=2e-13)
            np.testing.assert_allclose(report['navigator_phase_correction_rad'], corrections[receiver],
                                       rtol=0., atol=1e-13)
            self.assertEqual(report['navigator_fit_partition_count'], 4)
            self.assertEqual(report['array_axes'], ['readout', 'phase', 'partition'])
            json.dumps(report, allow_nan=False)

    def test_selected_receiver_retains_its_own_complex_image_and_index(self):
        scan, expected, _ = fixture()
        results = list(iter_epi3d_receivers('in_memory.dat', [1], scan=scan))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], 1)
        np.testing.assert_allclose(results[0][1], expected[1], rtol=2e-13, atol=2e-13)
        for indices in ([1, 1], [False], [2], [], [.5]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                list(iter_epi3d_receivers('in_memory.dat', indices, scan=scan))

    def test_missing_duplicate_rows_and_bad_navigators_are_rejected(self):
        original, _, _ = fixture()
        image_index = next(index for index, block in enumerate(original['mdb']) if block.is_image_scan())
        missing = deepcopy(original)
        del missing['mdb'][image_index]
        duplicate = deepcopy(original)
        duplicate['mdb'][image_index].cLin = duplicate['mdb'][image_index+1].cLin
        bad_nav = deepcopy(original)
        bad_nav['mdb'][1].flags.remove('REFLECT')
        other_echo = deepcopy(original)
        other_echo['mdb'][-1].cEco = 1
        for name, scan in [('missing', missing), ('duplicate', duplicate), ('navigator', bad_nav), ('echo', other_echo)]:
            with self.subTest(case=name), self.assertRaises(ValueError):
                list(iter_epi3d_receivers('in_memory.dat', scan=scan))

    def test_memmap_channel_offsets_preserve_data_and_reflect_without_loading_other_channels(self):
        from twixtools.mdh_def import Scan_header, Channel_header
        nx, receivers = 8, 3
        header = Scan_header()
        header.SamplesInScan, header.UsedChannels = nx, receivers
        offset = 17
        data = np.arange(receivers*nx).reshape(receivers, nx)*(1.+2.j)
        payload = bytearray(offset)+bytes(header)
        for receiver in range(receivers):
            payload.extend(bytes(Channel_header()))
            payload.extend(np.asarray(data[receiver], np.complex64).tobytes())
        block = SimpleNamespace(mem_pos=offset, version_is_ve=True, mdh=header,
            block_len=ctypes.sizeof(Channel_header)+nx*8, is_flag_set=lambda flag: flag == 'REFLECT')
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'raw.dat'
            path.write_bytes(payload)
            mapping = np.memmap(path, dtype=np.uint8, mode='r')
            try:
                # There is deliberately no .data property on this file-backed block.
                actual = _ordered_receiver(block, 1, mapping)
                np.testing.assert_array_equal(actual, data[1, ::-1])
            finally:
                mapping._mmap.close()


if __name__ == '__main__':
    unittest.main()
