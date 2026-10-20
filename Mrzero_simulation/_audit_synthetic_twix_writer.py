from pathlib import Path
import json
import sys
import tempfile
import struct
import twixtools
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / 'Mrzero_simulation'), str(ROOT / 'Experiment')]
from synthetic_acquisition import read_adc_samples
from synthetic_twix import write_synthetic_twix
from prepare_acdc_inputs import read_scan, read_gre
from read_acdc_b0 import read_b0_coil_images
from read_acdc_sequence import read_sequence
sequences = {'reference': next((ROOT / 'acquisition_015/sequences').glob('gre_32*.seq')),
             'scan': next((ROOT / 'acquisition_015/sequences').glob('*epi*.seq')),
             'b0': ROOT / 'acquisition_015/sequences/b0_map.seq'}
rng = np.random.default_rng(15)
reports = []
with tempfile.TemporaryDirectory(dir=ROOT / 'Mrzero_simulation', prefix='_audit_twix_') as temporary:
    for coils in (1, 2):
        for kind, sequence in sequences.items():
            samples = read_adc_samples(sequence)
            shape = (coils,) + samples['elapsed_s'].shape
            signal = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
            output = Path(temporary) / f'{kind}_{coils}.dat'
            report = write_synthetic_twix(output, signal, samples, kind)
            decoded = twixtools.read_twix(str(output), verbose=False)[-1]
            offset = int(decoded['raidfile_hdr']['off_'])
            header_length = len(decoded['hdr_str'])
            assert header_length % 512 == 0
            assert (offset + header_length) % 512 == 0
            report['protocol_header_length_bytes'] = header_length
            report['first_mdh_offset_bytes'] = offset + header_length
            report['first_mdh_aligned_512_bytes'] = True
            if kind == 'b0':
                result = read_b0_coil_images(output, sequence)
                report['reader_shape'] = list(result['coil_images'].shape)
            else:
                acq, _ = read_sequence(sequence)
                if kind == 'reference':
                    image, _ = read_gre(output, acq, return_intermediates=True)
                    report['reader_shape'] = list(image['magnitude'].shape)
                else:
                    image, _, _ = read_scan(output, acq, return_intermediates=True)
                    report['reader_shape'] = list(image.shape)
            report['kind'] = kind
            reports.append(report)
result = {'interpreter': sys.executable, 'native_reader_tests_passed': len(reports),
          'all_binary_roundtrips_exact': all(r['roundtrip_exact'] for r in reports),
          'all_first_mdh_offsets_aligned_512_bytes': all(r['first_mdh_aligned_512_bytes'] for r in reports),
          'cases': reports}
(ROOT / 'Mrzero_simulation/synthetic_twix_validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
