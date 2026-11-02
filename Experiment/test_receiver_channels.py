"""Physical receiver IDs, data-row ordering and B0 metadata propagation."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from read_acdc_b0 import read_b0_coil_images
from receiver_channels import receiver_channel_ids
from test_read_acdc_b0 import coil_images, fake_twix


def with_channels(block, channels):
    block.channel_hdr = [SimpleNamespace(ChannelId=value) for value in channels]
    return block


class ReceiverIdentityTests(unittest.TestCase):
    def test_nonconsecutive_ids_follow_data_rows_and_are_never_sorted(self):
        channels = [33, 0, 15, 31]
        blocks = [with_channels(SimpleNamespace(), channels) for _ in range(3)]
        self.assertEqual(receiver_channel_ids(blocks, 4), channels)

    def test_missing_headers_on_all_blocks_remain_unknown(self):
        blocks = [SimpleNamespace(), SimpleNamespace(channel_hdr=None)]
        self.assertIsNone(receiver_channel_ids(blocks, 4))

    def test_mixed_known_and_unknown_ids_are_rejected(self):
        blocks = [with_channels(SimpleNamespace(), [0, 33]), SimpleNamespace()]
        with self.assertRaisesRegex(ValueError, 'known and missing'):
            receiver_channel_ids(blocks, 2)

    def test_reordering_duplicate_ids_and_incomplete_headers_are_rejected(self):
        for channels, pattern in (([33, 0], 'order change'), ([0, 0], 'duplicate'),
                                  ([0], 'header count'), ([], 'header count')):
            blocks = [with_channels(SimpleNamespace(), [0, 33]),
                      with_channels(SimpleNamespace(), channels)]
            with self.assertRaisesRegex(ValueError, pattern):
                receiver_channel_ids(blocks, 2)

    def test_invalid_channel_identifiers_are_rejected(self):
        for value in (-1, 1.2, np.nan, np.inf, True, None, 1j):
            block = with_channels(SimpleNamespace(), [0, value])
            with self.assertRaisesRegex(ValueError, 'ChannelId'):
                receiver_channel_ids([block], 2)

    def test_b0_reader_preserves_verified_receiver_ids_in_metadata(self):
        scans = fake_twix(coil_images(np.full((4, 6), 30.)))
        channels = [0, 13, 15, 33]
        for block in scans[0]['mdb']:
            with_channels(block, channels)
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'synthetic.dat'
            path.write_bytes(b'B0 receiver identity fixture')
            with patch('twixtools.read_twix', return_value=scans):
                result = read_b0_coil_images(path)
        self.assertEqual(result['metadata']['receiver_channel_ids'], channels)
        np.testing.assert_allclose(result['coil_images'], coil_images(np.full((4, 6), 30.)), atol=1e-12)

    def test_b0_reader_rejects_receiver_order_change_between_echoes(self):
        scans = fake_twix(coil_images(np.full((4, 6), 30.)))
        for block in scans[0]['mdb']:
            with_channels(block, [0, 13, 15, 33])
        with_channels(scans[0]['mdb'][-1], [0, 15, 13, 33])
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'synthetic.dat'
            path.write_bytes(b'B0 receiver order fixture')
            with patch('twixtools.read_twix', return_value=scans):
                with self.assertRaisesRegex(ValueError, 'B0 Twix receiver channel IDs or their order change'):
                    read_b0_coil_images(path)


if __name__ == '__main__':
    unittest.main()
