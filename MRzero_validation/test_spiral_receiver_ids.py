"""Receiver identity must agree across spiral transport segments and NAVs."""

from types import SimpleNamespace
import unittest

from spiral_inputs import _spiral_twix_data
from test_spiral_inputs import raw_fixture


class SpiralReceiverIdentityTests(unittest.TestCase):
    def fixture(self):
        scan, acquisition, expected = raw_fixture()
        for block in scan['mdb']:
            if block.is_image_scan() or block.is_flag_set('PHASCOR'):
                block.channel_hdr = [SimpleNamespace(ChannelId=value) for value in [33, 15]]
        return scan, acquisition

    def test_nonconsecutive_receiver_ids_are_preserved_in_data_row_order(self):
        scan, acquisition = self.fixture()
        metadata, _ = _spiral_twix_data(scan, acquisition)
        self.assertEqual(metadata['receiver_channel_ids'], [33, 15])

    def test_unknown_synthetic_receiver_ids_are_not_invented(self):
        scan, acquisition, _ = raw_fixture()
        metadata, _ = _spiral_twix_data(scan, acquisition)
        self.assertNotIn('receiver_channel_ids', metadata)

    def test_each_segment_and_navigator_must_retain_receiver_order(self):
        for changed_index in (0, 1, 2, 3, 5, 6):
            scan, acquisition = self.fixture()
            scan['mdb'][changed_index].channel_hdr.reverse()
            with self.subTest(changed_index=changed_index):
                with self.assertRaisesRegex(ValueError, 'receiver channel IDs or their order change'):
                    _spiral_twix_data(scan, acquisition)

    def test_missing_receiver_header_after_known_header_is_rejected(self):
        scan, acquisition = self.fixture()
        del scan['mdb'][0].channel_hdr
        with self.assertRaisesRegex(ValueError, 'known and missing'):
            _spiral_twix_data(scan, acquisition)

    def test_duplicate_receiver_header_is_rejected(self):
        scan, acquisition = self.fixture()
        scan['mdb'][-1].channel_hdr[1].ChannelId = 33
        with self.assertRaisesRegex(ValueError, 'duplicate receiver channel IDs'):
            _spiral_twix_data(scan, acquisition)


if __name__ == '__main__':
    unittest.main()
