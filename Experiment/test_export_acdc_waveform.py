"""Check the waveform CSV format and rejection of invalid data."""

from pathlib import Path
import hashlib
import tempfile
import unittest

import numpy as np

from export_acdc_waveform import (export_waveform, export_zero_time_copy,
                                  quantize_currents, validate_waveform_settings)


class WaveformExportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "output"
        self.currents = np.array([[0.2, -0.3], [-0.4, 0.5]])
        self.edges = np.array([0.002, 0.003, 0.004])
        self.settings = {
            "channel_map": [10, 2], "channel_polarity": [-1, 1],
            "max_current_A": [0.6, 0.6], "total_current_limit_A": 2.0,
        }

    def export(self):
        return export_waveform(self.currents, self.edges, self.settings, self.output)

    def assert_rejected(self, pattern):
        with self.assertRaisesRegex(ValueError, pattern):
            self.export()
        self.assertFalse(self.output.exists())

    def test_writes_only_csv_with_mapping_and_zero_boundaries(self):
        result = self.export()
        self.assertEqual([p.name for p in self.output.iterdir()], ["waveform.csv"])
        self.assertEqual(Path(result["csv_file"]), self.output / "waveform.csv")
        data = np.loadtxt(result["csv_file"], delimiter=",")
        self.assertEqual(data.shape, (4, 17))  # time + two complete eight-output boards
        np.testing.assert_array_equal(data[:, 0], [0, *self.edges])
        np.testing.assert_array_equal(data[[0, -1], 1:], 0)
        np.testing.assert_array_equal(data[1:-1, 10], -self.currents[0])
        np.testing.assert_array_equal(data[1:-1, 2], self.currents[1])
        unused = [k + 1 for k in range(16) if k not in (1, 9)]
        np.testing.assert_array_equal(data[:, unused], 0)
        self.assertEqual(result["physical_channels"], 16)
        self.assertEqual(result["states"], 2)
        self.assertEqual(result["final_zero_s"], self.edges[-1])
        self.assertTrue(Path(result["csv_file"]).read_text().startswith("# time_s,ch0_A,ch1_A,"))

    def test_channel_64_pads_eight_boards(self):
        self.settings["channel_map"] = [1, 64]
        result = self.export()
        data = np.loadtxt(result["csv_file"], delimiter=",")
        self.assertEqual(data.shape, (4, 65))
        np.testing.assert_array_equal(data[1:-1, 1], -self.currents[0])
        np.testing.assert_array_equal(data[1:-1, 64], self.currents[1])

    def test_scalar_current_limit_is_expanded(self):
        self.settings["max_current_A"] = 0.6
        self.assertEqual(self.export()["settings"]["max_current_A"], [0.6, 0.6])

    def test_disabled_physical_output_follows_mapping_and_preserves_active_polarity(self):
        self.settings["disabled_channels"] = [2]
        self.currents[1] = 0.
        result = self.export()
        data = np.loadtxt(result["csv_file"], delimiter=",")
        self.assertEqual(result["settings"]["max_current_A"], [0.6, 0.])
        self.assertEqual(result["settings"]["disabled_channels"], [2])
        np.testing.assert_array_equal(data[:, 2], 0.)
        np.testing.assert_array_equal(data[1:-1, 10], -self.currents[0])

    def test_disabled_output_rejects_even_tiny_nonzero_current_before_export(self):
        self.settings["disabled_channels"] = [10]
        for value in (0.2, -0.2, 1e-30, -1e-30):
            with self.subTest(value=value):
                self.currents[0] = [value, 0.]
                self.assert_rejected("exceed max_current_A")
                self.assertEqual(self.currents[0, 0], value)

    def test_disabled_channel_normalization_copies_limits_and_is_idempotent(self):
        limits = np.array([0.6, 0.4])
        self.settings.update(max_current_A=limits, disabled_channels=[np.int64(10)])
        normalized = validate_waveform_settings(self.settings, 2)
        self.assertEqual(normalized["max_current_A"], [0., 0.4])
        self.assertEqual(normalized["disabled_channels"], [10])
        np.testing.assert_array_equal(limits, [0.6, 0.4])
        self.assertEqual(validate_waveform_settings(normalized, 2), normalized)

    def test_explicit_zero_bounds_report_disabled_physical_outputs(self):
        self.settings.update(max_current_A=[0., 0.6], disabled_channels=[])
        self.currents[0] = 0.
        result = self.export()
        self.assertEqual(result["settings"]["disabled_channels"], [10])
        self.assertEqual(result["settings"]["max_current_A"], [0., 0.6])
        np.testing.assert_array_equal(np.loadtxt(result["csv_file"], delimiter=",")[:, 10], 0.)

    def test_all_outputs_disabled_can_export_only_zero_currents(self):
        self.settings.update(max_current_A=0.6, disabled_channels=[10, 2])
        self.currents[:] = 0.
        result = self.export()
        self.assertEqual(result["settings"]["disabled_channels"], [2, 10])
        self.assertEqual(result["settings"]["max_current_A"], [0., 0.])
        np.testing.assert_array_equal(np.loadtxt(result["csv_file"], delimiter=",")[:, 1:], 0.)

    def test_rejects_invalid_disabled_physical_output_identifiers(self):
        for disabled in ([True], [np.bool_(False)], [10, True], [10.0], ["10"],
                         [np.nan], [None], [[10]], 10, None):
            with self.subTest(disabled=disabled):
                self.settings["disabled_channels"] = disabled
                self.assert_rejected("disabled_channels")
        for disabled in ([10, 10], [0], [65], [1]):
            with self.subTest(disabled=disabled):
                self.settings["disabled_channels"] = disabled
                self.assert_rejected("disabled_channels")

    def test_negative_current_bound_is_rejected_even_on_disabled_output(self):
        self.settings.update(max_current_A=[-0.1, 0.6], disabled_channels=[10])
        self.assert_rejected("nonnegative")

    def test_rejects_duplicate_physical_channels(self):
        self.settings["channel_map"] = [2, 2]
        self.assert_rejected("distinct integer")

    def test_rejects_invalid_channel_numbers_and_count(self):
        for channels in ([0, 2], [1, 65], [1.5, 2], [1], [1, 2, 3]):
            with self.subTest(channels=channels):
                self.settings["channel_map"] = channels
                self.assert_rejected("distinct integer")

    def test_rejects_invalid_polarity(self):
        self.settings["channel_polarity"] = [1, 0]
        self.assert_rejected("one \\+1 or -1")

    def test_rejects_missing_csv_settings(self):
        del self.settings["channel_map"]
        self.assert_rejected("Missing waveform settings")

    def test_rejects_nonfinite_or_complex_currents(self):
        self.currents[0, 0] = np.nan
        self.assert_rejected("finite")
        self.currents = np.ones((2, 2), dtype=complex) * 0.1j
        self.assert_rejected("real")

    def test_rejects_invalid_current_shape(self):
        self.currents = np.array([0.1, 0.2])
        self.assert_rejected("shape")

    def test_rejects_amplitude_without_clipping(self):
        self.currents[0, 0] = 0.60001
        self.assert_rejected("exceed max_current_A")

    def test_rejects_waveform_reader_amplitude_limit(self):
        self.settings["max_current_A"] = 3.1
        self.assert_rejected("at most 3 A")

    def test_rejects_total_current_at_limit(self):
        self.settings["total_current_limit_A"] = 0.8
        self.assert_rejected("total_current_limit_A")

    def test_rejects_nonpositive_total_current_limit(self):
        self.settings["total_current_limit_A"] = 0
        self.assert_rejected("positive scalar")

    def test_rejects_time_reset_inside_imaging(self):
        self.edges = self.edges[::-1]
        self.assert_rejected("increasing nonnegative")

    def test_state_start_at_time_origin_omits_zero_preamble(self):
        self.edges[0] = 0
        result = self.export()
        data = np.loadtxt(result["csv_file"], delimiter=",")
        self.assertEqual(data.shape, (3, 17))
        np.testing.assert_array_equal(data[:, 0], self.edges)
        np.testing.assert_array_equal(data[:-1, 10], -self.currents[0])
        np.testing.assert_array_equal(data[:-1, 2], self.currents[1])
        np.testing.assert_array_equal(data[-1, 1:], 0)

    def test_rebased_export_places_original_first_nonzero_command_at_zero(self):
        result = export_waveform(self.currents, self.edges, self.settings, self.output,
                                 start_at_first_nonzero=True)
        data = np.loadtxt(result["csv_file"], delimiter=",")
        np.testing.assert_array_equal(data[:, 0], self.edges - self.edges[0])
        np.testing.assert_array_equal(data[:-1, 10], -self.currents[0])
        np.testing.assert_array_equal(data[:-1, 2], self.currents[1])
        np.testing.assert_array_equal(data[-1, 1:], 0)
        np.testing.assert_array_equal(result["state_edges_s"], self.edges)
        np.testing.assert_array_equal(result["csv_state_edges_s"], data[:, 0])
        self.assertEqual(result["time_origin_shift_s"], self.edges[0])
        self.assertEqual(result["leading_zero_states"], 0)
        self.assertTrue(result["first_nonzero_found"])
        self.assertTrue(result["start_at_first_nonzero"])
        self.assertFalse(result["csv_includes_zero_preamble"])
        self.assertEqual(result["final_zero_s"], self.edges[-1])
        self.assertEqual(result["csv_final_zero_s"], data[-1, 0])

    def test_rebased_export_removes_only_initial_zero_states_and_preserves_intervals(self):
        currents = np.array([[0., 0., .2, 0., -.3], [0., 0., -.4, 0., .5]])
        edges = np.array([.005, .0052, .0057, .006, .0064, .0073])
        result = export_waveform(currents, edges, self.settings, self.output,
                                 start_at_first_nonzero=True)
        data = np.loadtxt(result["csv_file"], delimiter=",")
        np.testing.assert_array_equal(data[:, 0], edges[2:] - edges[2])
        np.testing.assert_array_equal(data[:-1, 10], -currents[0, 2:])
        np.testing.assert_array_equal(data[:-1, 2], currents[1, 2:])
        np.testing.assert_array_equal(data[[1, -1], 1:], 0)
        np.testing.assert_array_equal(np.diff(data[:, 0]), np.diff(edges[2:]))
        self.assertEqual(result["states"], 5)
        self.assertEqual(result["csv_states"], 3)
        self.assertEqual(result["leading_zero_states"], 2)
        self.assertEqual(result["time_origin_shift_s"], edges[2])
        np.testing.assert_array_equal(result["state_edges_s"], edges)
        self.assertEqual(result["final_zero_s"], edges[-1])
        copied = export_zero_time_copy(result["csv_file"])
        zero_time_data = np.loadtxt(copied["csv_file"], delimiter=",")
        self.assertEqual(zero_time_data.shape, data.shape)
        np.testing.assert_array_equal(zero_time_data[:, 0], 0)
        np.testing.assert_array_equal(zero_time_data[:, 1:], data[:, 1:])

    def test_rebased_export_with_only_zero_currents_keeps_states_and_reports_no_nonzero(self):
        result = export_waveform(np.zeros_like(self.currents), self.edges,
                                 self.settings, self.output, start_at_first_nonzero=True)
        data = np.loadtxt(result["csv_file"], delimiter=",")
        np.testing.assert_array_equal(data[:, 0], self.edges - self.edges[0])
        np.testing.assert_array_equal(data[:, 1:], 0)
        self.assertEqual(result["states"], self.currents.shape[1])
        self.assertEqual(result["csv_states"], self.currents.shape[1])
        self.assertEqual(result["leading_zero_states"], 0)
        self.assertFalse(result["first_nonzero_found"])
        self.assertEqual(result["time_origin_shift_s"], self.edges[0])

    def test_default_export_preserves_leading_zero_states_and_original_timing(self):
        self.currents[:, 0] = 0
        result = self.export()
        data = np.loadtxt(result["csv_file"], delimiter=",")
        np.testing.assert_array_equal(data[:, 0], [0, *self.edges])
        np.testing.assert_array_equal(data[:2, 1:], 0)
        self.assertEqual(result["leading_zero_states"], 0)
        self.assertEqual(result["time_origin_shift_s"], 0)
        self.assertFalse(result["start_at_first_nonzero"])
        np.testing.assert_array_equal(result["csv_state_edges_s"], self.edges)

    def test_rebased_export_without_final_row_preserves_original_command_count(self):
        result = export_waveform(self.currents, self.edges, self.settings, self.output,
                                 start_at_first_nonzero=True, include_final_zero_row=False)
        data = np.loadtxt(result["csv_file"], delimiter=",")
        self.assertEqual(data.shape[0], self.currents.shape[1])
        np.testing.assert_array_equal(data[:, 0], self.edges[:-1] - self.edges[0])
        self.assertIsNone(result["csv_final_zero_s"])

    def test_export_without_header_or_boundary_rows_uses_zero_time_column(self):
        result = export_waveform(
            self.currents, self.edges, self.settings, self.output,
            include_header=False, include_zero_preamble=False,
            include_final_zero_row=False, zero_time_column=True,
        )
        content = Path(result["csv_file"]).read_text()
        self.assertFalse(content.startswith("#"))
        self.assertEqual(len(content.splitlines()), self.currents.shape[1])
        data = np.loadtxt(result["csv_file"], delimiter=",")
        self.assertEqual(data.shape, (2, 17))
        np.testing.assert_array_equal(data[:, 0], np.zeros(2))
        np.testing.assert_array_equal(data[:, 10], -self.currents[0])
        np.testing.assert_array_equal(data[:, 2], self.currents[1])

    def test_rejects_nonfinite_or_incorrect_number_of_edges(self):
        self.edges[1] = np.inf
        self.assert_rejected("finite")
        self.edges = np.array([0.002, 0.003])
        self.assert_rejected("n_states\\+1")

    def test_optional_minimum_interval_checks_initial_and_current_states(self):
        self.settings["min_dac_interval_s"] = 10e-6
        self.edges[1] = self.edges[0] + 1e-6
        self.assert_rejected("interval below")
        self.edges = np.array([1e-6, 0.003, 0.004])
        self.assert_rejected("interval below")

    def test_rejects_invalid_optional_minimum_interval(self):
        self.settings["min_dac_interval_s"] = -1
        self.assert_rejected("positive scalar")

    def test_unknown_minimum_interval_does_not_invent_a_limit(self):
        self.settings["min_dac_interval_s"] = None
        self.edges = np.array([1e-9, 2e-9, 3e-9])
        result = self.export()
        self.assertNotIn("min_dac_interval_s", result["settings"])

    def test_csv_does_not_require_controller_programming_settings(self):
        normalized = validate_waveform_settings(self.settings, 2)
        self.assertEqual(set(normalized), set(self.settings) | {"disabled_channels"})
        self.assertEqual(normalized["disabled_channels"], [])
        self.export()
        self.assertEqual(len(list(self.output.iterdir())), 1)

    def test_quantization_matches_signed_c_conversion_and_polarity(self):
        values = np.array([-3., -0.20002, -1e-6, 0, 1e-6, 0.20002, 3.])
        expected = np.array([(int(float(value) / 5.0 * 32767.0) / 32767.0) * 5.0
                             for value in values])
        np.testing.assert_array_equal(quantize_currents(values), expected)
        np.testing.assert_array_equal(quantize_currents(-values), -expected)
        self.assertLess(np.max(np.abs(expected - values)), 5 / 32767)

    def test_quantization_can_increase_generated_field(self):
        currents = np.array([1.0, 0.0001])
        response_Hz_per_A = np.array([1.0, -1.0])
        self.assertGreater(response_Hz_per_A @ quantize_currents(currents),
                           response_Hz_per_A @ currents)

    def test_refuses_overwrite_without_modifying_existing_csv(self):
        result = self.export()
        before = Path(result["csv_file"]).read_bytes()
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertEqual(Path(result["csv_file"]).read_bytes(), before)
        self.assertEqual(len(list(self.output.iterdir())), 1)

    def test_numbered_exports_preserve_previous_files_and_time_column_choice(self):
        original = Path(self.export()["csv_file"])
        original_bytes = original.read_bytes()
        first = Path(export_waveform(
            self.currents, self.edges, self.settings, self.output,
            numbered_filename=True, zero_time_column=True,
        )["csv_file"])
        first_bytes = first.read_bytes()
        second = Path(export_waveform(
            self.currents, self.edges, self.settings, self.output,
            numbered_filename=True,
        )["csv_file"])
        self.assertEqual(first.name, "waveform1.csv")
        self.assertEqual(second.name, "waveform2.csv")
        self.assertEqual(original.read_bytes(), original_bytes)
        self.assertEqual(first.read_bytes(), first_bytes)
        first_data = np.loadtxt(first, delimiter=",")
        second_data = np.loadtxt(second, delimiter=",")
        np.testing.assert_array_equal(first_data[:, 0], 0)
        np.testing.assert_array_equal(second_data[:, 0], [0, *self.edges])
        np.testing.assert_array_equal(first_data[:, 1:], second_data[:, 1:])

    def test_numbered_export_continues_after_largest_suffix_when_numbers_have_gaps(self):
        self.output.mkdir()
        previous = self.output / "waveform9.csv"
        previous.write_text("previous export\n")
        (self.output / "waveform2.csv").write_text("older export\n")
        (self.output / "waveform_notes.csv").write_text("notes\n")
        result = export_waveform(
            self.currents, self.edges, self.settings, self.output,
            numbered_filename=True,
        )
        self.assertEqual(Path(result["csv_file"]).name, "waveform10.csv")
        self.assertEqual(previous.read_text(), "previous export\n")

    def test_named_exports_share_folder_and_preserve_all_previous_waveforms(self):
        self.output.mkdir()
        unrelated = self.output / "acquisition_notes.json"
        unrelated.write_text('{"reference": "measured"}\n')
        static = Path(export_waveform(
            self.currents, self.edges, self.settings, self.output,
            filename_prefix="waveform_static", numbered_filename=True,
        )["csv_file"])
        before = static.read_bytes()
        dynamic = Path(export_waveform(
            self.currents / 2, self.edges, self.settings, self.output,
            filename_prefix="waveform_dynamic", numbered_filename=True,
        )["csv_file"])
        second = Path(export_waveform(
            self.currents, self.edges, self.settings, self.output,
            filename_prefix="waveform_static", numbered_filename=True,
        )["csv_file"])
        self.assertEqual(static.name, "waveform_static.csv")
        self.assertEqual(second.name, "waveform_static2.csv")
        self.assertEqual(dynamic.name, "waveform_dynamic.csv")
        self.assertEqual(static.read_bytes(), before)
        self.assertEqual(second.read_bytes(), before)
        self.assertEqual(unrelated.read_text(), '{"reference": "measured"}\n')
        data = np.loadtxt(dynamic, delimiter=",")
        np.testing.assert_array_equal(data[:, 0], [0, *self.edges])
        np.testing.assert_array_equal(data[1:-1, 10], -self.currents[0] / 2)
        np.testing.assert_array_equal(data[1:-1, 2], self.currents[1] / 2)

    def test_named_export_without_numbering_refuses_overwrite_in_shared_folder(self):
        first = Path(export_waveform(self.currents, self.edges, self.settings, self.output,
                                    filename_prefix="waveform_static")["csv_file"])
        before = first.read_bytes()
        with self.assertRaisesRegex(FileExistsError, "never overwritten"):
            export_waveform(self.currents, self.edges, self.settings, self.output,
                            filename_prefix="waveform_static")
        self.assertEqual(first.read_bytes(), before)

    def test_named_export_continues_after_largest_number_and_ignores_other_stems(self):
        self.output.mkdir()
        for name in ("waveform_static.csv", "waveform_static7.csv", "waveform_dynamic90.csv",
                     "waveform_static_notes.csv"):
            (self.output / name).write_text("previous export\n")
        result = export_waveform(self.currents, self.edges, self.settings, self.output,
                                filename_prefix="waveform_static", numbered_filename=True)
        self.assertEqual(Path(result["csv_file"]).name, "waveform_static8.csv")
        self.assertEqual((self.output / "waveform_static7.csv").read_text(), "previous export\n")

    def test_named_export_rejects_paths_extensions_and_empty_or_nonstring_stems(self):
        for invalid in ("", ".", "..", "../waveform_static", "nested/waveform_dynamic",
                        "waveform_static.csv", "/tmp/waveform", "waveform\\static", 123, True):
            with self.subTest(prefix=invalid), self.assertRaisesRegex(ValueError, "filename_prefix"):
                export_waveform(self.currents, self.edges, self.settings, self.output,
                                filename_prefix=invalid)
            self.assertFalse(self.output.exists())

    def test_zero_time_companion_preserves_mapped_currents_and_boundary_rows(self):
        exported = export_waveform(self.currents, self.edges, self.settings, self.output,
                                   numbered_filename=True)
        source = Path(exported["csv_file"])
        original = source.read_bytes()
        result = export_zero_time_copy(source)
        output = Path(result["csv_file"])
        self.assertEqual(output.name, "waveform1_zero_time.csv")
        self.assertEqual(result["source_csv_file"], str(source))
        self.assertEqual(result["source_sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(result["csv_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
        self.assertTrue(result["time_column_all_zero"])
        self.assertFalse(result["already_present"])
        self.assertEqual(result["rows"], 4)
        self.assertEqual(result["physical_channels"], 16)
        self.assertEqual(source.read_bytes(), original)
        before = np.loadtxt(source, delimiter=",")
        after = np.loadtxt(output, delimiter=",")
        np.testing.assert_array_equal(after[:, 0], 0)
        np.testing.assert_array_equal(after[:, 1:], before[:, 1:])
        np.testing.assert_array_equal(after[[0, -1], 1:], 0)
        np.testing.assert_array_equal(after[1:-1, 10], -self.currents[0])
        np.testing.assert_array_equal(after[1:-1, 2], self.currents[1])
        original_lines = original.splitlines(keepends=True)
        copied_lines = output.read_bytes().splitlines(keepends=True)
        self.assertEqual(copied_lines[0], original_lines[0])
        self.assertEqual([line.partition(b",")[2] for line in copied_lines[1:]],
                         [line.partition(b",")[2] for line in original_lines[1:]])

    def test_zero_time_companion_preserves_comments_crlf_and_numeric_spelling(self):
        self.output.mkdir()
        source = self.output / "waveform1.csv"
        original = (b"# time_s,ch0_A,ch1_A\r\n\r\n"
                    b"0.000271, 0.20000000000000001,-0.0000000000000000\r\n"
                    b"# Keep this comment exactly.\r\n"
                    b"0.00054,-2.0000000000000001e-1,+0.5\r\n"
                    b"0.00081,0,0")
        source.write_bytes(original)
        result = export_zero_time_copy(source)
        expected = original.replace(b"0.000271,", b"0,").replace(
            b"0.00054,", b"0,").replace(b"0.00081,", b"0,")
        self.assertEqual(Path(result["csv_file"]).read_bytes(), expected)
        self.assertEqual(source.read_bytes(), original)

    def test_zero_time_companion_accepts_identical_file_and_refuses_different_file(self):
        source = Path(self.export()["csv_file"])
        result = export_zero_time_copy(source)
        copied = Path(result["csv_file"])
        self.assertTrue(export_zero_time_copy(source)["already_present"])
        copied.write_bytes(b"an existing different waveform\n")
        original = source.read_bytes()
        with self.assertRaisesRegex(FileExistsError, "never overwritten"):
            export_zero_time_copy(source)
        self.assertEqual(copied.read_bytes(), b"an existing different waveform\n")
        self.assertEqual(source.read_bytes(), original)

    def test_zero_time_companions_follow_source_number_without_consuming_a_number(self):
        first = Path(export_waveform(self.currents, self.edges, self.settings, self.output,
                                    numbered_filename=True)["csv_file"])
        first_copy = export_zero_time_copy(first)
        second = Path(export_waveform(self.currents, self.edges, self.settings, self.output,
                                     numbered_filename=True)["csv_file"])
        second_copy = export_zero_time_copy(second)
        self.assertEqual(first.name, "waveform1.csv")
        self.assertEqual(Path(first_copy["csv_file"]).name, "waveform1_zero_time.csv")
        self.assertEqual(second.name, "waveform2.csv")
        self.assertEqual(Path(second_copy["csv_file"]).name, "waveform2_zero_time.csv")

    def test_zero_time_companion_rejects_source_as_destination(self):
        source = Path(self.export()["csv_file"])
        original = source.read_bytes()
        with self.assertRaisesRegex(ValueError, "differ from the source"):
            export_zero_time_copy(source, source)
        self.assertEqual(source.read_bytes(), original)

    def test_zero_time_companion_rejects_malformed_or_empty_data_before_writing(self):
        self.output.mkdir()
        source = self.output / "waveform1.csv"
        destination = self.output / "custom_zero_time.csv"
        for malformed in (b"# header only\n", b"0,nan\n", b"0,0,0\n1,0\n",
                          b"0,a-current\n", b"0\n"):
            with self.subTest(source=malformed):
                source.write_bytes(malformed)
                with self.assertRaisesRegex(ValueError, "Source CSV"):
                    export_zero_time_copy(source, destination)
                self.assertFalse(destination.exists())
                self.assertEqual(source.read_bytes(), malformed)


if __name__ == "__main__":
    unittest.main()
