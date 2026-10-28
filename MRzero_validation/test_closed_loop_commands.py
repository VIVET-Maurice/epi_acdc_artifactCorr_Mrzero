"""Dry-run command provenance, DAC identity and fail-closed exports."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from acdc_hardware import HardwareConstraints
import closed_loop_commands as commands
from export_acdc_waveform import export_waveform, quantize_currents


def make_context(folder, *, origin="first_imaging_trigger", field_limit=500., basis=None):
    starts = .001 + np.arange(3) * .001
    ends = starts + .001
    model = SimpleNamespace(nc=2, nstates=3, line_start_s=starts, line_end_s=ends,
        current_response={"enabled": True, "tau_s": 50e-6},
        tensor=lambda value: torch.as_tensor(value, dtype=torch.float64),
        numpy=lambda value: value.detach().cpu().numpy())
    model.max_current_A = model.tensor([3., 3.])
    model.B_quality = model.tensor([[20., -20.], [10., 8.]] if basis is None else basis)
    model._check_currents = lambda value: model.tensor(value)
    model.hardware = HardwareConstraints(model, dict(cycle_duration_s=.006,
        min_update_interval_s=8e-6, max_slew_A_per_s=1e5,
        coil_R_ohm=[.3, .4], coil_L_H=[20e-6, 30e-6], max_voltage_V=[5., 5.]))
    settings = dict(channel_map=[3, 1], channel_polarity=[-1, 1], max_current_A=3.,
        total_current_limit_A=5., min_dac_interval_s=8e-6)
    path = Path(folder) / "example.seq"
    path.write_text("unit-test mock pulse sequence")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    inputs = dict(paths={"scan_sequence": path},
                  timing=dict(state_start_s=starts, state_end_s=ends),
                  source_sha256={"scan_sequence": digest})
    prepared = dict(report={"calibration_source_sha256": "b" * 64})
    trigger = dict(trigger_rf_s=.001, trigger_sequence_s=.002, channel="ext1",
        verified_first_imaging_trigger=True, sequence_sha256=digest,
        hardware_latency_measured=False)
    with patch.object(commands, "sequence_trigger_timing", return_value=trigger):
        context = commands.build_command_context(model, inputs, prepared,
            waveform_parameters=settings, csv_time_origin=origin, trigger_channel="ext1",
            total_current_limit_A=5., field_limit_Hz=field_limit)
    return context


def historical(folder, context, raw=None):
    raw = np.array([[.41, -.52, .63], [-.22, .32, -.42]]) if raw is None else raw
    folder = Path(folder)
    exported = export_waveform(raw, context["csv_edges_s"], context["metadata"]["settings"],
                               folder / "historical", numbered_filename=True)
    metadata = dict(deepcopy(context["metadata"]), waveform_export=exported)
    path = folder / "old.npz"
    np.savez(path, currents_A=raw, quantized_currents_A=quantize_currents(raw))
    path.with_suffix(".json").write_text(json.dumps(metadata))
    return Path(exported["csv_file"]), path, metadata


class CommandTests(unittest.TestCase):
    def test_context_restarts_without_model_and_detects_mutation(self):
        with tempfile.TemporaryDirectory(prefix="closed_loop_test_") as folder:
            context = make_context(folder)
            path = Path(folder) / "context.npz"
            commands.save_command_context(context, path)
            loaded = commands.load_command_context(path)
            self.assertEqual(loaded["fingerprint"], context["fingerprint"])
            np.testing.assert_array_equal(loaded["field_basis_Hz_per_A"], context["field_basis_Hz_per_A"])
            with self.assertRaises(FileExistsError):
                commands.save_command_context(context, path)
            loaded["max_current_A"][0] = 2.
            with self.assertRaisesRegex(ValueError, "integrity"):
                commands.evaluate_candidate(loaded, np.zeros((2, 3)), .5)

    def test_historical_npz_csv_exact_match_and_DAC_preference(self):
        with tempfile.TemporaryDirectory(prefix="closed_loop_test_") as folder:
            context = make_context(folder)
            csv, npz, _ = historical(folder, context)
            result = commands.load_historical_commands(context, csv_path=csv, npz_path=npz)
            self.assertTrue(result["report"]["stored_DAC_waveform_used"])
            self.assertFalse(result["report"]["scan_current_association_verified"])
            candidate = commands.evaluate_candidate(context, result["quantized_currents_A"], 1.)
            self.assertTrue(candidate["valid"])
            self.assertEqual(candidate["fingerprint"], result["fingerprint"])
            np.testing.assert_array_equal(candidate["quantized_currents_A"], result["quantized_currents_A"])

    def test_optional_disabled_metadata_is_compatible_with_persisted_legacy_contexts(self):
        for legacy_context in (False, True):
            for legacy_export in (False, True):
                with self.subTest(legacy_context=legacy_context, legacy_export=legacy_export):
                    with tempfile.TemporaryDirectory(prefix="closed_loop_test_") as folder:
                        context = make_context(folder)
                        csv, npz, metadata = historical(folder, context)
                        if legacy_context:
                            del context["metadata"]["settings"]["disabled_channels"]
                            context["fingerprint"] = commands._context_digest(context)
                        if legacy_export:
                            del metadata["waveform_export"]["settings"]["disabled_channels"]
                            del metadata["settings"]["disabled_channels"]
                        path = Path(folder) / "context.npz"
                        commands.save_command_context(context, path)
                        original = path.read_bytes()
                        loaded = commands.load_command_context(path)
                        original_metadata = deepcopy(loaded["metadata"])
                        result = commands.load_historical_commands(
                            loaded, csv_path=csv, npz_path=npz, metadata=metadata)
                        self.assertTrue(result["report"]["stored_DAC_waveform_used"])
                        self.assertEqual(loaded["metadata"], original_metadata)
                        self.assertEqual(loaded["fingerprint"], context["fingerprint"])
                        self.assertEqual(path.read_bytes(), original)

    def test_csv_only_requires_explicit_metadata_and_unmapped_outputs_zero(self):
        with tempfile.TemporaryDirectory(prefix="closed_loop_test_") as folder:
            context = make_context(folder)
            csv, _, metadata = historical(folder, context)
            with self.assertRaisesRegex(ValueError, "metadata missing"):
                commands.load_historical_commands(context, csv_path=csv)
            loaded = commands.load_historical_commands(context, csv_path=csv, metadata=metadata)
            self.assertFalse(loaded["report"]["stored_DAC_waveform_used"])
            text = np.loadtxt(csv, delimiter=",")
            text[0, 2] = .01  # physical output 2 is not mapped
            np.savetxt(csv, text, delimiter=",", fmt="%.17g",
                       header="time_s," + ",".join(f"ch{k}_A" for k in range(8)))
            with self.assertRaisesRegex(ValueError, "unmapped"):
                commands.load_historical_commands(context, csv_path=csv, metadata=metadata)

    def test_wrong_mapping_polarity_sequence_calibration_origin_trigger_fail(self):
        with tempfile.TemporaryDirectory(prefix="closed_loop_test_") as folder:
            context = make_context(folder)
            csv, _, metadata = historical(folder, context)
            changes = [
                ("waveform_export", "settings", "channel_map", [1, 3]),
                ("waveform_export", "settings", "channel_polarity", [1, 1]),
                ("waveform_export", "settings", "disabled_channels", [3]),
                ("source_sha256", "scan_sequence", "c" * 64),
                ("source_sha256", "calibration", "c" * 64),
                ("CSV_timing", "origin", "rf"),
                ("CSV_timing", "origin_rf_s", .002),
                ("trigger", "channel", "ext2"),
                ("trigger", "verified_first_imaging_trigger", False),
                ("trigger", "trigger_rf_s", .002),
                ("current_response", "tau_s", 100e-6),
            ]
            for change in changes:
                mutated = deepcopy(metadata)
                target = mutated
                for key in change[:-2]:
                    target = target[key]
                target[change[-2]] = change[-1]
                with self.subTest(change=change):
                    with self.assertRaises(ValueError):
                        commands.load_historical_commands(context, csv_path=csv, metadata=mutated)

    def test_timing_final_zero_and_archived_DAC_tampering_fail(self):
        with tempfile.TemporaryDirectory(prefix="closed_loop_test_") as folder:
            context = make_context(folder, origin="rf")
            csv, npz, metadata = historical(folder, context)
            original = np.loadtxt(csv, delimiter=",")
            for change in ("timestamps", "final", "preamble"):
                mutated = original.copy()
                if change == "timestamps":
                    mutated[1, 0] += 1e-5
                elif change == "final":
                    mutated[-1, 1] = .1
                else:
                    mutated[0, 1] = .1
                np.savetxt(csv, mutated, delimiter=",", fmt="%.17g",
                           header="time_s," + ",".join(f"ch{k}_A" for k in range(8)))
                with self.subTest(change=change):
                    with self.assertRaises(ValueError):
                        commands.load_historical_commands(context, csv_path=csv, metadata=metadata)
            with np.load(npz) as archive:
                raw, q = archive["currents_A"], archive["quantized_currents_A"].copy()
            q[0, 0] += 5. / 32767.
            np.savez(npz, currents_A=raw, quantized_currents_A=q)
            with self.assertRaisesRegex(ValueError, "DAC waveform"):
                commands.load_historical_commands(context, npz_path=npz)

    def test_DAC_idempotence_and_alpha_endpoints(self):
        # This integer is deliberately susceptible to a second floating truncation.
        bad_code = next(k for k in range(1, 19661)
                        if np.trunc((((k / 32767.) * 5.) / 5.) * 32767.) != k)
        with tempfile.TemporaryDirectory(prefix="closed_loop_test_") as folder:
            context = make_context(folder)
            old = np.full((2, 3), bad_code / 32767. * 5.)
            one = commands.evaluate_candidate(context, old, 1.)
            np.testing.assert_array_equal(one["dac_codes"], bad_code)
            np.testing.assert_array_equal(quantize_currents(one["command_currents_A"]), old)
            self.assertLess(one["maximum_CSV_encoding_offset_A"], 1e-14)
            zero = commands.evaluate_candidate(context, old, 0.)
            np.testing.assert_array_equal(zero["quantized_currents_A"], 0.)
            for bad in (-.01, 1.01, np.nan, True):
                with self.assertRaises(ValueError):
                    commands.evaluate_candidate(context, old, bad)

    def test_export_roundtrip_persistent_metadata_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="closed_loop_test_") as folder:
            context = make_context(folder, origin="rf")
            old = quantize_currents(np.array([[.41, -.52, .63], [-.22, .32, -.42]]))
            candidate = commands.evaluate_candidate(context, old, .55)
            destination = Path(folder) / "round_1"
            manifest = commands.export_candidate(context, candidate, destination)
            self.assertTrue(Path(manifest["csv_file"]).is_file())
            zero_time_path = Path(manifest["csv_zero_time_file"])
            self.assertEqual(zero_time_path, destination / "waveform1_zero_time.csv")
            original = np.loadtxt(manifest["csv_file"], delimiter=",")
            zero_time = np.loadtxt(zero_time_path, delimiter=",")
            np.testing.assert_array_equal(zero_time[:, 0], 0.)
            np.testing.assert_array_equal(zero_time[:, 1:], original[:, 1:])
            # Mapping and polarity stay physical-output commands, including
            # preamble/final zero rows. The companion never re-quantizes them.
            expected_outputs = np.zeros((4, 8))
            expected_outputs[:3, [2, 0]] = (candidate["command_currents_A"].T
                                                    * np.array([-1, 1]))
            np.testing.assert_array_equal(zero_time[:, 1:], expected_outputs)
            mapped_back = (zero_time[:3, 1:][:, [2, 0]] * [-1, 1]).T
            np.testing.assert_array_equal(quantize_currents(mapped_back),
                                          candidate["quantized_currents_A"])
            original_lines = Path(manifest["csv_file"]).read_bytes().splitlines()
            companion_lines = zero_time_path.read_bytes().splitlines()
            self.assertEqual(original_lines[0], companion_lines[0])
            for source, copied in zip(original_lines[1:], companion_lines[1:]):
                self.assertEqual(source.split(b",", 1)[1], copied.split(b",", 1)[1])
            report = manifest["zero_time_export"]
            self.assertEqual(report["source_csv_file"], manifest["csv_file"])
            self.assertEqual(report["csv_file"], str(zero_time_path))
            self.assertTrue(report["time_column_all_zero"])
            self.assertEqual(report["source_sha256"], manifest["source_sha256"]["csv"])
            self.assertEqual(report["csv_sha256"], manifest["source_sha256"]["csv_zero_time"])
            self.assertEqual(report["csv_sha256"], commands._sha_file(zero_time_path))
            metadata = json.loads(Path(manifest["metadata_file"]).read_text())
            self.assertEqual(metadata["waveform_export"]["zero_time_export"], report)
            restored = commands.load_historical_commands(context, csv_path=manifest["csv_file"],
                npz_path=manifest["npz_file"], metadata_path=manifest["metadata_file"])
            self.assertEqual(restored["fingerprint"], candidate["fingerprint"])
            with self.assertRaisesRegex(ValueError, "timestamps"):
                commands.load_historical_commands(context, csv_path=zero_time_path,
                    metadata_path=manifest["metadata_file"])
            with self.assertRaises(FileExistsError):
                commands.export_candidate(context, candidate, destination)
            self.assertEqual(zero_time_path.read_bytes().splitlines(), companion_lines)

    def test_rebased_export_restores_leading_zero_states_and_exact_DAC(self):
        with tempfile.TemporaryDirectory(prefix="closed_loop_test_") as folder:
            context = make_context(folder, origin="rf")
            old = quantize_currents(np.array([[0., .4, -.6], [0., -.2, .3]]))
            candidate = commands.evaluate_candidate(context, old, 1.)
            manifest = commands.export_candidate(context, candidate, Path(folder) / "rebased")
            table = np.loadtxt(manifest["csv_file"], delimiter=",")
            np.testing.assert_allclose(table[:, 0], [0., .001, .002], rtol=0, atol=1e-15)
            self.assertTrue(np.any(table[0, 1:] != 0))
            self.assertTrue(manifest["start_at_first_nonzero"])
            self.assertEqual(manifest["leading_zero_states"], 1)
            self.assertEqual(manifest["csv_states"], 2)
            self.assertAlmostEqual(manifest["time_origin_shift_s"], .002)
            self.assertEqual(manifest["states"], 3)
            np.testing.assert_array_equal(manifest["state_edges_s"], context["csv_edges_s"])
            metadata = json.loads(Path(manifest["metadata_file"]).read_text())
            self.assertEqual(metadata["CSV_timing"], context["metadata"]["CSV_timing"])
            for npz in (None, manifest["npz_file"]):
                restored = commands.load_historical_commands(context,
                    csv_path=manifest["csv_file"], npz_path=npz,
                    metadata_path=manifest["metadata_file"])
                np.testing.assert_array_equal(restored["dac_codes"], candidate["dac_codes"])
                np.testing.assert_array_equal(restored["nominal_currents_A"][:, 0], 0.)
                self.assertEqual(restored["fingerprint"], candidate["fingerprint"])
            companion = np.loadtxt(manifest["csv_zero_time_file"], delimiter=",")
            np.testing.assert_array_equal(companion[:, 0], 0.)
            np.testing.assert_array_equal(companion[:, 1:], table[:, 1:])

    def test_all_zero_rebased_export_keeps_original_states(self):
        with tempfile.TemporaryDirectory(prefix="closed_loop_test_") as folder:
            context = make_context(folder, origin="rf")
            candidate = commands.evaluate_candidate(context, quantize_currents(np.full((2, 3), .1)), 0.)
            manifest = commands.export_candidate(context, candidate, Path(folder) / "all_zero")
            self.assertFalse(manifest["first_nonzero_found"])
            self.assertEqual(manifest["leading_zero_states"], 0)
            self.assertEqual(manifest["csv_states"], 3)
            table = np.loadtxt(manifest["csv_file"], delimiter=",")
            np.testing.assert_allclose(table[:, 0], [0., .001, .002, .003], atol=1e-15)
            np.testing.assert_array_equal(table[:, 1:], 0.)
            restored = commands.load_historical_commands(context, csv_path=manifest["csv_file"],
                metadata_path=manifest["metadata_file"])
            np.testing.assert_array_equal(restored["quantized_currents_A"], np.zeros((2, 3)))

    def test_rebasing_metadata_and_first_nonzero_are_verified(self):
        with tempfile.TemporaryDirectory(prefix="closed_loop_test_") as folder:
            context = make_context(folder, origin="rf")
            old = quantize_currents(np.array([[0., .4, -.6], [0., -.2, .3]]))
            candidate = commands.evaluate_candidate(context, old, 1.)
            manifest = commands.export_candidate(context, candidate, Path(folder) / "rebased")
            metadata = json.loads(Path(manifest["metadata_file"]).read_text())
            changes = dict(time_origin_shift_s=.003, leading_zero_states=0, csv_states=3,
                csv_state_edges_s=[0., .001, .003], csv_final_zero_s=.003,
                first_nonzero_found=False, csv_includes_zero_preamble=True,
                start_at_first_nonzero=False)
            for key, value in changes.items():
                mutated = deepcopy(metadata)
                mutated["waveform_export"][key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    commands.load_historical_commands(context, csv_path=manifest["csv_file"], metadata=mutated)
            for key in changes:
                mutated = deepcopy(metadata)
                if key == "start_at_first_nonzero":
                    continue  # Missing flag means the validated original CSV layout.
                del mutated["waveform_export"][key]
                with self.subTest(missing=key), self.assertRaises(ValueError):
                    commands.load_historical_commands(context, csv_path=manifest["csv_file"], metadata=mutated)
            table = np.loadtxt(manifest["csv_file"], delimiter=",")
            table[0, 1:] = 0.
            np.savetxt(manifest["csv_file"], table, delimiter=",", fmt="%.17g",
                header="time_s," + ",".join(f"ch{k}_A" for k in range(8)))
            with self.assertRaisesRegex(ValueError, "first nonzero"):
                commands.load_historical_commands(context, csv_path=manifest["csv_file"], metadata=metadata)

    def test_legacy_export_without_rebasing_metadata_retains_original_timestamps(self):
        with tempfile.TemporaryDirectory(prefix="closed_loop_test_") as folder:
            context = make_context(folder, origin="rf")
            csv, npz, metadata = historical(folder, context)
            for key in ("start_at_first_nonzero", "time_origin_shift_s", "leading_zero_states",
                        "first_nonzero_found", "csv_state_edges_s", "csv_states",
                        "csv_includes_zero_preamble", "csv_final_zero_s"):
                metadata["waveform_export"].pop(key, None)
            before = csv.read_bytes()
            restored = commands.load_historical_commands(context, csv_path=csv, npz_path=npz, metadata=metadata)
            self.assertTrue(restored["report"]["metadata_verified"])
            self.assertEqual(csv.read_bytes(), before)
            table = np.loadtxt(csv, delimiter=",")
            np.testing.assert_allclose(table[:, 0], [0., .001, .002, .003, .004], atol=1e-15)

    def test_field_or_post_DAC_failure_never_exports_even_if_valid_flag_forged(self):
        with tempfile.TemporaryDirectory(prefix="closed_loop_test_") as folder:
            # Deliberate nearly cancelling field: DAC truncation destroys its cancellation.
            raw = np.array([[.10001] * 3, [.10000] * 3])
            context = make_context(folder, field_limit=.05, basis=[[10000., -10001.]])
            old = quantize_currents(raw)
            candidate = commands.evaluate_candidate(context, old, 1.)
            self.assertFalse(candidate["valid"])
            candidate["valid"] = True
            output = Path(folder) / "bad"
            with self.assertRaisesRegex(ValueError, "hardware"):
                commands.export_candidate(context, candidate, output)
            self.assertFalse(output.exists())
            self.assertFalse(list(Path(folder).glob(".closed_loop_export_*")))
            # The nominal historical commands cancel almost exactly, while
            # their two DAC channels round to the same code and no longer do.
            before = commands._diagnostics(context, raw)
            after = commands._diagnostics(context, old)
            self.assertTrue(before["all_limits_met"])
            self.assertFalse(after["checks"]["command_field"])
            csv, npz, _ = historical(folder, context, raw)
            with self.assertRaisesRegex(ValueError, "before or after DAC"):
                commands.load_historical_commands(context, csv_path=csv, npz_path=npz)

    def test_persisted_response_uses_original_physics_and_complete_cycle(self):
        with tempfile.TemporaryDirectory(prefix="closed_loop_test_") as folder:
            context = make_context(folder)
            old = quantize_currents(np.array([[.4, -.5, .6], [-.2, .3, -.4]]))
            waveform = commands.sample_command_waveform(context, old)
            self.assertEqual(waveform["time_s"][-1], .006)
            self.assertEqual(waveform["channel_map"], [3, 1])
            self.assertEqual(waveform["channel_polarity"], [-1, 1])
            np.testing.assert_array_equal(waveform["commanded_A"][:, -1], 0.)
            # Independent constant-command recurrence at each state end.
            actual = np.zeros(2)
            for state in range(3):
                actual = old[:, state] + (actual - old[:, state]) * np.exp(-.001 / 50e-6)
                t = context["line_end_s"][state]
                sample = commands.sample_command_waveform(context, old, [t])
                np.testing.assert_allclose(sample["actual_A"][:, 0], actual, rtol=1e-13, atol=1e-14)
            with self.assertRaises(ValueError):
                commands.sample_command_waveform(context, old, [-.001])

    def test_original_export_error_is_transactional(self):
        with tempfile.TemporaryDirectory(prefix="closed_loop_test_") as folder:
            context = make_context(folder)
            old = quantize_currents(np.full((2, 3), .1))
            candidate = commands.evaluate_candidate(context, old, .5)
            output = Path(folder) / "failed"
            with patch.object(commands, "_read_csv", side_effect=ValueError("roundtrip failed")):
                with self.assertRaisesRegex(ValueError, "roundtrip failed"):
                    commands.export_candidate(context, candidate, output)
            self.assertFalse(output.exists())
            self.assertFalse(list(Path(folder).glob(".closed_loop_export_*")))

    def test_zero_time_companion_error_is_transactional(self):
        with tempfile.TemporaryDirectory(prefix="closed_loop_test_") as folder:
            context = make_context(folder)
            old = quantize_currents(np.full((2, 3), .1))
            candidate = commands.evaluate_candidate(context, old, .5)
            output = Path(folder) / "failed"
            with patch.object(commands, "export_zero_time_copy", side_effect=OSError("copy failed")):
                with self.assertRaisesRegex(OSError, "copy failed"):
                    commands.export_candidate(context, candidate, output)
            self.assertFalse(output.exists())
            self.assertFalse(list(Path(folder).glob(".closed_loop_export_*")))


if __name__ == "__main__":
    unittest.main()
