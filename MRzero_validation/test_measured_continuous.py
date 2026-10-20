"""Receiver-axis, echo-order and measured-phase reconstruction checks."""

import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from measured_continuous import reconstruct_receivers, reconstruct_b0_echoes


def centered_fft(images):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(images, axes=(-2, -1))), axes=(-2, -1))


def acquisition(shape=(4, 6)):
    ny, nx = shape
    fov = np.asarray([.21, .17])
    kx, ky = np.meshgrid((np.arange(nx) - nx // 2) / fov[0],
                         (np.arange(ny) - ny // 2) / fov[1])
    return {"kx_mInv": kx, "ky_mInv": ky, "FOV_m": fov}


def direct_reconstruction(samples, acq):
    ny, nx = samples.shape[-2:]
    x = (np.arange(nx) - nx // 2) * acq["FOV_m"][0] / nx
    y = (np.arange(ny) - ny // 2) * acq["FOV_m"][1] / ny
    X, Y = np.meshgrid(x, y)
    cycles = (acq["kx_mInv"].reshape(-1, 1) * X.reshape(1, -1)
              + acq["ky_mInv"].reshape(-1, 1) * Y.reshape(1, -1))
    kernel = np.exp(2j * np.pi * cycles) / (ny * nx)
    return (samples.reshape(-1, ny * nx) @ kernel).reshape(samples.shape)


class MeasuredContinuousTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(19)

    def complex_values(self, shape):
        return self.rng.normal(size=shape) + 1j * self.rng.normal(size=shape)

    def test_receiver_order_rss_and_native_diagnostics_preserved(self):
        acq = acquisition()
        raw_images = self.complex_values((3, 4, 6))
        corrected_images = raw_images * np.exp(1j * self.rng.normal(size=(3, 4, 6)))
        raw = np.moveaxis(centered_fft(raw_images), 0, 1)
        corrected = np.moveaxis(centered_fft(corrected_images), 0, 1)
        old = np.moveaxis(raw_images, 0, 1)
        diagnostics = {"kspace": raw, "corrected_kspace": corrected,
                       "coil_images": old.copy(), "uncorrected_coil_images": old.copy(),
                       "uncorrected_image": np.sqrt(np.sum(np.abs(old)**2, axis=1))}
        image, report = reconstruct_receivers(diagnostics, acq)
        np.testing.assert_allclose(diagnostics["coil_images"], np.moveaxis(corrected_images, 0, 1), atol=2e-14)
        np.testing.assert_allclose(diagnostics["uncorrected_coil_images"], old, atol=2e-14)
        np.testing.assert_array_equal(diagnostics["native_coil_images"], old)
        np.testing.assert_allclose(image, np.sqrt(np.sum(np.abs(corrected_images)**2, axis=0)), atol=2e-14)
        self.assertFalse(report["receiver_phase_reapplied"])
        self.assertEqual(report["receiver_count"], 3)

    def test_receiver_off_grid_matches_direct_sum_without_phase_reapplication(self):
        acq = acquisition()
        acq["kx_mInv"] += np.arange(4)[:, None] * .03
        acq["phase_offset_rad"] = np.full((4, 6), 1.7)
        samples = self.complex_values((2, 4, 6))
        diagnostics = {"kspace": np.moveaxis(samples, 0, 1)}
        image, report = reconstruct_receivers(diagnostics, acq)
        expected = direct_reconstruction(samples, acq)
        np.testing.assert_allclose(diagnostics["coil_images"], np.moveaxis(expected, 0, 1), atol=2e-14)
        np.testing.assert_allclose(image, np.sqrt(np.sum(np.abs(expected)**2, axis=0)), atol=2e-14)
        self.assertEqual(report["method"], "row_separable")

    def b0_fixture(self, path, polarity=-1):
        ny, nx, ne, receivers = 4, 6, 3, 2
        acq = acquisition((ny, nx))
        acq["kx_mInv"] += np.arange(ny)[:, None] * .02
        phase_order = np.asarray([2, 0, 3, 1])
        trajectory = np.zeros((3, ny, ne, nx))
        for echo in range(ne):
            trajectory[0, :, echo] = acq["kx_mInv"][phase_order]
            trajectory[1, :, echo] = acq["ky_mInv"][phase_order]
        if polarity < 0:
            trajectory = trajectory[..., ::-1]
        info = {"image_shape": [ny, nx], "echo_count": ne, "polarity": polarity,
                "FOV_m": acq["FOV_m"].tolist(), "acquired_phase_rows": phase_order.tolist(),
                "sequence_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        samples = self.complex_values((ne, receivers, ny, nx))
        old = np.full(samples.shape, 2 + 3j)
        raw = {"kspace": samples, "coil_images": old.copy(), "metadata": {"sequence_info": info}}
        sequence = SimpleNamespace(read=Mock(), calculate_kspace=Mock(
            return_value=(trajectory.reshape(3, -1), None, None, None, None)))
        return raw, acq, old, sequence

    def test_b0_echoes_use_actual_k_and_restore_negative_readout_phase_order(self):
        for polarity in [-1, 1]:
            with self.subTest(polarity=polarity), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "b0.seq"
                path.write_text("validated synthetic sequence", encoding="utf-8")
                raw, acq, old, sequence = self.b0_fixture(path, polarity)
                expected = direct_reconstruction(raw["kspace"], acq)
                factory = Mock(return_value=sequence)
                with patch.dict("sys.modules", {"pypulseq": SimpleNamespace(Sequence=factory)}):
                    reports = reconstruct_b0_echoes(raw, path)
                np.testing.assert_allclose(raw["coil_images"], expected, atol=3e-14)
                np.testing.assert_array_equal(raw["native_coil_images"], old)
                sequence.read.assert_called_once_with(str(path))
                sequence.calculate_kspace.assert_called_once_with()
                self.assertEqual(len(reports), 3)
                self.assertTrue(all(r["method"] == "row_separable" for r in reports))
                self.assertTrue(all(not r["receiver_phase_reapplied"] for r in reports))

    def test_b0_preserves_known_echo_phase_and_receive_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "b0.seq"
            path.write_text("validated synthetic sequence", encoding="utf-8")
            raw, acq, _, sequence = self.b0_fixture(path)
            base = self.complex_values((2, 4, 6))
            te = np.asarray([.002, .004, .006])
            field_hz = 73.0
            raw["kspace"] = base[None] * np.exp(-2j * np.pi * field_hz * te[:, None, None, None])
            with patch.dict("sys.modules", {"pypulseq": SimpleNamespace(Sequence=lambda: sequence)}):
                reconstruct_b0_echoes(raw, path)
            products = np.sum(raw["coil_images"][1:] * raw["coil_images"][:-1].conj(), axis=1)
            estimated = -np.angle(products) / (2 * np.pi * np.diff(te)[:, None, None])
            np.testing.assert_allclose(estimated, field_hz, atol=2e-12)

    def test_missing_raw_data_and_unvalidated_b0_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "raw kspace"):
            reconstruct_receivers({}, acquisition())
        with self.assertRaisesRegex(ValueError, "sequence_info"):
            reconstruct_b0_echoes({}, "unused.seq")


if __name__ == "__main__":
    unittest.main()
