"""Independent complex 3D Fourier sums, readout convergence, and derivatives."""

import unittest
import warnings

import numpy as np
import torch

from volume_epi_model import VolumeEPIModel


def fixture(shape=(4, 3, 2), high_field=False):
    nx, ny, nz = shape
    i, j, k = np.indices(shape)
    fov = np.array([.20, .18, .12])
    volume = dict(amplitude=.5 + .09 * i + .04 * j + .07 * k,
                  b0_Hz=(40 * i - 15 * j + 25 * k) * (9 if high_field else 1.),
                  fieldmaps_Hz_per_A=np.stack((20 + 5 * i - 7 * j + 9 * k,
                                               -8 + 3 * i + 11 * j - 13 * k), axis=-1),
                  FOV_m=fov, R2_sInv=3 + i + .2 * j, amplitude_reference_time_s=.0007)
    kx = np.broadcast_to((np.arange(nx) - nx // 2) / fov[0], (ny, nx)).copy()
    kx += np.linspace(-.17, .11, ny)[:, None]  # Preserve negative/positive actual offsets.
    ky = np.broadcast_to(((np.arange(ny) - ny // 2) / fov[1])[:, None], (ny, nx)).copy()
    starts = .001 + np.arange(ny) * .002
    times = starts[:, None] + .0002 + np.linspace(0, .0007, nx)[None, :]
    times[1::2] = times[1::2, ::-1]  # Arrays are sorted by kx, not by time.
    acquisition = dict(kx_mInv=kx, ky_mInv=ky, time_s=times,
        kz_partition_mInv=(np.arange(nz) - nz // 2) / fov[2] - .09,
        line_start_s=starts[:, None], line_end_s=(starts + .0015)[:, None],
        rf_time_s=np.zeros_like(times), FOV_m=fov, TR_s=float(starts[-1] + .003),
        phase_offset_rad=.03 * np.sin(np.arange(ny * nx).reshape(ny, nx)))
    currents = np.stack((np.linspace(.17, -.12, ny), np.linspace(-.09, .23, ny)))
    return volume, acquisition, currents


def explicit_fourier(volume, acquisition, currents, tau=None):
    """Direct independent point-spin DFT at EACH actual 3D ADC sample."""
    shape = volume["amplitude"].shape
    axes = [(np.arange(n) - n // 2) * f / n for n, f in zip(shape, volume["FOV_m"])]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    starts = acquisition["line_start_s"].reshape(-1)
    ends = acquisition["line_end_s"].reshape(-1)
    result = np.zeros(shape, complex)
    def primitive(age, time_constant):
        age = max(0., age)
        return age if time_constant is None else age - time_constant * (1 - np.exp(-age / time_constant))
    for row in range(shape[1]):
        for column in range(shape[0]):
            t = acquisition["time_s"][row, column]
            integrals = np.array([sum(currents[c, state] *
                (primitive(t - start, None if tau is None else tau[c])
                 - primitive(t - end, None if tau is None else tau[c]))
                for state, (start, end) in enumerate(zip(starts, ends)))
                for c in range(currents.shape[0])])
            phase = (volume["b0_Hz"] * t
                     + np.sum(volume["fieldmaps_Hz_per_A"] * integrals, axis=-1)
                     + acquisition["kx_mInv"][row, column] * x
                     + acquisition["ky_mInv"][row, column] * y)
            amplitude = volume["amplitude"] * np.exp(-volume.get("R2_sInv", 0.) *
                (t - volume.get("amplitude_reference_time_s", 0.)))
            for partition, kz in enumerate(acquisition["kz_partition_mInv"]):
                result[column, row, partition] = np.sum(amplitude * np.exp(-2j * np.pi * (phase + kz * z)))
                result[column, row, partition] *= np.exp(1j * acquisition["phase_offset_rad"][row, column])
    return result


class FullVolumeSignalTests(unittest.TestCase):
    def model(self, volume, acquisition, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return VolumeEPIModel(volume, acquisition, **kwargs)

    def test_exact_readout_matches_full_complex_xyz_sum_with_filtered_currents(self):
        volume, acquisition, currents = fixture()
        tau = np.array([.00018, .00031])
        model = self.model(volume, acquisition, interpolation_nodes="exact",
                           current_response=dict(enabled=True, tau_s=tau))
        image, kdata = model.forward(currents)
        expected = explicit_fourier(volume, acquisition, currents, tau)
        np.testing.assert_allclose(model.numpy(kdata), expected, rtol=2e-12, atol=2e-12)
        expected_image = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(expected)))
        np.testing.assert_allclose(model.numpy(image), expected_image, rtol=2e-12, atol=2e-12)
        # Different z values and nonzero partition encodings are actually used.
        self.assertGreater(np.linalg.norm(expected[:, :, 0] - expected[:, :, 1]), 1.)

    def test_instantaneous_response_matches_independent_integral(self):
        volume, acquisition, currents = fixture()
        model = self.model(volume, acquisition, interpolation_nodes="exact", current_response={"enabled": False})
        np.testing.assert_allclose(model.numpy(model.forward(currents)[1]),
                                   explicit_fourier(volume, acquisition, currents), rtol=2e-12, atol=2e-12)

    def test_zero_field_full_3d_fft_recovers_point_amplitudes(self):
        volume, acquisition, currents = fixture((5, 4, 3))
        volume["b0_Hz"][:] = 0
        volume["R2_sInv"] = 0.
        for axis, n, fov in zip(("kx_mInv", "ky_mInv"), volume["amplitude"].shape[:2], volume["FOV_m"][:2]):
            vector = (np.arange(n) - n // 2) / fov
            acquisition[axis] = np.broadcast_to(vector[None, :] if axis == "kx_mInv" else vector[:, None], (4, 5)).copy()
        acquisition["kz_partition_mInv"] = (np.arange(3) - 1) / volume["FOV_m"][2]
        acquisition["phase_offset_rad"][:] = 0
        model = self.model(volume, acquisition, interpolation_nodes=3)
        image, _ = model.forward(np.zeros_like(currents))
        np.testing.assert_allclose(model.numpy(image), volume["amplitude"], rtol=2e-12, atol=2e-12)

    def test_chebyshev_readout_converges_on_large_b0_and_keeps_sample_times(self):
        volume, acquisition, currents = fixture((24, 3, 3), high_field=True)
        model = self.model(volume, acquisition, interpolation_nodes=3,
                           current_response=dict(enabled=True, tau_s=[.00018, .00031]))
        coarse = model.convergence_check(currents, refined_nodes="exact")
        self.assertGreater(coarse["relative_complex_kdata_error"], .05)
        result = model.adapt_interpolation(currents, tolerance=1e-7)
        self.assertTrue(result["passed"])
        final = model.convergence_check(currents, refined_nodes="exact")
        self.assertLess(final["relative_complex_kdata_error"], 1e-7)
        np.testing.assert_allclose(model.numpy(model.forward(currents)[1]),
            explicit_fourier(volume, acquisition, currents, np.array([.00018, .00031])), rtol=1e-9, atol=1e-9)

    def test_a_line_center_model_does_not_reproduce_readout_off_resonance(self):
        volume, acquisition, currents = fixture((8, 3, 3), high_field=True)
        model = self.model(volume, acquisition, interpolation_nodes="exact")
        actual = model.numpy(model.forward(currents)[1])
        collapsed = {**acquisition, "time_s": np.repeat(acquisition["time_s"].mean(1)[:, None], 8, axis=1)}
        center_only = explicit_fourier(volume, collapsed, currents,
                                      np.full(2, 133e-6 / np.log(100.)))
        self.assertGreater(np.linalg.norm(actual-center_only) / np.linalg.norm(actual), .1)


class DifferentiationAndContractTests(unittest.TestCase):
    def test_gradient_matches_finite_differences_and_checkpoint_does_not_change_it(self):
        volume, acquisition, currents = fixture((6, 3, 3))
        kwargs = dict(interpolation_nodes="exact", current_response=dict(enabled=True, tau_s=[.00018, .00031]))
        model = VolumeEPIModel(volume, acquisition, checkpoint_lines=True, **kwargs)
        reference = volume["amplitude"] * (1 + .07 * np.indices(volume["amplitude"].shape)[2])
        model.set_target(reference, gain=.9)
        value, gradient, image, kdata = model.objective_and_grad(currents)
        self.assertTrue(np.isfinite(value))
        step = 1e-5
        for index in np.ndindex(currents.shape):
            plus, minus = currents.copy(), currents.copy()
            plus[index] += step
            minus[index] -= step
            with torch.no_grad():
                finite = (float(model.torch_objective(plus)) - float(model.torch_objective(minus))) / (2*step)
            self.assertAlmostEqual(gradient[index], finite, delta=max(1e-8, abs(finite) * 1e-5))
        plain = VolumeEPIModel(volume, acquisition, checkpoint_lines=False, **kwargs)
        plain.set_target(reference, gain=.9)
        plain_value, plain_gradient, plain_image, plain_kdata = plain.objective_and_grad(currents)
        self.assertEqual(value, plain_value)
        np.testing.assert_allclose(gradient, plain_gradient, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(image, plain_image)
        np.testing.assert_array_equal(kdata, plain_kdata)

    def test_native_torch_optimizer_and_hardware_adapter(self):
        from acdc_hardware import HardwareConstraints
        volume, acquisition, currents = fixture()
        model = VolumeEPIModel(volume, acquisition, interpolation_nodes="exact", max_current_A=[.5, 0.])
        model.set_target(volume["amplitude"])
        commands = model.tensor(currents).requires_grad_()
        loss = model.torch_objective(commands)
        loss.backward()
        self.assertEqual(commands.grad.shape, commands.shape)
        hardware = HardwareConstraints(model, {"cycle_duration_s": acquisition["TR_s"]})
        zero = np.zeros_like(currents)
        report = hardware.diagnostics(zero, 1., 1000.)
        self.assertTrue(report["all_limits_met"])
        np.testing.assert_array_equal(model.numpy(model.max_current_A), [.5, 0.])

    def test_missing_target_and_invalid_geometry_are_rejected(self):
        volume, acquisition, currents = fixture()
        model = VolumeEPIModel(volume, acquisition, interpolation_nodes="exact")
        with self.assertRaises(RuntimeError):
            model.torch_objective(currents)
        bad = {**acquisition, "kx_mInv": acquisition["kx_mInv"].copy()}
        bad["kx_mInv"][1, 2] += .1
        with self.assertRaises(ValueError):
            VolumeEPIModel(volume, bad)
        bad = {**acquisition, "rf_time_s": np.ones_like(acquisition["time_s"])}
        with self.assertRaises(ValueError):
            VolumeEPIModel(volume, bad)


if __name__ == "__main__":
    unittest.main()
