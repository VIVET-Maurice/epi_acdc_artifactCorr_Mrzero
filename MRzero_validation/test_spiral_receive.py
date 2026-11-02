"""Measured complex echo phase transfer, channel IDs and fixed object weights."""

from copy import deepcopy
import unittest

import numpy as np

from spiral_receive import prepare_spiral_receive_model


def fixture():
    shape, channels = (3, 4), 3
    rng = np.random.default_rng(753)
    initial = rng.normal(size=(channels,)+shape) + 1j*rng.normal(size=(channels,)+shape)
    field = np.linspace(-120., 150., 12).reshape(shape)
    anchor = .003
    echo = initial * np.exp(-2j*np.pi*field*anchor)[None]
    index = np.array([1, 3, 7, 9])
    mask = np.zeros(shape, bool); mask.ravel()[index] = True
    prepared = dict(parent_valid_mask=mask, parent_flat_index=index,
                    expanded_experiment=dict(m0_proxy=np.ones((4,2))), report={})
    fit = dict(amplitude_reference_time_s=anchor,
        multi_echo=dict(complex_receiver_amplitude=echo, refined_b0_Hz=field,
                        amplitude_reference_time_s=anchor))
    inputs = dict(acquisition=dict(sequence_kind="spiral"),
        scan_metadata=dict(receiver_count=channels, receiver_channel_ids=[8,1,3]),
        b0_raw=dict(metadata=dict(receiver_channel_ids=[1,3,8])))
    return prepared, fit, inputs, initial


class SpiralReceiveTests(unittest.TestCase):
    def test_restores_excitation_phase_and_channel_order_without_changing_object(self):
        prepared, fit, inputs, initial = fixture()
        original = deepcopy(prepared)
        result = prepare_spiral_receive_model(prepared, fit, inputs)
        index = prepared["parent_flat_index"]
        expected = initial[[2,0,1]].reshape(3,-1)[:,index]
        expected /= np.sqrt(np.sum(abs(expected)**2, axis=0))[None]
        np.testing.assert_allclose(result["expanded_experiment"]["receive_weights"],
            np.repeat(expected[...,None],2,axis=2), atol=2e-15)
        for key in ("parent_flat_index", "parent_valid_mask"):
            np.testing.assert_array_equal(result[key], original[key])
        np.testing.assert_array_equal(result["expanded_experiment"]["m0_proxy"], original["expanded_experiment"]["m0_proxy"])
        self.assertNotIn("receive_weights", prepared["expanded_experiment"])
        self.assertTrue(result["report"]["receive_model"]["channel_ID_order_verified"])

    def test_frozen_receive_phase_is_independent_of_spiral_fitted_frequency_offset(self):
        prepared, fit, inputs, _ = fixture()
        before = prepare_spiral_receive_model(prepared, fit, inputs)
        prepared["expanded_experiment"]["field"] = dict(b0Hz=np.full((4,2), 70.))
        after = prepare_spiral_receive_model(prepared, fit, inputs)
        np.testing.assert_array_equal(before["expanded_experiment"]["receive_weights"], after["expanded_experiment"]["receive_weights"])

    def test_missing_complex_data_falls_back_only_when_explicitly_allowed(self):
        prepared, fit, inputs, _ = fixture(); fit.pop("multi_echo")
        self.assertEqual(prepare_spiral_receive_model(prepared,fit,inputs)["report"]["receive_model"]["mode"], "rss_proxy")
        with self.assertRaises(ValueError):prepare_spiral_receive_model(prepared,fit,inputs,mode="complex_echo")

    def test_invalid_receiver_encoding_is_rejected(self):
        for reason in ("channels", "zero", "nan", "anchor"):
            prepared, fit, inputs, _ = fixture()
            if reason == "channels":inputs["scan_metadata"]["receiver_channel_ids"]=[8,1,4]
            elif reason == "zero":fit["multi_echo"]["complex_receiver_amplitude"][:,0,1]=0
            elif reason == "nan":fit["multi_echo"]["refined_b0_Hz"][0,1]=np.nan
            else:fit["multi_echo"]["amplitude_reference_time_s"] += .001
            with self.subTest(reason=reason),self.assertRaises(ValueError):prepare_spiral_receive_model(prepared,fit,inputs)


if __name__ == "__main__":unittest.main()
