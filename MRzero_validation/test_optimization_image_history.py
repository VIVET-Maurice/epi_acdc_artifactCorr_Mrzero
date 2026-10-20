"""Periodic snapshots, independent image storage and portable archive checks."""

from base64 import b64decode
from io import BytesIO
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np
from PIL import Image

from optimization_image_history import OptimizationImageHistory


class ImageHistoryTests(unittest.TestCase):
    def test_periodic_multiples_only_and_terminal_off_multiple_ignored(self):
        history = OptimizationImageHistory(np.ones((3, 4)), np.zeros((3, 4)), every=20)
        for iteration in (0, 20, 40):
            self.assertTrue(history.record(dict(iteration=iteration, image=np.ones((3, 4)) * iteration,
                                               currents_A=object())))
        # No image access is needed for a terminal callback outside the cadence.
        self.assertFalse(history.record(dict(iteration=47)))
        self.assertFalse(history.record(dict(iteration=40, image=np.zeros((3, 4)))))
        arrays = history.arrays()
        np.testing.assert_array_equal(arrays["iterations"], [0, 20, 40])
        self.assertEqual(arrays["images"].dtype, np.float32)
        self.assertEqual(arrays["images"].shape, (3, 3, 4))
        self.assertFalse(any("current" in key for key in arrays))

    def test_complex_magnitude_is_copied_and_no_callback_arrays_are_retained(self):
        history = OptimizationImageHistory(np.ones((2, 2)), np.zeros((2, 2)), every=1)
        source = np.full((2, 2), 3 + 4j)
        history.record(dict(iteration=0, image=source, currents_A=np.full((32, 64), 9.)))
        source[:] = 0
        np.testing.assert_array_equal(history.arrays()["images"], np.full((1, 2, 2), 5., np.float32))
        exported = history.arrays()
        exported["images"][:] = 99
        np.testing.assert_array_equal(history.arrays()["images"], 5.)
        self.assertFalse(any("current" in key for key in vars(history)))

    def test_archive_roundtrip_preserves_true_iterations_and_float32_images(self):
        history = OptimizationImageHistory(np.ones((2, 3)) * 2, np.ones((2, 3)) * 3, every=7)
        history.record(dict(iteration=0, image=np.arange(6).reshape(2, 3) / 7))
        history.record(dict(iteration=7, image=np.arange(6).reshape(2, 3) / 9))
        with tempfile.TemporaryDirectory() as folder:
            path = history.save_npz(Path(folder) / "nested" / "images.npz")
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), set(history.arrays()))
                for key, value in history.arrays().items():
                    np.testing.assert_array_equal(archive[key], value)

    def test_offline_html_has_embedded_images_true_labels_and_common_lower_origin_scale(self):
        reference = np.array([[0., 0.], [1., 1.]])
        history = OptimizationImageHistory(reference, np.zeros((2, 2)), every=20,
                                           title='Test <sans injection>', origin="lower")
        history.record(dict(iteration=0, image=reference))
        history.record(dict(iteration=20, image=reference * 2))
        html = history.html(standalone=True)
        self.assertTrue(html.startswith("<!doctype html>"))
        self.assertIn('type="range"', html)
        self.assertIn('Itération 20', html)
        self.assertIn('Itérations archivées : 0, 20', html)
        self.assertIn('Test &lt;sans injection&gt;', html)
        self.assertNotIn('https://', html)
        self.assertNotIn('<script src=', html)
        encoded = re.search(r'data:image/png;base64,([^"\s]+)', html).group(1)
        pixels = np.asarray(Image.open(BytesIO(b64decode(encoded))))
        # Native first row is at the bottom, matching imshow(origin='lower').
        np.testing.assert_array_equal(pixels, [[255, 255], [0, 0]])
        with tempfile.TemporaryDirectory() as folder:
            path = history.write_html(Path(folder) / "viewer.html")
            self.assertEqual(path.read_text(encoding="utf-8"), html)

    def test_default_upper_origin_matches_live_current_without_vertical_flip(self):
        reference = np.array([[0., 0.], [1., 1.]])
        history = OptimizationImageHistory(reference, reference)
        html = history.html()
        encoded = re.search(r'data:image/png;base64,([^"\s]+)', html).group(1)
        pixels = np.asarray(Image.open(BytesIO(b64decode(encoded))))
        np.testing.assert_array_equal(pixels, [[0, 0], [255, 255]])
        self.assertIn("Origine des images en haut", html)
        self.assertEqual(history.arrays()["origin"].item(), "upper")
        with self.assertRaises(ValueError):
            OptimizationImageHistory(reference, reference, origin="center")

    def test_empty_history_remains_displayable_and_invalid_snapshots_are_rejected(self):
        history = OptimizationImageHistory(np.zeros((2, 3)), np.zeros((2, 3)))
        self.assertEqual(history.arrays()["images"].shape, (0, 2, 3))
        self.assertIn("Aucune image archivée", history.html())
        self.assertIn(' disabled', history.html())
        with self.assertRaises(ValueError):
            history.record(dict(iteration=True, image=np.zeros((2, 3))))
        with self.assertRaises(ValueError):
            history.record(dict(iteration=0, image=np.zeros((3, 3))))
        history.record(dict(iteration=20, image=np.zeros((2, 3))))
        with self.assertRaises(ValueError):
            history.record(dict(iteration=0, image=np.zeros((2, 3))))


if __name__ == "__main__":
    unittest.main()
