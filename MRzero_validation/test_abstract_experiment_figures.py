"""Scientific safeguards for archived references and publication panels."""

import json
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from abstract_experiment_figures import (
    _archived_reference, compare_registered_images, plot_epi_comparison,
    plot_metrics_and_profiles, sha256,
)


def record(image, label):
    y, x = np.asarray(image).shape
    X, Y = np.meshgrid((np.arange(x) - x // 2) * .001,
                       (np.arange(y) - y // 2) * .002)
    return dict(image=np.asarray(image), X_m=X, Y_m=Y, label=label)


def bundle(reference, baseline, corrected):
    result = compare_registered_images(record(reference, "GRE reference"),
        record(baseline, "Without AC/DC"), [record(corrected, "Image-domain")])
    result.update(study={"title": "Synthetic acquisition"},
                  acquisition_ids=["MID00001", "MID00002"])
    return result


class AbstractExperimentFigureTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_archive_requires_exact_input_hashes_and_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sequences").mkdir()
            study = dict(folder=root, baseline="baseline.dat", b0_before="b0.dat",
                epi_sequence="epi.seq", b0_sequence="b0.seq", reference_sequence="gre.seq",
                reference="reference.dat", reference_archive=root / "archive.npz")
            paths = dict(scan=root / "baseline.dat", b0=root / "b0.dat",
                scan_sequence=root / "sequences/epi.seq", b0_sequence=root / "sequences/b0.seq",
                reference_sequence=root / "sequences/gre.seq")
            for key, path in paths.items():
                path.write_bytes(key.encode())
            ref = record(np.ones((3, 4)), "GRE")
            np.savez(study["reference_archive"], measured_GRE_native_magnitude=ref["image"],
                measured_GRE_native_X_m=ref["X_m"], measured_GRE_native_Y_m=ref["Y_m"])
            report = dict(source_sha256={key: sha256(path) for key, path in paths.items()},
                input_selection={"reference": "reference.dat"},
                reference_sequence_validation={"validated": True},
                reference_geometry_checks=["reference GRE/EPI: " + key for key in
                    ("slice_position_mm", "slice_thickness_mm", "slice_quaternion_wxyz")],
                baseline_metadata_audit={"frequency_reference": {"acquisitions": {"GRE": {}}}})
            report["source_sha256"]["reference"] = "original GRE digest"
            report_path = root / "archive.json"
            report_path.write_text(json.dumps(report))
            actual, provenance = _archived_reference(study)
            np.testing.assert_array_equal(actual["image"], ref["image"])
            self.assertFalse(provenance["raw_reference_available"])
            for key, path in paths.items():
                with self.subTest(changed_source=key):
                    original = path.read_bytes()
                    path.write_bytes(original + b" changed")
                    with self.assertRaisesRegex(ValueError, key + " provenance mismatch"):
                        _archived_reference(study)
                    path.write_bytes(original)
                    expected = report["source_sha256"].pop(key)
                    report_path.write_text(json.dumps(report))
                    with self.assertRaisesRegex(ValueError, key + " provenance mismatch"):
                        _archived_reference(study)
                    report["source_sha256"][key] = expected
                    report_path.write_text(json.dumps(report))
            report["reference_sequence_validation"]["validated"] = False
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "validated geometry/sequence"):
                _archived_reference(study)
            report["reference_sequence_validation"]["validated"] = True
            report["reference_geometry_checks"].remove("reference GRE/EPI: slice_thickness_mm")
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "validated geometry/sequence"):
                _archived_reference(study)

    def test_fixed_gain_and_roi_exclude_background_from_metrics_only(self):
        reference = np.zeros((7, 9))
        reference[2:5, 3:6] = 2.
        baseline = reference * .5
        corrected = reference * .25
        corrected[0, 0] = 50.
        result = bundle(reference, baseline, corrected)
        self.assertEqual(result["gain"], 2.)
        self.assertEqual(result["mask"].sum(), 9)
        self.assertAlmostEqual(result["rows"][0]["NRMSE_percent"], 0.)
        self.assertAlmostEqual(result["rows"][1]["NRMSE_percent"], 50.)
        self.assertEqual(result["images"][2][0, 0], 50.)

    def test_epi_panels_keep_full_fov_and_one_fixed_color_scale(self):
        reference = np.ones((7, 9))
        result = bundle(reference, reference * .5, reference * 1.5)
        fig = plot_epi_comparison(result, vmax=1.2, show_contour=False)
        for ax, expected in zip(fig.axes[:3], result["images"]):
            mesh = ax.collections[0]
            self.assertEqual(mesh.get_clim(), (0., 1.2))
            np.testing.assert_array_equal(np.asarray(mesh.get_array()).reshape(expected.shape), expected)
            np.testing.assert_allclose(ax.get_xlim(), [-4.5, 4.5])
            np.testing.assert_allclose(ax.get_ylim(), [-7., 7.])
        self.assertEqual(result["images"][2].max(), 3.)

    def test_physical_profiles_preserve_displaced_peaks_and_metric_values(self):
        reference = np.ones((7, 9))
        reference[3, 4] = 3.
        corrected = np.ones_like(reference) * .5
        corrected[3, 6] = corrected[5, 4] = 1.5
        result = bundle(reference, reference * .5, corrected)
        fig = plot_metrics_and_profiles([result])
        metric_ax, readout_ax, phase_ax = fig.axes
        np.testing.assert_allclose([p.get_height() for p in metric_ax.patches],
                                  [r["NRMSE_percent"] for r in result["rows"]])
        readout_line, phase_line = readout_ax.lines[2], phase_ax.lines[2]
        np.testing.assert_array_equal(readout_line.get_ydata(), result["images"][2][3])
        np.testing.assert_array_equal(phase_line.get_ydata(), result["images"][2][:, 4])
        self.assertEqual(readout_line.get_xdata()[np.argmax(readout_line.get_ydata())], 2.)
        self.assertEqual(phase_line.get_xdata()[np.argmax(phase_line.get_ydata())], 4.)


if __name__ == "__main__":
    unittest.main()
