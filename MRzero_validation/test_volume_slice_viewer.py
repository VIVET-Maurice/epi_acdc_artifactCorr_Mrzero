"""World-axis slice orientation and explicit-selection lifecycle, without widgets."""

import importlib
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from volume_slice_viewer import VolumeSliceViewer, extract_volume_slice


def make_bundle(shape=(3, 4, 5)):
    x, y, z = np.indices(shape)
    reference = 100 * x + 10 * y + z
    return dict(reference_magnitude=reference.astype(float),
                epi_magnitude=reference.astype(float) + 1000,
                b0_Hz=reference.astype(float) - 150,
                axes_m=tuple(np.arange(size, dtype=float) * spacing + origin
                             for size, spacing, origin in zip(shape, (.001, .002, .003), (.01, -.02, .03))))


class ExtractionTests(unittest.TestCase):
    def test_axis_landmarks_and_ascending_display_coordinates(self):
        bundle = make_bundle()
        # Codes identify all three world indices independently; a transpose or
        # mirror error cannot pass these landmark/coordinate checks.
        for axis, index, shape, row, column, first, last in (
                ("x", 1, (5, 4), "z", "y", 100, 134),
                ("y", 2, (5, 3), "z", "x", 20, 224),
                ("z", 3, (4, 3), "y", "x", 3, 233)):
            with self.subTest(axis=axis):
                plane = extract_volume_slice(bundle, axis, index)
                self.assertEqual(plane["reference_magnitude"].shape, shape)
                self.assertEqual(plane["reference_magnitude"][0, 0], first)
                self.assertEqual(plane["reference_magnitude"][-1, -1], last)
                np.testing.assert_array_equal(plane["epi_magnitude"] - plane["reference_magnitude"], 1000)
                np.testing.assert_array_equal(plane["b0_Hz"] - plane["reference_magnitude"], -150)
                self.assertEqual(plane["orientation"]["row_axis"], row)
                self.assertEqual(plane["orientation"]["column_axis"], column)
                self.assertEqual(plane["orientation"]["display_origin"], "lower")
                self.assertTrue(np.all(np.diff(plane["row_coordinates_m"]) > 0))
                self.assertTrue(np.all(np.diff(plane["column_coordinates_m"]) > 0))
                expected_coord = bundle["axes_m"][("x", "y", "z").index(axis)][index]
                self.assertEqual(plane["coordinate_m"], expected_coord)

    def test_nonuniform_physical_pixel_edges_and_extent(self):
        bundle = make_bundle()
        bundle["axes_m"] = (np.array([.01, .013, .019]), bundle["axes_m"][1], bundle["axes_m"][2])
        plane = extract_volume_slice(bundle, "z", 0)
        np.testing.assert_allclose(plane["column_edges_m"], [.0085, .0115, .016, .022])
        np.testing.assert_allclose(plane["extent_m"], [.0085, .022, -.021, -.013])

    def test_b0_missing_coverage_preserved_and_input_unchanged(self):
        bundle = make_bundle()
        bundle["b0_valid"] = np.ones((3, 4, 5), bool)
        bundle["b0_valid"][1, 2, 3] = False
        bundle["b0_Hz"][2, 2, 3] = np.nan
        before = bundle["b0_Hz"].copy()
        plane = extract_volume_slice(bundle, "z", 3)
        self.assertFalse(plane["b0_valid"][2, 1])
        self.assertFalse(plane["b0_valid"][2, 2])
        self.assertTrue(plane["b0_valid"][0, 0])
        np.testing.assert_array_equal(bundle["b0_Hz"], before)

    def test_singleton_normal_axis_has_a_valid_plane_without_invented_width(self):
        plane = extract_volume_slice(make_bundle((3, 4, 1)), "z", 0)
        self.assertEqual(plane["reference_magnitude"].shape, (4, 3))
        line = extract_volume_slice(make_bundle((3, 4, 1)), "x", 0)
        self.assertEqual(line["extent_m"][2], line["extent_m"][3])

    def test_invalid_axis_grid_shape_and_slice_are_rejected(self):
        for axis, index in (("a", 0), (None, 0), ("x", True), ("x", 1.0), ("x", -1), ("x", 3)):
            with self.subTest(axis=axis, index=index), self.assertRaises(ValueError):
                extract_volume_slice(make_bundle(), axis, index)
        for mutation in (
                lambda b: b.update(axes_m=(np.array([0., 0., 1.]), *b["axes_m"][1:])),
                lambda b: b.update(axes_m=(np.array([2., 1., 0.]), *b["axes_m"][1:])),
                lambda b: b.update(epi_magnitude=np.ones((4, 3, 5))),
                lambda b: b.update(b0_valid=np.zeros((1, 2, 3))),
                lambda b: b.update(b0_valid=np.full((3, 4, 5), 2)),
                lambda b: b.update(reference_magnitude=-np.ones((3, 4, 5)))):
            bundle = make_bundle()
            mutation(bundle)
            with self.assertRaises(ValueError):
                VolumeSliceViewer(bundle)


class SelectionTests(unittest.TestCase):
    def test_three_panels_share_physical_axes_and_mask_the_b0_color_plot(self):
        import matplotlib
        matplotlib.use("Agg")
        bundle = make_bundle()
        bundle["b0_valid"] = np.ones((3, 4, 5), bool)
        bundle["b0_valid"][1, 2, 3] = False
        viewer = VolumeSliceViewer(bundle, axis="z", index=3)
        viewer.output = MagicMock()
        figures = []
        with patch("IPython.display.display", side_effect=figures.append):
            viewer._render()
        figure = figures[0]
        self.assertEqual(len(figure.axes), 4)  # Three views plus the B0 colorbar.
        expected_extent_mm = np.asarray(viewer.selected_slice()["extent_m"]) * 1000
        for ax in figure.axes[:3]:
            self.assertEqual(ax.get_xlabel(), "X [mm]")
            np.testing.assert_allclose(ax.get_xlim(), expected_extent_mm[:2])
            np.testing.assert_allclose(ax.get_ylim(), expected_extent_mm[2:])
            self.assertTrue(ax.get_shared_x_axes().joined(ax, figure.axes[0]))
        b0_mask = np.ma.getmaskarray(figure.axes[2].collections[0].get_array()).reshape(4, 3)
        self.assertTrue(b0_mask[2, 1])
        self.assertEqual(figure.axes[3].get_ylabel(), "B0 [Hz]")
        self.assertFalse(viewer.confirmed)  # Rendering cannot approve a slice.

    def test_confirmation_and_generation_make_stale_slices_unusable(self):
        viewer = VolumeSliceViewer(make_bundle(), axis="z", index=4)
        with self.assertRaisesRegex(RuntimeError, "Use this slice"):
            viewer.snapshot()
        confirmed = viewer.confirm_selection()
        self.assertEqual(confirmed["axis"], "z")
        self.assertEqual(confirmed["index"], 4)
        self.assertEqual(confirmed["generation"], 0)
        self.assertTrue(viewer.confirmed)
        viewer.select(axis="x")  # Clamp 4 to the final X index, 2.
        self.assertEqual(viewer.index, 2)
        self.assertEqual(viewer.generation, 1)
        self.assertFalse(viewer.confirmed)
        with self.assertRaises(RuntimeError):
            viewer.snapshot()
        viewer.confirm_selection()
        viewer.select(index=1)
        self.assertEqual(viewer.generation, 2)
        with self.assertRaises(RuntimeError):
            viewer.snapshot()
        viewer.select(axis="z", index=4)  # Returning to an old slice cannot restore approval.
        self.assertEqual(viewer.generation, 3)
        with self.assertRaises(RuntimeError):
            viewer.snapshot()

    def test_noop_retains_confirmation_and_snapshot_cannot_mutate_viewer(self):
        viewer = VolumeSliceViewer(make_bundle(), index=1)
        old = viewer.confirm_selection()
        viewer.select(axis="Z", index=1)
        self.assertEqual(viewer.snapshot(), old)
        old["index"] = 999
        old["orientation"]["row_axis"] = "wrong"
        self.assertEqual(viewer.snapshot()["index"], 1)
        self.assertEqual(viewer.snapshot()["orientation"]["row_axis"], "y")

    def test_explicit_invalid_index_does_not_change_confirmed_selection(self):
        viewer = VolumeSliceViewer(make_bundle(), index=2)
        expected = viewer.confirm_selection()
        with self.assertRaises(ValueError):
            viewer.select(axis="x", index=99)
        self.assertEqual(viewer.snapshot(), expected)

    def test_widgets_are_not_imported_for_headless_use(self):
        with patch.dict(sys.modules, {"ipywidgets": None}):
            module = importlib.reload(sys.modules["volume_slice_viewer"])
            viewer = module.VolumeSliceViewer(make_bundle())
            viewer.confirm_selection()
            self.assertEqual(viewer.snapshot()["axis"], "z")
            with self.assertRaisesRegex(ImportError, "kernel environment"):
                _ = viewer.widget


if __name__ == "__main__":
    unittest.main()
