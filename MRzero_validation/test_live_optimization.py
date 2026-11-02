"""Keep displayed ROI boundaries on the same pixels as live images."""

import unittest
from unittest.mock import Mock, patch

import matplotlib
matplotlib.use("Agg")
import numpy as np

from live_optimization import LiveOptimizationPlot


class LiveOptimizationContourTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((11, 13), dtype=bool)
        # Deliberately off-centre and non-square: a vertical flip cannot hide.
        self.mask[1:4, 2:6] = True
        self.reference = self.mask.astype(float) * 2
        for target, kwargs in (
            ("live_optimization.display", {"return_value": Mock()}),
            ("live_optimization.LiveOptimizationPlot._render_image", {"return_value": None}),
        ):
            patcher = patch(target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

    def make_plot(self, extent=None):
        return LiveOptimizationPlot(
            self.reference, self.reference * 0.9, 3,
            roi_mask=self.mask, extent_mm=extent,
        )

    def assert_contour_bounds(self, plot, expected):
        titles = ("Reference", "Baseline", "Current", "Signed magnitude residual")
        for axis in plot.figure.axes[:4]:
            with self.subTest(panel=axis.get_title()):
                self.assertIn(axis.get_title(), titles)
                self.assertEqual(len(axis.collections), 1)
                vertices = np.concatenate([
                    path.vertices for path in axis.collections[0].get_paths()
                ])
                bounds = np.stack((vertices.min(axis=0), vertices.max(axis=0)))
                np.testing.assert_allclose(bounds, expected, rtol=0, atol=1e-12)

    def test_row_column_contours_follow_upper_origin_pixel_centres(self):
        plot = self.make_plot()
        self.assertEqual(plot.figure.axes[0].images[0].origin, "upper")
        self.assert_contour_bounds(plot, [[1.5, 0.5], [5.5, 3.5]])
        for axis in plot.figure.axes[:4]:
            self.assertGreater(axis.get_ylim()[0], axis.get_ylim()[1])

    def test_physical_contours_follow_extent_even_when_axis_is_reversed(self):
        for extent in ((-130, 130, -55, 55), (130, -130, 55, -55)):
            with self.subTest(extent=extent):
                plot = self.make_plot(extent)
                self.assertEqual(plot.figure.axes[0].images[0].origin, "lower")
                left, right, bottom, top = extent
                # The rectangle starts/ends at column edges 2/6 and row edges 1/4.
                x = left + np.array([2, 6]) * (right - left) / 13
                y = bottom + np.array([1, 4]) * (top - bottom) / 11
                self.assert_contour_bounds(plot, [[x.min(), y.min()], [x.max(), y.max()]])

    def test_display_keeps_reference_and_objective_roi_in_original_array_frame(self):
        before_reference = self.reference.copy()
        before_mask = self.mask.copy()
        plot = self.make_plot()
        current = self.reference.copy()
        current[self.mask] += 1
        current[~self.mask] = 1e9
        plot.update(0, current, np.zeros((2, 3)), 0.25)
        # The huge error outside the original ROI must not enter the display metric.
        self.assertIn("NRMSE 50.00%", plot._last_summary)
        self.assertEqual(plot.losses, [0.25])
        np.testing.assert_array_equal(self.reference, before_reference)
        np.testing.assert_array_equal(plot.reference, before_reference)
        np.testing.assert_array_equal(self.mask, before_mask)
        np.testing.assert_array_equal(plot.roi_mask, before_mask)
        np.testing.assert_array_equal(plot._current_image.get_array(), current)


if __name__ == "__main__":
    unittest.main()
