"""Browse registered reference/EPI/B0 volumes and explicitly confirm one slice.

The bundle is already resampled onto one world grid. This module never reads
acquisitions, resamples data, starts a simulation, or communicates with hardware.
Arrays use (x, y, z) order; display rows point upward along the vertical world
axis. Widget and plotting imports are deferred until the viewer is displayed.
"""

from __future__ import annotations

from copy import deepcopy
import numbers

import numpy as np


_AXES = ("x", "y", "z")
_VOLUMES = ("reference_magnitude", "epi_magnitude", "b0_Hz")
# (normal axis, displayed row axis, displayed column axis)
_PLANES = {"x": (0, 2, 1), "y": (1, 2, 0), "z": (2, 1, 0)}


def _axis_name(axis):
    if not isinstance(axis, str) or axis.lower() not in _PLANES:
        raise ValueError("Slice axis must be 'x', 'y', or 'z'.")
    return axis.lower()


def _index(value, size):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError("Slice index must be an integer.")
    value = int(value)
    if not 0 <= value < size:
        raise ValueError(f"Slice index must be between 0 and {size - 1}.")
    return value


def _validated_bundle(bundle):
    if not isinstance(bundle, dict):
        raise ValueError("volume_bundle must be a dictionary.")
    if "axes_m" not in bundle:
        raise ValueError("volume_bundle needs three ascending world-coordinate axes_m.")
    axes = bundle["axes_m"]
    if not isinstance(axes, (tuple, list)) or len(axes) != 3:
        raise ValueError("axes_m must contain three coordinate vectors in (x, y, z) order.")
    checked_axes = []
    for name, values in zip(_AXES, axes):
        axis = np.asarray(values)
        if (axis.ndim != 1 or axis.size < 1 or np.iscomplexobj(axis)
                or not np.issubdtype(axis.dtype, np.number)
                or not np.isfinite(axis).all()):
            raise ValueError(f"World axis {name} must be a finite, strictly ascending coordinate vector.")
        axis = axis.astype(float, copy=True)
        if np.any(np.diff(axis) <= 0):
            raise ValueError(f"World axis {name} must be a finite, strictly ascending coordinate vector.")
        checked_axes.append(axis)
    shape = tuple(len(axis) for axis in checked_axes)
    checked = dict(axes_m=tuple(checked_axes))
    for name in _VOLUMES:
        values = np.asarray(bundle.get(name))
        if (values.shape != shape or np.iscomplexobj(values)
                or not np.issubdtype(values.dtype, np.number)):
            raise ValueError(f"{name} must be a real numeric volume with shape {shape} in (x, y, z) order.")
        if name.endswith("magnitude") and np.any(values[np.isfinite(values)] < 0):
            raise ValueError(f"{name} must contain nonnegative magnitudes.")
        # Volume arrays are held by reference, never modified or copied merely
        # to browse. Their registered grid must stay fixed for this viewer.
        checked[name] = values
    if "b0_valid" in bundle:
        mask = np.asarray(bundle["b0_valid"])
        if mask.shape != shape or not np.isin(mask, [False, True]).all():
            raise ValueError("b0_valid must be a binary mask matching the world grid.")
        checked["b0_valid"] = mask.astype(bool, copy=False)
    else:
        checked["b0_valid"] = None
    return checked


def _coordinate_edges(centers):
    """Physical pixel edges, including nonuniform rectilinear coordinates.

    A singleton coordinate has no inferable physical width. Keep its two
    boundaries at the measured coordinate instead of inventing a spacing.
    Extraction remains supported; the interactive plot reports this case.
    """
    if len(centers) == 1:
        return np.repeat(centers, 2)
    middle = (centers[:-1] + centers[1:]) / 2
    return np.concatenate(([centers[0] - (centers[1] - centers[0]) / 2], middle,
                           [centers[-1] + (centers[-1] - centers[-2]) / 2]))


def _extract(bundle, axis, index):
    normal, row, column = _PLANES[axis]
    axes = bundle["axes_m"]
    selection = [slice(None)] * 3
    selection[normal] = index
    selection = tuple(selection)
    # Remaining array axes are ascending; each requested plane displays their
    # transpose, giving rows/columns as documented in _PLANES.
    planes = {name: bundle[name][selection].T for name in _VOLUMES}
    b0_valid = np.isfinite(planes["b0_Hz"])
    if bundle["b0_valid"] is not None:
        b0_valid &= bundle["b0_valid"][selection].T
    column_edges = _coordinate_edges(axes[column])
    row_edges = _coordinate_edges(axes[row])
    orientation = dict(normal_axis=axis, row_axis=_AXES[row], column_axis=_AXES[column],
                       array_axes=[_AXES[row], _AXES[column]],
                       normal_vector_world=np.eye(3)[normal].tolist(),
                       display_origin="lower")
    return dict(planes, b0_valid=b0_valid, axis=axis, index=index,
                coordinate_m=float(axes[normal][index]), orientation=orientation,
                row_coordinates_m=axes[row].copy(), column_coordinates_m=axes[column].copy(),
                row_edges_m=row_edges, column_edges_m=column_edges,
                extent_m=[float(column_edges[0]), float(column_edges[-1]),
                          float(row_edges[0]), float(row_edges[-1])])


def extract_volume_slice(volume_bundle, axis, index):
    """Return a deterministic world-aligned plane without interpolation.

    ``x`` normal: rows=z, columns=y; ``y``: rows=z, columns=x;
    ``z``: rows=y, columns=x. ``extent_m`` is (left, right, bottom, top) at
    pixel boundaries. The full edges are also returned for nonuniform grids.
    NaNs and B0 validity are retained; no missing field is extrapolated.
    """
    bundle = _validated_bundle(volume_bundle)
    axis = _axis_name(axis)
    index = _index(index, len(bundle["axes_m"][_PLANES[axis][0]]))
    return _extract(bundle, axis, index)


class VolumeSliceViewer:
    """Linked three-panel browser with an explicit, invalidated-on-change choice.

    Construction and ``select``/``confirm_selection``/``snapshot`` are headless.
    Access ``widget`` or call ``display()`` in Jupyter to create the controls.
    Browsing changes only a slice and never launches downstream computation.
    The supplied volume arrays must remain unchanged during this viewer's life.
    """

    def __init__(self, volume_bundle, *, axis="z", index=None):
        self._bundle = _validated_bundle(volume_bundle)
        self._axis = _axis_name(axis)
        size = len(self._bundle["axes_m"][_PLANES[self._axis][0]])
        self._index = size // 2 if index is None else _index(index, size)
        self._generation = 0
        self._confirmed = None
        self._widget = None
        self._updating_controls = False
        self._plot_limits = None

    @property
    def axis(self):
        return self._axis

    @property
    def index(self):
        return self._index

    @property
    def generation(self):
        return self._generation

    @property
    def confirmed(self):
        return self._confirmed is not None

    def select(self, *, axis=None, index=None):
        """Change the displayed slice, invalidating its previous confirmation.

        When only the normal axis changes, retain the current index clamped to
        the new axis length. An explicit invalid index is rejected.
        """
        axis = self._axis if axis is None else _axis_name(axis)
        size = len(self._bundle["axes_m"][_PLANES[axis][0]])
        index = min(self._index, size - 1) if index is None else _index(index, size)
        if (axis, index) == (self._axis, self._index):
            return
        self._axis, self._index = axis, index
        self._generation += 1
        self._confirmed = None
        if self._widget is not None:
            self._sync_controls()
            self._render()

    def selected_slice(self):
        """Return the current display plane; confirmation is not required."""
        return _extract(self._bundle, self._axis, self._index)

    def confirm_selection(self):
        """Confirm exactly the visible axis/index/generation and return its record."""
        plane = self.selected_slice()
        self._confirmed = {key: deepcopy(plane[key])
                           for key in ("axis", "index", "coordinate_m", "orientation")}
        self._confirmed["generation"] = self._generation
        if self._widget is not None:
            self._update_status()
        return self.snapshot()

    def snapshot(self):
        """Return a confirmed selection, or reject an unconfirmed/stale choice."""
        if self._confirmed is None or self._confirmed["generation"] != self._generation:
            raise RuntimeError("No confirmed slice. Choose a slice and click 'Use this slice' first.")
        return deepcopy(self._confirmed)

    def _sync_controls(self):
        self._updating_controls = True
        try:
            self.orientation_dropdown.value = self._axis
            maximum = len(self._bundle["axes_m"][_PLANES[self._axis][0]]) - 1
            self.slice_slider.max = maximum
            self.slice_slider.value = self._index
        finally:
            self._updating_controls = False
        self._update_status()

    def _update_status(self):
        coordinate = self._bundle["axes_m"][_PLANES[self._axis][0]][self._index] * 1000
        selected = f"Normal {self._axis.upper()}, slice {self._index}: {coordinate:.3f} mm."
        state = "Confirmed. Ready to prepare this slice." if self.confirmed else "Click <b>Use this slice</b> to confirm."
        self.status.value = selected + " " + state

    def _render(self):
        import matplotlib.pyplot as plt
        from IPython.display import display

        plane = self.selected_slice()
        with self.output:
            self.output.clear_output(wait=True)
            # A plane with a singleton display axis is a line, not a 2D image.
            # Do not invent a millimetre extent for a missing spatial dimension.
            if min(plane["reference_magnitude"].shape) < 2:
                print("This orientation has only one voxel along a display axis. Choose another orientation.")
                return
            fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True, sharey=True,
                                     constrained_layout=True)
            x = plane["column_edges_m"] * 1000
            y = plane["row_edges_m"] * 1000
            if self._plot_limits is None:
                # Fixed contrasts make successive slices comparable. Scan each
                # volume only on the first plot, never on every slider change.
                magnitudes = [array[np.isfinite(array)] for array in
                              (self._bundle["reference_magnitude"], self._bundle["epi_magnitude"])]
                magnitude_max = max((float(values.max()) for values in magnitudes if values.size), default=1.)
                b0_values = self._bundle["b0_Hz"]
                b0_mask = np.isfinite(b0_values)
                if self._bundle["b0_valid"] is not None:
                    b0_mask &= self._bundle["b0_valid"]
                b0_max = float(np.max(np.abs(b0_values[b0_mask]))) if b0_mask.any() else 1.
                self._plot_limits = (magnitude_max if magnitude_max > 0 else 1.,
                                     b0_max if b0_max > 0 else 1.)
            magnitude_max, b0_max = self._plot_limits
            for ax, key, title in zip(axes, _VOLUMES, ("Reference magnitude", "EPI magnitude", "B0 field")):
                values = np.ma.masked_invalid(plane[key])
                if key == "b0_Hz":
                    values = np.ma.masked_where(~plane["b0_valid"], values)
                    mesh = ax.pcolormesh(x, y, values, cmap="RdBu_r", vmin=-b0_max, vmax=b0_max,
                                         shading="flat")
                    fig.colorbar(mesh, ax=ax, label="B0 [Hz]")
                else:
                    ax.pcolormesh(x, y, values, cmap="gray", vmin=0, vmax=magnitude_max,
                                  shading="flat")
                ax.set_title(title)
                ax.set_aspect("equal")
                ax.set_facecolor("black")
                ax.set_xlabel(plane["orientation"]["column_axis"].upper() + " [mm]")
            axes[0].set_ylabel(plane["orientation"]["row_axis"].upper() + " [mm]")
            fig.suptitle(f"Normal {self.axis.upper()} | slice {self.index} | {plane['coordinate_m'] * 1000:.3f} mm")
            display(fig)
            plt.close(fig)

    @property
    def widget(self):
        """Create the ipywidgets controls only when an interactive view is requested."""
        if self._widget is None:
            try:
                import ipywidgets as widgets
            except ImportError as error:
                raise ImportError("The interactive volume viewer needs ipywidgets in the notebook kernel environment.") from error
            self.orientation_dropdown = widgets.Dropdown(
                options=[("Normal X", "x"), ("Normal Y", "y"), ("Normal Z", "z")],
                value=self.axis, description="Orientation:")
            maximum = len(self._bundle["axes_m"][_PLANES[self.axis][0]]) - 1
            self.slice_slider = widgets.IntSlider(value=self.index, min=0, max=maximum,
                description="Slice:", continuous_update=False)
            self.confirm_button = widgets.Button(description="Use this slice", button_style="primary")
            self.status = widgets.HTML()
            self.output = widgets.Output()
            self._widget = widgets.VBox([
                widgets.HBox([self.orientation_dropdown, self.slice_slider, self.confirm_button]),
                self.status, self.output])

            def orientation_changed(change):
                if not self._updating_controls:
                    self.select(axis=change["new"])

            def slice_changed(change):
                if not self._updating_controls:
                    self.select(index=change["new"])

            self.orientation_dropdown.observe(orientation_changed, names="value")
            self.slice_slider.observe(slice_changed, names="value")
            self.confirm_button.on_click(lambda button: self.confirm_selection())
            self._update_status()
            self._render()
        return self._widget

    def display(self):
        from IPython.display import display
        display(self.widget)


__all__ = ["VolumeSliceViewer", "extract_volume_slice"]
