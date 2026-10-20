"""One live optimization figure for Jupyter and VS Code notebooks.

The display uses an IPython display handle rather than a GUI event loop. It
works with the usual inline Matplotlib backend; ``ipympl`` is not required.
"""

from __future__ import annotations

from io import BytesIO
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from IPython.display import Image, display


def _as_numpy(value):
    """Detach Torch tensors (including CUDA tensors), or accept NumPy arrays."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _magnitude_image(value, name):
    image = np.abs(_as_numpy(value)).astype(float, copy=False)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError(f"{name} must be a finite two-dimensional image.")
    return image


class LiveOptimizationPlot:
    """Update the same figure with evaluated optimizer states and loss history.

    Parameters
    ----------
    reference, baseline : array-like or Torch tensor, shape (rows, columns)
        Complex or magnitude images. All image panels display magnitudes.
        The signed residual is ``abs(current) - abs(reference)``.
    current_limit_a : float
        Positive symmetric current bound, in amperes. The current heatmap
        always uses this fixed range, even when the optimizer is near zero.
    extent_mm : sequence of four floats, optional
        Matplotlib image extent ``(left, right, bottom, top)`` in millimetres.
        Images are displayed without transposition. Physical coordinates use
        origin="lower"; without an extent, origin="upper" matches the original
        notebook's row/column image panels.
    roi_mask : array-like of bool, optional
        Image-shaped mask for the displayed contour and default NRMSE. Pass
        ``nrmse_percent`` to update() when the objective uses a different mask.
    title : str
        Experiment label shown above the figure.

    ``currents_a`` passed to :meth:`update` has shape (channels, events).
    A static channel vector is also accepted. Event columns can represent
    individual EPI lines or independent shim states, as chosen by the caller.

    Magnitude and residual colour ranges are fixed from the reference and
    baseline. Updating an image never rescales them. GPU values are detached
    only for display; plotting does not participate in autograd.

    Call :meth:`update` every iteration, or at periodic checkpoints. For a
    periodic display, call :meth:`set_history` first with all completed scalar
    evaluations so the objective curve still includes every iteration.
    """

    def __init__(
        self,
        reference,
        baseline,
        current_limit_a,
        *,
        extent_mm=None,
        roi_mask=None,
        title="MRzero dynamic shimming",
    ):
        self.reference = _magnitude_image(reference, "reference")
        self.baseline = _magnitude_image(baseline, "baseline")
        if self.baseline.shape != self.reference.shape:
            raise ValueError("baseline and reference must have the same shape.")
        self.current_limit_a = float(current_limit_a)
        if not np.isfinite(self.current_limit_a) or self.current_limit_a <= 0:
            raise ValueError("current_limit_a must be finite and positive.")
        if extent_mm is not None:
            extent_mm = np.asarray(extent_mm, dtype=float)
            if extent_mm.shape != (4,) or not np.isfinite(extent_mm).all():
                raise ValueError("extent_mm must contain four finite coordinates.")
            if extent_mm[0] == extent_mm[1] or extent_mm[2] == extent_mm[3]:
                raise ValueError("extent_mm must span nonzero image dimensions.")
        self.roi_mask = None
        if roi_mask is not None:
            self.roi_mask = _as_numpy(roi_mask).astype(bool)
            if self.roi_mask.shape != self.reference.shape or not self.roi_mask.any():
                raise ValueError("roi_mask must match the image and select some pixels.")

        self.title = str(title)
        self.iterations = []
        self.losses = []
        self._started = perf_counter()
        self._finished = False
        self._last_summary = "Waiting for the initial objective evaluation"
        self._current_shape = None
        self._layout_frozen = False
        # A zero-valued reference remains displayable without a degenerate range.
        magnitude_max = max(self.reference.max(), self.baseline.max(), np.finfo(float).eps)

        self.figure = plt.figure(figsize=(15, 8), layout="constrained")
        grid = self.figure.add_gridspec(2, 4, height_ratios=(1.0, 0.85))
        image_axes = [self.figure.add_subplot(grid[0, column]) for column in range(4)]
        image_origin = "lower" if extent_mm is not None else "upper"
        image_options = {"origin": image_origin, "extent": extent_mm, "interpolation": "nearest"}
        images = (self.reference, self.baseline, self.baseline)
        magnitude_artists = []
        for axis, image, label in zip(image_axes[:3], images, ("Reference", "Baseline", "Current")):
            artist = axis.imshow(image, cmap="gray", vmin=0, vmax=magnitude_max, **image_options)
            magnitude_artists.append(artist)
            axis.set_title(label)
        self._current_image = magnitude_artists[-1]
        self._residual_image = image_axes[3].imshow(
            self.baseline - self.reference,
            cmap="RdBu_r",
            vmin=-magnitude_max,
            vmax=magnitude_max,
            **image_options,
        )
        image_axes[3].set_title("Signed magnitude residual")
        for axis in image_axes:
            axis.set_xlabel("x (mm)" if extent_mm is not None else "Column")
            axis.set_ylabel("y (mm)" if extent_mm is not None else "Row")
            # A full-image mask has no interior boundary to draw.
            if self.roi_mask is not None and not self.roi_mask.all():
                axis.contour(
                    self.roi_mask.astype(float),
                    levels=[0.5],
                    colors="limegreen",
                    linewidths=0.6,
                    origin=image_origin,
                    extent=extent_mm,
                )
        self.figure.colorbar(
            magnitude_artists[0], ax=image_axes[:3], shrink=0.75, pad=0.02, label="Magnitude (a.u.)"
        )
        self.figure.colorbar(
            self._residual_image, ax=image_axes[3], shrink=0.75, pad=0.02, label="Current - reference (a.u.)"
        )

        self._loss_axis = self.figure.add_subplot(grid[1, :2])
        (self._loss_line,) = self._loss_axis.plot([], [], color="tab:blue", linewidth=1.7)
        self._loss_axis.set(title="Objective history", xlabel="Iteration", ylabel="Relative magnitude MSE")
        self._loss_axis.grid(alpha=0.25)
        self._loss_axis.set_xlim(0, 1)

        self._current_axis = self.figure.add_subplot(grid[1, 2:])
        self._current_map = self._current_axis.imshow(
            np.zeros((1, 1)),
            cmap="RdBu_r",
            vmin=-self.current_limit_a,
            vmax=self.current_limit_a,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(0.5, 1.5, 0.5, 1.5),
        )
        self._current_axis.set(xlabel="Event / EPI line", ylabel="ACDC channel")
        self._current_axis.set_title(f"Currents: bound ±{self.current_limit_a:g} A")
        self.figure.colorbar(
            self._current_map,
            ax=self._current_axis,
            label="Current (A)",
            ticks=[-self.current_limit_a, 0, self.current_limit_a],
            pad=0.02,
        )
        # Reserve three text lines, including the optional field constraint.
        self.figure.suptitle(f"{self.title}\n{self._last_summary}\n", fontsize=12)

        # Keep the handle's output, while preventing a second automatic inline
        # figure at the end of the notebook cell. Recent Matplotlib versions
        # remove the Agg canvas on close, so attach an explicit rendering canvas.
        plt.close(self.figure)
        self._canvas = FigureCanvasAgg(self.figure)
        self._display_handle = display(self._render_image(), display_id=True)

    def set_history(self, iterations, losses):
        """Store the full scalar history without rendering another frame.

        Both arrays must be nonempty and one-dimensional with the same shape.
        Iterations must be strictly increasing nonnegative integers, and
        losses must be finite nonnegative relative magnitude MSE values.
        A subsequent ``update`` at the latest iteration replaces that last
        loss while displaying the full history, including earlier peaks.
        """
        if self._finished:
            raise RuntimeError("This plot is finished; create a new plot for another run.")
        steps = _as_numpy(iterations)
        values = _as_numpy(losses)
        if steps.ndim != 1 or values.ndim != 1 or steps.size == 0 or steps.shape != values.shape:
            raise ValueError("iterations and losses must have the same nonempty one-dimensional shape.")
        if (not np.issubdtype(steps.dtype, np.number) or np.iscomplexobj(steps)
                or not np.isfinite(steps).all() or np.any(steps < 0)
                or np.any(steps != np.floor(steps)) or np.any(steps[1:] <= steps[:-1])):
            raise ValueError("iterations must be strictly increasing nonnegative integers.")
        if (not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values)
                or not np.isfinite(values).all() or np.any(values < 0)):
            raise ValueError("losses must be finite nonnegative relative magnitude MSE values.")
        self.iterations = [int(step) for step in steps]
        self.losses = [float(value) for value in values]

    def update(
        self,
        iteration,
        image,
        currents_a,
        loss,
        *,
        nrmse_percent=None,
        max_field_hz=None,
        field_limit_hz=None,
        phase="Adam",
    ):
        """Show the evaluated state and append its loss to the live history.

        Call for iteration zero and each completed optimizer step, or for
        periodic checkpoints after ``set_history`` stores all scalar steps.
        Repeating the latest iteration replaces its loss (useful for a final
        accepted state); decreasing iteration numbers are rejected. ``loss``
        should be the relative magnitude MSE: the sum of squared magnitude
        residuals divided by the sum of squared reference magnitudes, using
        the objective's selected pixels.
        ``max_field_hz`` is the largest absolute ACDC field on the constraint
        mask, excluding the background B0, when following the original method.
        """
        if self._finished:
            raise RuntimeError("This plot is finished; create a new plot for another run.")
        step = int(iteration)
        if step != iteration or step < 0 or (self.iterations and step < self.iterations[-1]):
            raise ValueError("iteration must be a nonnegative, nondecreasing integer.")
        current_image = _magnitude_image(image, "image")
        if current_image.shape != self.reference.shape:
            raise ValueError("image must have the same shape as reference.")
        currents = _as_numpy(currents_a).astype(float, copy=False)
        if currents.ndim == 1:
            currents = currents[:, None]
        if currents.ndim != 2 or 0 in currents.shape or not np.isfinite(currents).all():
            raise ValueError("currents_a must be finite with shape (channels, events).")
        loss = float(loss)
        if not np.isfinite(loss) or loss < 0:
            raise ValueError("loss must be a finite, nonnegative relative magnitude MSE.")

        if self.iterations and step == self.iterations[-1]:
            self.losses[-1] = loss
        else:
            self.iterations.append(step)
            self.losses.append(loss)
        self._current_image.set_data(current_image)
        self._residual_image.set_data(current_image - self.reference)
        self._loss_line.set_data(self.iterations, self.losses)
        self._loss_axis.set_xlim(0, max(1, step))
        # Setting a lower limit disables Matplotlib's automatic y-scaling.
        # Derive both limits from the complete history so early Adam peaks
        # remain visible when the first updates increase the objective.
        upper = max(max(self.losses), np.finfo(float).eps) * 1.05
        self._loss_axis.set_ylim(0, upper)

        channels, events = currents.shape
        self._current_map.set_data(currents)
        if self._current_shape != currents.shape:
            self._current_map.set_extent((0.5, events + 0.5, 0.5, channels + 0.5))
            for count, setter in (
                (events, self._current_axis.set_xticks),
                (channels, self._current_axis.set_yticks),
            ):
                setter(np.unique(np.linspace(1, count, min(count, 8), dtype=int)))
            self._current_shape = currents.shape
        max_current = float(np.abs(currents).max())
        current_violation = max_current > self.current_limit_a * (1 + 1e-6)
        self._current_axis.set_title(
            f"Currents: max |I| = {max_current:.3g} / {self.current_limit_a:g} A",
            color="crimson" if current_violation else "black",
        )

        if nrmse_percent is None:
            mask = self.roi_mask if self.roi_mask is not None else np.ones(self.reference.shape, bool)
            denominator = np.linalg.norm(self.reference[mask])
            nrmse_percent = (
                100 * np.linalg.norm((current_image - self.reference)[mask]) / denominator
                if denominator > 0 else np.nan
            )
        summary = [f"{phase} iteration {step}", f"relative MSE {loss:.6g}"]
        if np.isfinite(float(nrmse_percent)):
            summary.append(f"NRMSE {float(nrmse_percent):.2f}%")
        if max_field_hz is not None:
            field_text = f"max |ACDC field| {float(max_field_hz):.3g}"
            if field_limit_hz is not None:
                field_text += f" / {float(field_limit_hz):g}"
            summary.append(field_text + " Hz")
        summary.append(f"elapsed {perf_counter() - self._started:.1f} s")
        # Short lines fit the fixed canvas without repeatedly recomputing layout.
        self._last_summary = " | ".join(summary[:3])
        if len(summary) > 3:
            self._last_summary += "\n" + " | ".join(summary[3:])
        self.figure.suptitle(f"{self.title}\n{self._last_summary}", fontsize=12)
        self._refresh()

    def _render_image(self):
        """Draw once and encode a PNG without IPython's extra tight-bbox pass."""
        buffer = BytesIO()
        self._canvas.print_png(buffer, pil_kwargs={"compress_level": 1})
        return Image(data=buffer.getvalue(), format="png")

    def _refresh(self):
        """Replace the existing notebook output; never append another figure."""
        image = self._render_image()
        if not self._layout_frozen:
            # The first update establishes the actual channel/event tick labels.
            # Keep these positions for the following iterations; the images and
            # colour scales retain exactly the same size throughout the run.
            self.figure.set_layout_engine("none")
            self._layout_frozen = True
        if self._display_handle is not None:
            self._display_handle.update(image)

    def finish(self, status="Completed"):
        """Keep the final state visible and release pyplot's figure registration."""
        if self._finished:
            return
        self.figure.suptitle(f"{self.title} — {status}\n{self._last_summary}", fontsize=12)
        self._refresh()
        plt.close(self.figure)
        self._finished = True
