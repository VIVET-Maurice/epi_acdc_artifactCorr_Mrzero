"""Keep periodic magnitude images and browse them without current histories.

The NPZ contains float32 images and their true iteration numbers. The HTML
viewer embeds its PNG images and works offline without Python or widgets.
Notebook HTML interactivity requires trusted outputs; the standalone HTML
is the portable fallback. Image row zero is displayed at the top by default,
matching the existing live plot when no physical image extent is supplied.
"""

from base64 import b64encode
from html import escape
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


def _magnitude(value, name):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    image = np.abs(np.asarray(value)).astype(np.float32)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError(f"{name} must be a finite two-dimensional image.")
    return image.copy()


class OptimizationImageHistory:
    """Archive callback images only at exact multiples of ``every``.

    ``record(snapshot)`` accepts the ordinary optimizer callback dictionary.
    It reads only ``iteration`` and ``image``; no current arrays are retained.
    Iteration zero is included. A terminal callback outside the requested
    multiples is ignored. Images are independent float32 magnitude copies.
    ``html()`` returns an embeddable slider viewer; ``display()`` creates or
    updates the same IPython display. Both initially show the latest image.
    ``origin='upper'`` matches the default live plot. Choose ``'lower'`` when
    the live plot uses a physical image extent and a lower image origin.
    """

    def __init__(self, reference, baseline, every=20, *,
                 title="Evolution du scan simulé", origin="upper"):
        if isinstance(every, (bool, np.bool_)) or int(every) != every or every < 1:
            raise ValueError("every must be a positive integer.")
        self.every = int(every)
        self.reference = _magnitude(reference, "reference")
        self.baseline = _magnitude(baseline, "baseline")
        if self.reference.shape != self.baseline.shape:
            raise ValueError("reference and baseline must have the same image shape.")
        if origin not in ("upper", "lower"):
            raise ValueError("origin must be 'upper' or 'lower'.")
        self.origin = origin
        self.vmax = float(max(self.reference.max(), self.baseline.max(), np.finfo(np.float32).eps))
        self.title = str(title)
        self._iterations = []
        self._images = []
        self._display_handle = None

    def record(self, snapshot):
        """Return True only when a new periodic image has been archived."""
        iteration = snapshot["iteration"]
        if (isinstance(iteration, (bool, np.bool_)) or not np.isscalar(iteration)
                or not np.isfinite(iteration) or int(iteration) != iteration or iteration < 0):
            raise ValueError("snapshot iteration must be a nonnegative integer.")
        iteration = int(iteration)
        if iteration % self.every:
            return False
        if self._iterations and iteration <= self._iterations[-1]:
            if iteration == self._iterations[-1]:
                return False
            raise ValueError("Archived iterations must arrive in increasing order.")
        image = _magnitude(snapshot["image"], "snapshot image")
        if image.shape != self.reference.shape:
            raise ValueError("Snapshot image must match reference and baseline shape.")
        self._iterations.append(iteration)
        self._images.append(image)
        return True

    def arrays(self):
        """Return portable archive arrays; no complex data or current history."""
        images = (np.stack(self._images) if self._images
                  else np.empty((0,) + self.reference.shape, dtype=np.float32))
        return {"iterations": np.asarray(self._iterations, dtype=np.int64),
                "images": images, "reference": self.reference.copy(),
                "baseline": self.baseline.copy(), "display_vmax": np.float32(self.vmax),
                "every": np.int64(self.every), "origin": np.asarray(self.origin)}

    def save_npz(self, path):
        """Save lossless float32 magnitude images and iteration labels."""
        path = Path(path)
        if path.suffix.lower() != ".npz":
            raise ValueError("Image history archive path must end in .npz.")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.arrays())
        return path

    def _png(self, image):
        # All panels use the baseline/reference range, including during live
        # updates. Match the caller's matplotlib image origin explicitly.
        gray = np.rint(np.clip(image / self.vmax, 0., 1.) * 255).astype(np.uint8)
        if self.origin == "lower":
            gray = np.flipud(gray)
        buffer = BytesIO()
        Image.fromarray(gray).save(buffer, format="PNG")
        return "data:image/png;base64," + b64encode(buffer.getvalue()).decode("ascii")

    def html(self, *, standalone=False):
        """Return an offline viewer; slider positions use true iteration labels."""
        def panel(image, caption):
            return (f'<figure style="margin:0;min-width:0"><img alt="{escape(caption)}" '
                    f'src="{self._png(image)}" style="display:block;width:100%;image-rendering:pixelated">'
                    f'<figcaption style="text-align:center">{escape(caption)}</figcaption></figure>')
        frames = []
        for index, (iteration, image) in enumerate(zip(self._iterations, self._images)):
            hidden = " hidden" if index != len(self._images) - 1 else ""
            frames.append(f'<div data-history-frame{hidden}>' + panel(image, f"Itération {iteration}") + '</div>')
        if not frames:
            frames.append('<div data-history-frame>' + panel(self.baseline, "Aucune image archivée") + '</div>')
        last = max(0, len(self._images) - 1)
        labels = ", ".join(str(value) for value in self._iterations) or "aucune"
        disabled = " disabled" if len(self._images) < 2 else ""
        # An inline handler avoids external JS libraries or notebook widgets.
        handler = ("const root=this.closest('[data-opt-images]');"
                   "root.querySelectorAll('[data-history-frame]').forEach((f,i)=>"
                   "{f.hidden=i!==Number(this.value);});")
        body = (f'<section data-opt-images style="font:14px system-ui,sans-serif;max-width:1100px;margin:1em auto">'
            f'<h3>{escape(self.title)}</h3>'
            '<div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px">'
            + panel(self.reference, "Référence") + panel(self.baseline, "Baseline")
            + '<div>' + ''.join(frames) + '</div></div>'
            f'<label style="display:block;margin-top:12px">Scan simulé : '
            f'<input aria-label="Image archivée" type="range" min="0" max="{last}" value="{last}" step="1"'
            f' style="width:75%;vertical-align:middle" oninput="{escape(handler, quote=True)}"{disabled}></label>'
            f'<p style="margin:6px 0;color:#666">Itérations archivées : {labels}. '
            f'Échelle commune : 0 à {self.vmax:.6g}. '
            f'Origine des images en {"haut" if self.origin == "upper" else "bas"}.</p></section>')
        if standalone:
            return ('<!doctype html><html lang="fr"><meta charset="utf-8">'
                    '<meta name="viewport" content="width=device-width,initial-scale=1">'
                    f'<title>{escape(self.title)}</title><body>{body}</body></html>')
        return body

    def write_html(self, path):
        """Write a standalone slider with all PNGs embedded, requiring no server."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.html(standalone=True), encoding="utf-8")
        return path

    def display(self):
        """Create/update one notebook output, showing the latest stored image."""
        from IPython.display import HTML, display
        content = HTML(self.html())
        if self._display_handle is None:
            self._display_handle = display(content, display_id=True)
        else:
            self._display_handle.update(content)
        return self._display_handle


__all__ = ["OptimizationImageHistory"]
