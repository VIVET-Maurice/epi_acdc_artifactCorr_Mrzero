"""Execute a notebook with this Python interpreter and save its outputs."""

import argparse
import json
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager


def execute(path, timeout=600):
    path = path.resolve()
    notebook = nbformat.read(path, as_version=4)
    # A temporary kernel specification avoids registering a global Jupyter kernel.
    with tempfile.TemporaryDirectory(prefix="acdc_mrzero_") as temporary:
        temporary = Path(temporary)
        kernel = temporary / "kernels" / "acdc-mrzero"
        kernel.mkdir(parents=True)
        environment = {"MPLCONFIGDIR": str(temporary / "matplotlib"),
                       "IPYTHONDIR": str(temporary / "ipython")}
        (kernel / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "ACDC MRzero", "language": "python", "env": environment,
        }))
        manager = AsyncKernelManager(kernel_name="acdc-mrzero",
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel.parent)]))
        client = NotebookClient(notebook, km=manager, timeout=timeout,
            resources={"metadata": {"path": str(path.parent)}})
        try:
            # We supplied our own manager, so explicitly close its kernel.
            # Windows cannot remove IPython's temporary SQLite history while
            # the kernel process still holds it open.
            client.execute(cleanup_kc=True)
        finally:
            nbformat.write(notebook, path)
    print(f"Executed all cells successfully: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", nargs="?", type=Path,
        default=Path(__file__).with_name("02_dynamic_shimming_MRzero.ipynb"))
    parser.add_argument("--timeout", type=int, default=600, help="Per-cell timeout in seconds.")
    arguments = parser.parse_args()
    execute(arguments.notebook, arguments.timeout)
