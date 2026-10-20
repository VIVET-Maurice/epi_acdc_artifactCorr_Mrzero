"""Execute the readable reconstruction notebook and preserve its outputs.

Run this script with the Python environment described in README.md.
"""
import argparse
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
import sys

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("notebook", nargs="?", type=Path,
                    default=root / "02_acdc_experiment_step_by_step.ipynb")
parser.add_argument("--timeout", type=int, default=1800, help="Maximum seconds per cell")
args = parser.parse_args()
notebook_path = args.notebook.resolve()
notebook = nbformat.read(notebook_path, as_version=4)
def before_cell(cell, cell_index, **kwargs):
    if cell.cell_type == "code":
        first_line = cell.source.splitlines()[0] if cell.source else ""
        print(f"Cell {cell_index + 1}: {first_line}", flush=True)

def after_cell(**kwargs):
    # Keep completed figures and diagnostics available even if a later cell fails.
    nbformat.write(notebook, notebook_path)

client = NotebookClient(notebook, timeout=args.timeout,
                        resources={"metadata": {"path": str(notebook_path.parent)}},
                        on_cell_start=before_cell, on_cell_executed=after_cell)
# Launch this environment's Python directly, without registering a global kernel.
client.km = KernelManager(kernel_name="python3")
client.km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
try:
    client.execute()
finally:
    nbformat.write(notebook, notebook_path)
print(f"Notebook executed: {notebook_path}")
