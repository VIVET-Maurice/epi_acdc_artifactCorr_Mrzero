"""Inspect executed notebook outputs and record the concrete validation evidence."""
import base64
import json
from pathlib import Path
import sys

import nbformat

HERE = Path(__file__).resolve().parent
checks = {}
for filename in ("01_generer_acquisition_MRzero.ipynb", "02_tester_pipeline_V4.ipynb"):
    notebook = nbformat.read(HERE / filename, as_version=4)
    nbformat.validate(notebook)
    codes = [c for c in notebook.cells if c.cell_type == "code"]
    errors = [o for c in codes for o in c.outputs if o.output_type == "error"]
    assert not errors, errors
    assert all(c.execution_count is not None for c in codes)
    checks[filename] = dict(code_cells=len(codes), all_executed=True, errors=0)
    # Export one plot for a visual check without a browser dependency.
    target = "phantom_preview.png" if filename.startswith("01") else "pipeline_preview.png"
    images = [(i, o.data["image/png"]) for i,c in enumerate(notebook.cells)
              if c.cell_type == "code" for o in c.outputs
              if hasattr(o, "data") and "image/png" in o.data]
    selected = images[0] if filename.startswith("01") else next(item for item in images if item[0] == 15)
    (HERE / target).write_bytes(base64.b64decode(selected[1]))
report = json.loads((HERE / "simulation_metadata/acquisition_synthetic_twix_001/generation_validation.json").read_text(encoding="utf-8"))
assert report["epi_signal"]["signal_noise_consistent"]
unit = json.loads((HERE / "synthetic_unit_validation.json").read_text(encoding="utf-8"))
checks["environment"] = dict(actual_interpreter=sys.executable,
    requested=str(HERE.parent / ".venv-gpu-win/Scripts/python.exe"),
    requested_executable_here=True,
    requested_interpreter_used=Path(sys.executable).resolve() == (HERE.parent / ".venv-gpu-win/Scripts/python.exe").resolve())
checks["generation"] = report
manifest_path = HERE / "acquisition_synthetic_twix_001/acquisition.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
assert manifest["format"] == "mrzero_synthetic_twix"
for name in ("reference", "b0", "scan"):
    assert (manifest_path.parent / manifest["files"][name]).suffix == ".dat"
    assert manifest["reports"][name]["twix_export"]["roundtrip_exact"]
checks["native_twix_files"] = {name: manifest["files"][name] for name in ("reference", "b0", "scan")}

checks["separate_unit_checks"] = unit
checks["native_twix_roundtrip_tests"] = json.loads((HERE / "synthetic_twix_validation.json").read_text(encoding="utf-8"))
source = HERE.parent / "MRzero_validation/04_acdc_hardware_continuous_MRzero_V4.ipynb"
original = nbformat.read(source, as_version=4)
copy = nbformat.read(HERE / "02_tester_pipeline_V4.ipynb", as_version=4)
# Numerical model, fit, reconstruction, regularization, scoring and frame tests.
for i in (5, 7, 8, 10, 12, 14, 18, 20):
    a, b = original.cells[i].source, copy.cells[i].source
    a = a.replace("Trois échos mesurés", "Trois échos synthétiques")
    a = a.replace("EPI mesurée, repère commun", "EPI synthétique, repère commun")
    assert a == b, f"Unexpected change to original V4 numerical cell {i}"
checks["V4_numerical_cells_unchanged"] = [5, 7, 8, 10, 12, 14, 18, 20]
summary_path = max((HERE / "pipeline_validation").glob("*/twin_validation.json"), key=lambda path: path.stat().st_mtime)
summary = json.loads(summary_path.read_text(encoding="utf-8"))
checks["pipeline_summary"] = str(summary_path.relative_to(HERE))
checks["pipeline_best_variant"] = summary["best_on_calibration_scan"]
checks["pipeline_metrics"] = summary["metrics"]
(HERE / "synthetic_notebooks_validation.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
print(json.dumps({"notebooks": {k:v for k,v in checks.items() if k.endswith('.ipynb')},
    "best_variant": checks["pipeline_best_variant"],
    "best_NRMSE_percent": summary["metrics"][checks["pipeline_best_variant"]]["NRMSE_percent"],
    "frame_signal_error": summary["frame_signal_relative_error"],
    "frame_image_error": summary["frame_image_relative_error"]}, indent=2))
