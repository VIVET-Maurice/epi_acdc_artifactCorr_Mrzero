"""Export code/notebooks for GitHub without raw data or saved image outputs.

The destination is a separate checkout. Existing destination files are never
removed, and local executed notebooks are never edited by this exporter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import os
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
CODE_DIRS = ("Experiment", "MRzero_validation", "Mrzero_simulation", "docs", "scripts", "eval_other_subj")
SUFFIXES = {".py", ".m", ".md", ".ipynb", ".sh", ".ps1", ".yml", ".yaml", ".toml"}
SKIP_DIRS = {"__pycache__", ".git", "tmp", "pipeline_validation", "simulation_metadata", ".ipynb_checkpoints"}


def source_files():
    for path in ROOT.iterdir():
        if path.is_file() and (path.suffix in SUFFIXES or path.name in {".gitignore", ".gitattributes"}
                               or path.name.startswith("requirements") and path.suffix == ".txt"):
            yield path
    for name in CODE_DIRS:
        directory = ROOT / name
        if not directory.is_dir():
            continue
        for current, directories, names in os.walk(directory):
            directories[:] = [name for name in directories if name not in SKIP_DIRS
                               and not name.startswith((".venv", "results", "acquisition_synthetic", "_audit_twix_"))]
            for name in names:
                path = Path(current) / name
                if path.suffix in SUFFIXES or path.name.startswith("requirements") and path.suffix == ".txt":
                    yield path
    for path in sorted(ROOT.glob("acquisition_*/sequences/*.seq")):
        if path.is_file():
            yield path


def portable_notebook(path):
    notebook = nbformat.read(path, as_version=4)
    changes = ["saved outputs and execution state removed", "portable python3 kernel"]
    notebook.metadata = {
        "kernelspec": {"display_name": "Python (MRzero)", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    for cell in notebook.cells:
        cell.metadata = {key: value for key, value in cell.metadata.items() if key == "tags"}
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
            updated = re.sub(r'''(GPU_PROFILE\s*=\s*['"])gtx1660ti(['"])''', r'\1auto\2', cell.source)
            if updated != cell.source:
                changes.append("GPU profile default changed to auto")
                cell.source = updated
        cell.source = cell.source.replace("Python (MRzero GPU Windows)", "Python (MRzero)")
        if "audit = json.loads((PROJECT_ROOT/\"MRzero_validation/acquisition_015_geometry_audit.json\")" in cell.source:
            start = cell.source.index('audit = json.loads(')
            diagnostic = cell.source[start:]
            diagnostic = diagnostic.replace(
                'audit = json.loads((PROJECT_ROOT/"MRzero_validation/acquisition_015_geometry_audit.json").read_text(encoding="utf-8-sig"))',
                'audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))', 1)
            cell.source = (cell.source[:start]
                + 'audit_path=PROJECT_ROOT/"MRzero_validation/acquisition_015_geometry_audit.json"\n'
                + 'if audit_path.is_file():\n'
                + ''.join('    '+line+'\n' for line in diagnostic.splitlines())
                + 'else:\n    print("Audit historique local non fourni ; lecture des acquisitions effectuée.")\n')
            changes.append("historical local audit display made optional")
    nbformat.validate(notebook)
    return nbformat.writes(notebook).encode("utf-8"), sorted(set(changes))


def export(destination):
    destination = Path(destination).resolve()
    if destination == ROOT or ROOT.is_relative_to(destination):
        raise ValueError("Use a separate destination, not the source root or one of its parents.")
    if destination.is_relative_to(ROOT) and not destination.is_relative_to(ROOT / "tmp"):
        raise ValueError("Within the source workspace, export only under tmp/ to avoid recursive copying.")
    # A destination inside tmp/ is convenient for the first publication.
    destination.mkdir(parents=True, exist_ok=True)
    files, notebooks = [], {}
    for source in sorted(set(source_files())):
        relative = source.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".ipynb":
            content, changes = portable_notebook(source)
            notebooks[relative.as_posix()] = changes
        elif relative.as_posix() in {"MRzero_validation/build_experiment_notebook.py", "Mrzero_simulation/build_synthetic_notebooks.py"}:
            text = source.read_text(encoding="utf-8-sig")
            text = text.replace("Python (MRzero GPU Windows)", "Python (MRzero)")
            text = text.replace('"mrzero-gpu-win"', '"python3"')
            if source.name == "build_experiment_notebook.py":
                text = text.replace('parameters = parameters.replace(\'GPU_PROFILE = "auto"\', \'GPU_PROFILE = "gtx1660ti"\')',
                                    '# Keep the portable auto GPU profile from the template.')
            content = text.encode("utf-8")
        else:
            content = source.read_bytes()
        if source.suffix != ".seq":
            # Match Git's LF checkout so manifest hashes also work on Linux.
            content = content.removeprefix(b"\xef\xbb\xbf").replace(b"\r\n", b"\n")
        target.write_bytes(content)
        files.append({"path": relative.as_posix(), "bytes": len(content),
                      "sha256": hashlib.sha256(content).hexdigest()})
    manifest = {"scope": "Code, notebooks without saved outputs, and Pulseq sequence files",
                "raw_acquisitions_included": False, "calibrations_included": False,
                "saved_notebook_outputs_included": False,
                "notebook_count": len(notebooks), "file_count": len(files),
                "notebook_export_adjustments": notebooks, "files": files}
    (destination / "publication_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Exported {len(files)} source files, including {len(notebooks)} notebooks, to {destination}")
    print(f"Source size: {sum(item['bytes'] for item in files)/1024/1024:.2f} MiB")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    export(parser.parse_args().destination)
