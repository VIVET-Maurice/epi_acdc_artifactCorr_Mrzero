"""Check imports and the actual CUDA backend, without acquisition data."""
from __future__ import annotations

import argparse
import importlib
from importlib.metadata import version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true", help="Fail if CUDA is unavailable.")
    args = parser.parse_args()
    packages = {
        "torch": "torch", "MRzeroCore": "MRzeroCore", "numpy": "numpy",
        "scipy": "scipy", "matplotlib": "matplotlib", "h5py": "h5py",
        "pypulseq": "pypulseq", "twixtools": "twixtools", "pydicom": "pydicom",
        "PIL": "Pillow", "nbformat": "nbformat", "nbclient": "nbclient",
    }
    for module, distribution in packages.items():
        importlib.import_module(module)
        print(f"{distribution}: {version(distribution)}")
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA is required: check the GPU allocation, driver and PyTorch build.")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA build: {torch.version.cuda}; capability: {torch.cuda.get_device_capability(0)}")
        # Availability alone does not prove that kernels support this GPU.
        sample = torch.ones((32, 32), device="cuda")
        assert (sample @ sample)[0, 0].item() == 32
        torch.cuda.synchronize()
        print("CUDA kernel execution: OK")
    print("Server import/backend check: OK")


if __name__ == "__main__":
    main()
