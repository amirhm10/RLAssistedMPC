"""Private, pinned NumPy/SciPy builds for new recovery workers only.

The shared Conda installation and scientific sources are never changed. Both
packages keep their installed versions but use the official Windows wheels'
bundled BLAS libraries instead of Conda's faulting openblas.dll for NumPy/SciPy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / "change-reports/run_recovery_20261020/runtime"
WHEELS = {
    "numpy-2.4.6-cp313-cp313-win_amd64.whl": "c4fc99836233ea196540b17ab0983aff60ed07941751930f5f4d05bc3b3b7359",
    "scipy-1.17.1-cp313-cp313-win_amd64.whl": "37425bc9175607b0268f493d79a292c39f9d001a357bebb6b88fdfaff13f6448",
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prepare(directory=DEFAULT):
    """Verify downloaded wheels and their extracted tree; no package install."""
    directory = Path(directory).resolve()
    site = directory / "site-packages"
    site.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, expected in WHEELS.items():
        wheel = directory / "wheels" / name
        if digest(wheel) != expected:
            raise RuntimeError(f"Downloaded wheel hash mismatch: {name}")
        with zipfile.ZipFile(wheel) as archive:
            for entry in archive.infolist():
                if entry.is_dir():
                    continue
                target = (site / entry.filename).resolve()
                if not target.is_relative_to(site):
                    raise RuntimeError("Wheel contains a path outside the private runtime.")
                content = archive.read(entry)
                sha = hashlib.sha256(content).hexdigest()
                if target.exists() and digest(target) != sha:
                    raise RuntimeError(f"Existing overlay differs from pinned wheel: {entry.filename}")
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
                files[entry.filename] = sha
    record = {"source": "https://pypi.org/simple", "wheels": WHEELS, "files": files,
              "numpy": "2.4.6", "scipy": "1.17.1", "site_packages": str(site),
              "expected_blas_versions": ["0.3.31.188.0", "0.3.30"],
              "shared_environment_modified": False}
    (directory / "manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return {key: value for key, value in record.items() if key != "files"} | {"verified_files": len(files)}


def activate(directory=DEFAULT, expected_manifest_hash=None):
    """Must run before any scientific import, including imports in workers."""
    from run_polymer_e3_recovery import environment
    if "numpy" in sys.modules or "scipy" in sys.modules:
        raise RuntimeError("Private BLAS runtime must be selected before NumPy/SciPy import.")
    directory = Path(directory).resolve()
    manifest_path = directory / "manifest.json"
    if expected_manifest_hash and digest(manifest_path) != expected_manifest_hash:
        raise RuntimeError("Private numerical runtime manifest changed after preparation.")
    record = read(manifest_path)
    if record["wheels"] != WHEELS:
        raise RuntimeError("Unexpected numerical wheel inventory.")
    site = directory / "site-packages"
    for relative, expected in record["files"].items():
        target = (site / relative).resolve()
        if not target.is_relative_to(site) or digest(target) != expected:
            raise RuntimeError(f"Private runtime file changed: {relative}")
    os.environ.update(environment())
    sys.path.insert(0, str(site))
    return record


def runtime_info(directory=DEFAULT):
    import numpy as np
    import scipy
    import scipy.linalg.blas
    import scipy.optimize
    from numpy._core import _multiarray_umath
    from threadpoolctl import threadpool_info
    site = (Path(directory).resolve() / "site-packages")
    record = read(site.parent / "manifest.json")
    if (not Path(np.__file__).resolve().is_relative_to(site)
            or not Path(scipy.__file__).resolve().is_relative_to(site)
            or np.__version__ != record["numpy"] or scipy.__version__ != record["scipy"]):
        raise RuntimeError("NumPy/SciPy did not load the approved private builds.")
    pools = threadpool_info()
    private = [p for p in pools if p["internal_api"] == "openblas"
               and Path(p["filepath"]).resolve().is_relative_to(site)]
    if (sorted(p["version"] for p in private) != sorted(record["expected_blas_versions"])
            or any(p["architecture"] != "Nehalem" or p["num_threads"] != 1 for p in private)
            or _multiarray_umath.__cpu_features__.get("X86_V3") is not False):
        raise RuntimeError("Private NumPy/SciPy BLAS dispatch did not validate.")
    return {"python": sys.executable, "numpy": np.__version__, "scipy": scipy.__version__,
            "numpy_file": np.__file__, "scipy_file": scipy.__file__, "private_blas": private,
            "threadpools": pools, "numpy_x86_v3_enabled": False,
            "runtime_manifest_sha256": digest(site.parent / "manifest.json"),
            "runtime_bootstrap_sha256": digest(__file__), "shared_environment_modified": False,
            "scope": "New processes only; NumPy/SciPy math/settings unchanged; floating-point build differences recorded.",
            "limitation": "Other dependencies can still load Conda BLAS; bounded checks cannot prove multi-day stability."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        value = prepare(args.directory)
    else:
        activate(args.directory)
        value = runtime_info(args.directory)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
