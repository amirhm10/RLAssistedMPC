"""Pinned process-local NumPy recovery; never changes the shared environment."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / 'change-reports/recovery_20261021/runtime'
LEGACY = ROOT / 'change-reports/run_recovery_20261020/runtime'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def manifest():
    return json.loads((RUNTIME / 'manifest.json').read_text(encoding='utf-8'))


def activate(directory=None, expected_manifest_hash=None):
    record = manifest()
    if digest(__file__) != record['recovery_bootstrap_sha256']:
        raise RuntimeError('Recovery numerical bootstrap changed')
    if directory is not None and Path(directory).resolve() != LEGACY.resolve():
        raise RuntimeError('Recovery runtime received an unrecognized legacy runtime')
    if digest(LEGACY / 'manifest.json') != record['legacy_manifest_sha256']:
        raise RuntimeError('Legacy runtime manifest changed')
    if expected_manifest_hash and expected_manifest_hash != record['legacy_manifest_sha256']:
        raise RuntimeError('Prepared job expects another numerical runtime')
    if digest(ROOT / 'review_blas_runtime.py') != record['legacy_bootstrap_sha256']:
        raise RuntimeError('Legacy numerical bootstrap changed')
    for base, files in ((RUNTIME/'site-packages',record['files']),
                        (LEGACY/'site-packages',json.loads((LEGACY/'manifest.json').read_text())['files'])):
        for name, expected in files.items():
            target = (base/name).resolve()
            if not target.is_relative_to(base.resolve()) or digest(target) != expected:
                raise RuntimeError(f'Private numerical file differs: {name}')
    if 'numpy' in sys.modules and not Path(sys.modules['numpy'].__file__).resolve().is_relative_to(RUNTIME/'site-packages'):
        raise RuntimeError('Recovery selected after another NumPy was already imported')
    from run_polymer_e3_recovery import environment
    os.environ.update(environment())
    for site in (str(LEGACY/'site-packages'),str(RUNTIME/'site-packages')):
        if site in sys.path:
            sys.path.remove(site)
        sys.path.insert(0,site)
    return record


def runtime_info(directory=None):
    import numpy as np
    import scipy
    import scipy.linalg.blas
    from numpy._core import _multiarray_umath
    from threadpoolctl import threadpool_info
    record=manifest()
    if np.__version__ != record['numpy'] or scipy.__version__ != record['scipy']:
        raise RuntimeError('Recovery package versions differ')
    if not Path(np.__file__).resolve().is_relative_to(RUNTIME/'site-packages'):
        raise RuntimeError('NumPy did not load from the recovery runtime')
    if _multiarray_umath.__cpu_features__.get('X86_V3') is not False:
        raise RuntimeError('Conservative NumPy dispatch was not selected')
    pools=threadpool_info()
    private=[p for p in pools if p['internal_api']=='openblas' and any(Path(p['filepath']).resolve().is_relative_to(base/'site-packages') for base in (RUNTIME,LEGACY))]
    if len(private)!=2 or any(p['num_threads']!=1 or p['architecture'].lower()!='nehalem' for p in private):
        raise RuntimeError('Recovery BLAS dispatch differs')
    return dict(python=sys.executable,numpy=np.__version__,scipy=scipy.__version__,numpy_file=np.__file__,
                scipy_file=scipy.__file__,private_blas=private,threadpools=pools,numpy_x86_v3_enabled=False,
                runtime_manifest_sha256=digest(RUNTIME/'manifest.json'),
                runtime_bootstrap_sha256=digest(__file__),shared_environment_modified=False,
                recovery_runtime=str(RUNTIME),legacy_manifest_sha256=record['legacy_manifest_sha256'],
                limitation='Targeted private NumPy build replacement; multi-day stability remains to be established. Other optional dependencies may load Conda BLAS.')


def install():
    import review_blas_runtime
    review_blas_runtime.activate=activate
    review_blas_runtime.runtime_info=runtime_info
    activate()
