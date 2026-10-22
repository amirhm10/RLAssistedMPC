# Private recovery runtime

This directory records the machine-local numerical recovery used by the October 21 remaining-run repair. Generated wheel contents, the virtual environment and the binary wheel remain local and are not committed.

Base interpreter: `C:\Users\hamediaa\AppData\Local\PythonEnvs\rl-assisted-env\python.exe` (Python 3.13.2). The local `env` was created with `-m venv --system-site-packages --without-pip`; the shared environment was not edited. `pypi_metadata.json` records the official download URL, exact filename and SHA-256. The wheel is unpacked in `site-packages`; `manifest.json` pins all 945 files, original private-runtime manifest/bootstrap and the new `review_numeric_recovery.py` bootstrap.

The environment's `Lib/site-packages/sitecustomize.py` prepends the scientific repository and calls `review_numeric_recovery.install()` before numerical imports. That bootstrap selects the pinned new NumPy directory first and the original private SciPy directory second, sets conservative single-thread dispatch, and verifies both file inventories. Failure exits with code 78. `runtime_info()` verifies actual versions, paths and private BLAS thread/architecture settings; worker records preserve that evidence.

The broker uses the original base `pythonw.exe` so its direct Windows Task Scheduler parent can be verified. Every new supervisor and scientific worker explicitly selects `env/Scripts/python.exe`. Windows virtual-environment proxy/child process identities are recorded separately.

Keep the selected runtime and all sealed recovery bootstrap files unchanged while the prepared/running jobs depend on their hashes. This is a targeted native-crash mitigation; multi-day stability remains to be established. See `../RECOVERY.md` for the bounded tests, provenance and scientific limits.
