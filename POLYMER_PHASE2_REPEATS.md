# Five repetitions of the polymer Scenario 2 study

`run_polymer_phase2_repeats.py` wraps the existing five polymer entrypoints. It does not edit a controller, agent, shared defaults file, plant model, or existing result bundle.

Here Phase 2 means the paper's Scenario 2, which currently has one result per configuration. Each repetition executes the complete existing `robustness_200_100` study: 200 initial episodes followed by 100 continuation episodes, with the existing within-run learning continuity.

## Schedule

Default order: horizon, weights, dynamic matrix (`markov`), residual, combined. Each category launches its five SG repetitions and five plain/ungated repetitions together: ten parallel simulation processes. Only after all ten finish and save does the next category start. This schedule also applies to the combined controller.

The five seed values are 7, 11, 23, 37, and 53, reused across categories and modes. The default plan has five sequential category groups and 50 complete runs. Selecting only one mode launches five processes per category. Custom seed lists change the number of repetitions launched together. The baseline is reused through the existing comparison code; the launcher does not regenerate it.

The wrapper selects SG/plain, selects the Scenario 2 profile, sets training seeds, and redirects output paths and prefixes in each worker process. Every other existing controller, reward, network, solver, disturbance, warm-up, and training setting is inherited. A worker-local defaults adapter covers the combined entrypoint without adding a hook to its source. No modified runner copy is generated.

## Run

Use the same scientific Python environment/interpreter you use for the working polymer simulations. The launcher defaults to its own `sys.executable` for every worker. It checks NumPy, SciPy, PyTorch, and Matplotlib before creating a new batch, and does not install or update packages.

From the repository directory, inspect the plan:

```powershell
python run_polymer_phase2_repeats.py --dry-run
```

Start all categories and both modes:

```powershell
python run_polymer_phase2_repeats.py
```

Run a selected category or mode:

```powershell
python run_polymer_phase2_repeats.py --categories horizon --modes sg plain
python run_polymer_phase2_repeats.py --categories weights markov residual combined --modes sg
```

If the terminal's Python differs from your working simulation interpreter:

```powershell
python run_polymer_phase2_repeats.py --python 'C:\path\to\your\working\python.exe'
```

Optional `--seeds 7 11 23 37 53`, `--threads-per-run 1`, and `--output-dir PATH` customize only replication/scheduling/output settings. The thread default is one per worker to avoid multiplying CPU/BLAS thread pools across ten simultaneous processes. The existing device selection is retained.

## Results and failures

New batches are stored under `Polymer/Results/phase2_repeats/<unique-batch-id>/`. Each category/mode/seed/attempt has an isolated directory with:

- `job.json`: repetition settings and source provenance;
- `run.log`: that worker's complete output/errors;
- `worker_status.json`: completion or failure status;
- `rl/<timestamp>/input_data.pkl` and the figures produced by the existing entrypoint;
- `rl/<timestamp>/replication_metadata.json`: seed, category/mode, profile, source commit/hash, and Python;
- the existing baseline-comparison figures under `comparison/` when available.

The batch's `manifest.json` records every job and result path. Separate result roots and short prefixes prevent simultaneous workers with identical second-resolution timestamps from overwriting one another.

If any repetition fails, the wrapper waits for the other gated and ungated repetitions in that category to finish saving, then stops before the next category. Finished results remain intact. To resume the saved plan:

```powershell
python run_polymer_phase2_repeats.py --resume 'Polymer\Results\phase2_repeats\<batch-id>'
```

Resume skips completed runs whose saved bundle exists, and reruns failed/interrupted jobs into a fresh attempt directory. It preserves earlier attempts. It restarts an interrupted repetition from the beginning of its complete 300-episode study; it does not invent a mid-training checkpoint resume. Source hashes are checked before resuming to avoid mixing changed implementations in one batch.

Resume refuses a batch still marked running. Use one launcher per batch and stop it through its own terminal before resuming.

Ctrl+C stops only the launcher's own child processes and leaves saved work available for resume. Seeds are recorded explicitly; identical numerical trajectories across different hardware/library versions are not promised.

## Verification

The focused tests use real short-lived Python workers to verify ten-way overlap across both modes, the between-category barrier, separate save paths, failure handling, and resume. Worker/defaults tests check mode selection and preservation of controller/learning settings without running training. Run them with:

```powershell
python -m unittest discover -s tests -p test_polymer_phase2_repeats.py -v
```
