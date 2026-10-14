# Distillation Scenario 2 repetition wrapper

`run_distillation_phase2_repeats.py` repeats the existing distillation study with the same schedule as the ten-process polymer wrapper: five SG and five plain runs together, followed by the next channel only after every current run has saved its results and closed its Aspen application.

The five channels run in this order:

| Channel | Existing entrypoint |
| --- | --- |
| Horizon | `distillation_RL_assisted_MPC_horizons_unified.py` |
| Weights | `distillation_RL_assisted_MPC_weights_unified.py` |
| Dynamic matrix (`markov`) | `distillation_RL_assisted_MPC_markov_unified.py` |
| Residual | `distillation_RL_assisted_MPC_residual_unified.py` |
| Combined | `distillation_RL_assisted_MPC_combined_unified.py` |

The default plan has 50 complete repetitions. Each uses the existing `temperature_flip_200_100` profile: 200 first-phase episodes followed by 100 second-phase episodes, with `run_mode="disturb"` and `disturbance_profile="fluctuation"`. The existing phase-two learning, frozen exploration, and final evaluation behavior comes from the unchanged scenario code. This wrapper executes the complete study, just as the polymer wrapper does.

## Aspen files 1 through 10

Every channel uses the following assignments. The same seed is paired across SG/plain modes, with a separate Aspen file for each concurrent repetition.

| Seed | SG model | Plain model |
| --- | --- | --- |
| 7 | `C2S_SS_simulation1.dynf` | `C2S_SS_simulation6.dynf` |
| 11 | `C2S_SS_simulation2.dynf` | `C2S_SS_simulation7.dynf` |
| 23 | `C2S_SS_simulation3.dynf` | `C2S_SS_simulation8.dynf` |
| 37 | `C2S_SS_simulation4.dynf` | `C2S_SS_simulation9.dynf` |
| 53 | `C2S_SS_simulation5.dynf` | `C2S_SS_simulation10.dynf` |

The wrapper assigns the matching snapshot directory for each model using the existing snapshot resolver, including the `AM_C2S_SS_simulationN` convention. It overrides each worker's model/snapshot paths in memory. It never rewrites the family map or the five experiment sources. Consequently, even the Markov channel uses files 1–10 in this wrapper.

The next channel reuses these ten files only after all current worker processes have exited successfully with saved results and confirmed Aspen cleanup. Duplicate slots, shared snapshot directories, and plans exceeding ten runs per channel are rejected before starting processes. File locks also prevent another worker using this wrapper in the same repository from opening a reserved model. Keep files 1–10 closed in other Aspen sessions before starting: standalone runners and manually opened documents do not participate in the wrapper's locks.

The default Aspen folder is resolved through the existing settings. A manual model override supplies its parent directory; otherwise the existing `aspen_root_override` or `DISTILLATION_ASPEN_ROOT`/default root is used. `--aspen-root` selects a different folder explicitly. All required model files are checked before creating a batch or opening Aspen.

Aspen can return `DISP_E_EXCEPTION` (`-2147352567`, "A string value was expected") when `win32com` probes `openDocument` or `CloseDocument` as properties before supplying their arguments. Each worker temporarily adds that HRESULT to `win32com.client.dynamic.ERRORS_BAD_CONTEXT`, allowing pywin32 to recognize the callable method and pass the existing arguments. This context remains active through Aspen cleanup and is restored afterwards. The installed `dynamic.py`, experiment sources, and scientific settings remain unchanged; errors from actual method calls still propagate. The dispatch behavior is documented in the [pywin32 source](https://github.com/mhammond/pywin32/blob/main/com/win32com/client/dynamic.py).

If a batch failed with this COM lookup error before training, start a new batch after updating the wrapper. The source-hash check intentionally prevents resuming a batch with changed wrapper code. Earlier logs remain available.

## Run

Use the existing `rl-assisted-env` environment with Aspen Dynamics and `pywin32`. In a Conda-enabled terminal, activate it and inspect the plan:

```powershell
conda activate rl-assisted-env
python run_distillation_phase2_repeats.py --dry-run
```

Start the complete study from the repository directory:

```powershell
python run_distillation_phase2_repeats.py
```

If your terminal uses another Python, select the existing simulation environment explicitly:

```powershell
python run_distillation_phase2_repeats.py --python 'C:\Users\hamediaa\AppData\Local\PythonEnvs\rl-assisted-env\python.exe'
```

Optional selections follow the polymer wrapper's interface:

```powershell
python run_distillation_phase2_repeats.py --categories horizon --modes sg plain
python run_distillation_phase2_repeats.py --categories weights markov residual combined --modes sg
python run_distillation_phase2_repeats.py --aspen-root 'D:\AspenModels\Plant'
```

`--seeds 7 11 23 37 53`, `--threads-per-run 1`, and `--output-dir PATH` control replication and output settings. The default is one CPU/BLAS thread per worker. Device selection, Aspen visibility, controller settings, rewards, learning schedules, network sizes, disturbances, setpoints, and episode settings are retained from the existing runners. For standalone channels, the wrapper sets `agent_mode`; for the combined channel it sets `combined_agent_mode`. Combined's standalone parity checks retain matching agent seeds.

`--dry-run` imports only the standard library, displays the full slot assignment, and creates no batch. It does not check dependencies or proprietary model files. A real launch checks dependencies and model paths before starting training and never installs packages.

## Outputs and resume

Results are isolated under `Distillation/Results/phase2_repeats/<batch-id>/<channel>_<mode>/s<seed>/a<attempt>/`. Each attempt contains:

- `job.json`: seed, mode, Aspen number, model/snapshot paths, and source provenance;
- `run.log`: the worker's complete console output;
- `worker_status.json`: completion/failure and Aspen cleanup status;
- `rl/<timestamp>/input_data.pkl` and figures saved by the unchanged entrypoint;
- `rl/<timestamp>/replication_metadata.json`: replication settings and source metadata;
- the existing comparison output under `comparison/`, when produced.

The batch's `manifest.json` records every repetition, attempt, and result path. If a run fails, the other repetitions in that channel finish before the launcher stops. The next channel does not start.

Resume a stopped batch with:

```powershell
python run_distillation_phase2_repeats.py --resume 'Distillation\Results\phase2_repeats\<batch-id>'
```

Resume keeps saved runs, preserves the original Aspen numbers, and reruns failed/interrupted repetitions in new attempt directories. It restarts their complete 300-episode study. It checks source hashes and rejects batches marked running or changes to the saved channel/mode/seed/Aspen selections.

Worker cleanup also covers exceptions and partial Aspen initialization. Ctrl+C asks the launcher's own workers to exit and close their Aspen applications before terminating workers that remain unresponsive. After a forced stop or an Aspen cleanup error, close any remaining documents for files 1–10 before resuming.

## Verification

```powershell
python -m unittest discover -s tests -p test_distillation_phase2_repeats.py -v
```

The focused checks use real short-lived subprocesses and mocked Aspen objects. They verify ten simultaneous runs, sequential channels, unique models/snapshots, exclusive locks, failure/resume, preservation of actual scientific defaults, combined parity, and cleanup on normal completion, runner errors, and partial construction. They do not open Aspen or establish throughput for full training.
