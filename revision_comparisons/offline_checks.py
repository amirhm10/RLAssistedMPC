"""Compressed technical checks using the identified linear model as a fake plant.

No Aspen instance is opened. These checks exercise the real entrypoint, runner,
networks, calibration and exports. Their results are not scientific trajectories.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from pathlib import Path
import pickle
import types
from unittest.mock import patch

import numpy as np

from comparison_studies.common import compress_bundle, smoke_runner, write_json
from revision_comparisons.column import ROOT, run_method


def run(output):
    import torch
    import systems.distillation as exported
    import systems.distillation.plant as plants
    import utils.mpc_baseline_runner as baseline_module
    from comparison_studies.checkpoints import save_inference_checkpoint
    from review_experiments.e3.runtime import resolve_aspen
    with (ROOT/"Distillation/Data/mpc_results_disturb_fluctuation.pickle").open("rb") as stream:
        saved = pickle.load(stream)
    with (ROOT/"Distillation/Data/system_dict.pickle").open("rb") as stream:
        model = pickle.load(stream)
    lo, hi = saved["data_min"], saved["data_max"]

    class OfflineColumn:
        created = 0
        closed = 0
        def __init__(self, path, ss_inputs, initialization_point, delta_t=1/6, visible=True):
            type(self).created += 1
            self.ss_inputs = np.asarray(ss_inputs).copy()
            self.y_ss = np.asarray(saved["steady_states"]["y_ss"]).copy()
            self.current_input, self.current_output = self.ss_inputs.copy(), self.y_ss.copy()
            self.delta_t, self.x = float(delta_t), np.zeros(model["A"].shape[0])
            self.feed = types.SimpleNamespace(FmR=types.SimpleNamespace(Value=float(initialization_point[0])))
            self._closed = False
        def step(self, disturbances=None):
            u = (self.current_input-self.ss_inputs)/(hi[:2]-lo[:2])
            self.x = model["A"]@self.x+model["B"]@u
            self.current_output = self.y_ss+(model["C"]@self.x)*(hi[2:]-lo[2:])
            if disturbances is not None:
                self.feed.FmR.Value = float(np.atleast_1d(disturbances)[0])
        def close(self, *args, **kwargs):
            if not self._closed:
                type(self).closed += 1
                self._closed = True

    instances = []
    original_init = OfflineColumn.__init__
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        instances.append(self)
    OfflineColumn.__init__ = initialize

    @contextmanager
    def offline_resources(paths):
        start = len(instances)
        with patch.object(plants, "DistillationColumnAspen", OfflineColumn), \
             patch.object(exported, "DistillationColumnAspen", OfflineColumn):
            try:
                yield
            finally:
                for instance in instances[start:]:
                    instance.close()

    original_resolve = baseline_module.resolve_episode_bundle
    def compressed_resolve(*args, **kwargs):
        return compress_bundle(original_resolve(*args, **kwargs))
    output.mkdir(parents=True, exist_ok=False)
    aspen = resolve_aspen("combined", number_override=12)  # inspect assigned path/hash only
    rows = []
    torch.set_num_threads(1)
    with ExitStack() as stack:
        stack.enter_context(patch("torch.cuda.is_available", return_value=False))
        stack.enter_context(patch("review_experiments.e3.runtime.aspen_resources", offline_resources))
        stack.enter_context(patch.object(baseline_module, "resolve_episode_bundle", compressed_resolve))
        stack.enter_context(smoke_runner("markov"))
        stack.enter_context(smoke_runner("combined"))
        for method in ("nominal", "ls", "bo", "threshold", "sg"):
            directory = output/method
            directory.mkdir()
            result, agents = run_method(method, directory, aspen, 7, bo_trials=5)
            if not all((directory / name).is_file() for name in ("baseline_audit.json", "baseline_metrics.json", "resolved_defaults.json")):
                raise AssertionError(f"{method} omitted mandatory completed-worker sidecars.")
            if result["nFE"] != 400 or len(result["avg_rewards"]) != 200:
                raise AssertionError("Technical clock did not preserve 200 episode boundaries.")
            if not all(np.all(np.isfinite(result[k])) for k in ("y", "u", "rewards_step")):
                raise AssertionError("Nonfinite offline interface-check trajectory.")
            if not result["test_train_dict"][max(result["test_train_dict"])]:
                raise AssertionError("Final evaluation cycle was lost.")
            row = {"method": method, "steps": 400, "cycles": 200,
                   "original_architectures": True, "live_aspen_opened": False,
                   "scientific_performance_result": False}
            if agents is not None:
                if set(agents) != {"horizon_agent", "weight_agent", "markov_agent", "residual_agent"}:
                    raise AssertionError("Missing combined agent.")
                row["checkpoint"] = save_inference_checkpoint(directory/"inference.pt", agents,
                    metadata={"technical_smoke": True, "plant": "linear_stand_in_for_column"})
            rows.append(row)
            print("OFFLINE CHECK PASSED: " + method, flush=True)
    if OfflineColumn.created != OfflineColumn.closed:
        raise AssertionError("Offline resource ownership/cleanup was not balanced.")
    write_json(output/"summary.json", {"state": "completed", "methods": rows,
        "owned_fake_cases_created": OfflineColumn.created, "owned_fake_cases_closed": OfflineColumn.closed,
        "live_aspen_opened": False, "training_clock_stride": 200,
        "scope": "Original column entrypoints and networks on compressed clock with an identified-model fake plant; no plant performance evidence."})
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args().output.resolve()))
