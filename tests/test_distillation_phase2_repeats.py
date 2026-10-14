"""Scheduling, Aspen ownership, and configuration checks without opening Aspen."""

import ast
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_distillation_phase2_repeats as launcher


FAKE_WORKER = r'''
import json, pathlib, sys, time
sys.path.insert(0, sys.argv[2])
import run_distillation_phase2_repeats as launcher
job_file = pathlib.Path(sys.argv[1])
job = json.loads(job_file.read_text())
root = pathlib.Path(job['repo_root'])
group = root / job['category']
group.mkdir(exist_ok=True)
with launcher.aspen_file_lock(pathlib.Path(job['aspen_path']), root / 'locks'):
    started = time.monotonic_ns()
    (group / ('ready_' + job['mode'] + '_' + str(job['seed']))).touch()
    deadline = time.monotonic() + 10
    while len(list(group.glob('ready_*'))) < 10:
        if time.monotonic() > deadline:
            raise RuntimeError('All ten SG/plain workers were not started together')
        time.sleep(.01)
    time.sleep(.05)
    if (root / 'fail_horizon_sg_seed11').exists() and job['category'] == 'horizon' and job['mode'] == 'sg' and job['seed'] == 11:
        launcher.write_json(job_file.parent / 'worker_status.json', {'state':'failed','error':'injected failure'})
        sys.exit(1)
    result = job_file.parent / 'input_data.pkl'
    result.write_bytes(b'saved fixture result')
status = {'state':'completed','result_file':str(result),'aspen_closed':True,
          'started_ns':started,'finished_ns':time.monotonic_ns(),'aspen_number':job['aspen_number']}
if (root / 'fail_cleanup').exists() and job['category'] == 'horizon' and job['aspen_number'] == 1:
    status['aspen_closed'] = False
launcher.write_json(job_file.parent / 'worker_status.json', status)
'''


class LauncherTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / "_codex_tmp_distillation_phase2_wrapper_validation"
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(scratch.resolve()))
        self.aspen_root = self.root / "Plant"
        self.aspen_root.mkdir()
        for number in range(1, 11):
            (self.aspen_root / f"C2S_SS_simulation{number}.dynf").write_bytes(b'fixture Aspen model')
            (self.aspen_root / f"AM_C2S_SS_simulation{number}").mkdir()

    def tearDown(self):
        # TemporaryDirectory's absolute target is verified inside the scratch root above.
        self.temporary.cleanup()

    def plan(self, categories=None):
        jobs = launcher.build_jobs(categories or list(launcher.RUNNERS), ["sg", "plain"], launcher.DEFAULT_SEEDS)
        for job in jobs:
            number = job["aspen_number"]
            job.update(aspen_path=str(self.aspen_root / f"C2S_SS_simulation{number}.dynf"),
                       snaps_path=str(self.aspen_root / f"AM_C2S_SS_simulation{number}"))
        return jobs

    def make_manifest(self):
        worker = self.root / "fixture_worker.py"
        worker.write_text(FAKE_WORKER, encoding="utf-8")
        batch = self.root / "batch"
        batch.mkdir()
        manifest = {
            "repo_root": str(self.root), "python": sys.executable, "threads_per_run": 1,
            "source_hashes": {entry: "fixture-hash" for _, entry in launcher.RUNNERS.values()},
            "jobs": self.plan(["horizon", "weights"]),
        }
        return manifest, batch, lambda python, job: [python, str(worker), str(job), str(ROOT)]

    def test_default_plan_has_fifty_runs_and_exactly_slots_one_through_ten_per_channel(self):
        jobs = self.plan()
        self.assertEqual(len(jobs), 50)
        self.assertEqual(len({(j["category"], j["mode"], j["seed"]) for j in jobs}), 50)
        self.assertEqual(list(dict.fromkeys(j["category"] for j in jobs)), list(launcher.RUNNERS))
        for category in launcher.RUNNERS:
            group = [j for j in jobs if j["category"] == category]
            self.assertEqual([j["aspen_number"] for j in group], list(range(1, 11)))
            self.assertEqual([j["seed"] for j in group[:5]], list(launcher.DEFAULT_SEEDS))
            self.assertTrue(all(j["mode"] == "sg" for j in group[:5]))
            self.assertTrue(all(j["mode"] == "plain" for j in group[5:]))
            self.assertEqual(len({j["aspen_path"] for j in group}), 10)
            self.assertEqual(len({j["snaps_path"] for j in group}), 10)

    def test_invalid_plans_are_rejected_before_launching(self):
        for modes, seeds in ((["sg", "plain"], range(6)), (["sg", "sg"], [7]), (["sg"], [7, 7]),
                             (["sg"], [-1]), (["sg"], [2**32])):
            with self.assertRaises(ValueError):
                launcher.build_jobs(["horizon"], modes, seeds)
        manifest, batch, command = self.make_manifest()
        manifest["jobs"][1]["aspen_number"] = 1
        with patch.object(launcher.subprocess, "Popen") as spawn:
            with self.assertRaises(ValueError):
                launcher.run_batches(manifest, batch, command)
        spawn.assert_not_called()
        manifest["jobs"][1]["aspen_number"] = 2
        manifest["jobs"][1]["snaps_path"] = manifest["jobs"][0]["snaps_path"]
        with self.assertRaises(ValueError):
            launcher.validate_jobs(manifest["jobs"], require_paths=True)

    def assert_config_equal(self, actual, expected):
        import numpy as np
        if isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_config_equal(actual[key], expected[key])
        elif isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for a, e in zip(actual, expected):
                self.assert_config_equal(a, e)
        else:
            self.assertEqual(actual, expected)

    def test_actual_defaults_preserve_all_scientific_settings_for_both_modes(self):
        from systems.distillation import get_distillation_notebook_defaults
        for category, (family, _) in launcher.RUNNERS.items():
            original = get_distillation_notebook_defaults(family)
            before = deepcopy(original)
            for mode in ("sg", "plain"):
                job = next(j for j in self.plan() if j["category"] == category and j["mode"] == mode and j["seed"] == 53)
                job["results_directory"] = str(self.root)
                result = launcher.configure_defaults(original, job)
                expected = deepcopy(original)
                expected.update(run_mode="disturb", disturbance_profile="fluctuation", training_profile_override=launcher.PROFILE,
                                results_dir_override=str(self.root), result_prefix_override="rl", compare_prefix_override="comparison",
                                aspen_preset=job["aspen_number"], aspen_path_override=job["aspen_path"], snaps_path_override=job["snaps_path"])
                expected["combined_agent_mode" if category == "combined" else "agent_mode"] = mode
                for key in launcher.AGENT_CONFIG_KEYS:
                    if isinstance(expected.get(key), dict):
                        expected[key]["seed"] = 53
                self.assert_config_equal(result, expected)
            self.assert_config_equal(original, before)

    def test_actual_combined_standalone_parity_survives_each_seed(self):
        import numpy as np
        import systems.distillation as distillation
        import systems.distillation.plant as plant
        source = ast.parse((ROOT / launcher.RUNNERS["combined"][1]).read_text(encoding="utf-8"))
        nodes = [node for node in source.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("_assert_equal_config", "validate_standalone_parity")]
        self.assertEqual(len(nodes), 2)
        original_provider = distillation.get_distillation_notebook_defaults
        for job in [j for j in self.plan(["combined"])]:
            job["results_directory"] = str(self.root)
            with launcher.worker_adapters(distillation, plant, job):
                namespace = {"np": np, "get_distillation_notebook_defaults": distillation.get_distillation_notebook_defaults}
                exec(compile(ast.Module(body=nodes, type_ignores=[]), "combined_parity", "exec"), namespace)
                defaults = distillation.get_distillation_notebook_defaults("combined")
                namespace["validate_standalone_parity"](defaults)
                self.assertEqual(defaults["combined_agent_mode"], job["mode"])
        self.assertIs(distillation.get_distillation_notebook_defaults, original_provider)

    def test_paths_use_existing_root_but_replace_shared_manual_model_and_snapshots(self):
        import systems.distillation as distillation
        provider = lambda family: {"aspen_path_override": str(self.aspen_root / "C2S_SS_simulation11.dynf"),
                                   "snaps_path_override": str(self.aspen_root / "shared"), "n_tests_override": None}
        with patch.object(distillation, "get_distillation_notebook_defaults", provider):
            planned = launcher.resolve_job_paths(launcher.build_jobs(launcher.RUNNERS, ["sg", "plain"], launcher.DEFAULT_SEEDS))
        for job in planned:
            self.assertEqual(Path(job["aspen_path"]).parent, self.aspen_root)
            self.assertEqual(Path(job["snaps_path"]).name, f"AM_C2S_SS_simulation{job['aspen_number']}")
        launcher.check_aspen_files(planned)

    def test_ten_processes_overlap_and_files_are_reused_only_after_channel_save_barrier(self):
        manifest, batch, command = self.make_manifest()
        with redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.run_batches(manifest, batch, command), 0)
        intervals = {}
        paths = []
        for job in manifest["jobs"]:
            attempt = Path(job["attempts"][0]["directory"])
            status = json.loads((attempt / "worker_status.json").read_text())
            payload = json.loads((attempt / "job.json").read_text())
            self.assertEqual(payload["aspen_number"], job["aspen_number"])
            self.assertEqual(payload["aspen_path"], job["aspen_path"])
            intervals.setdefault(job["category"], []).append((status["started_ns"], status["finished_ns"]))
            paths.append(job["result_file"])
        for group in intervals.values():
            self.assertEqual(len(group), 10)
            self.assertLess(max(start for start, _ in group), min(end for _, end in group))
        self.assertLessEqual(max(end for _, end in intervals["horizon"]), min(start for start, _ in intervals["weights"]))
        self.assertEqual(len(set(paths)), 20)
        self.assertEqual(manifest["state"], "completed")

    def test_failure_stops_next_channel_and_resume_preserves_saved_runs_and_slots(self):
        manifest, batch, command = self.make_manifest()
        failure = self.root / "fail_horizon_sg_seed11"
        failure.touch()
        with redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.run_batches(manifest, batch, command), 1)
        self.assertTrue(all(not j["attempts"] for j in manifest["jobs"] if j["category"] == "weights"))
        saved = {(j["mode"], j["seed"]): j["result_file"] for j in manifest["jobs"] if j["state"] == "completed"}
        self.assertEqual(len(saved), 9)
        failure.unlink()
        with redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.run_batches(manifest, batch, command), 0)
        for job in manifest["jobs"][:10]:
            failed = job["mode"] == "sg" and job["seed"] == 11
            self.assertEqual(len(job["attempts"]), 2 if failed else 1)
            if not failed:
                self.assertEqual(job["result_file"], saved[(job["mode"], job["seed"])])
            for attempt in job["attempts"]:
                payload = json.loads((Path(attempt["directory"]) / "job.json").read_text())
                self.assertEqual(payload["aspen_number"], job["aspen_number"])
        self.assertEqual(manifest["state"], "completed")

    def test_missing_cleanup_confirmation_prevents_next_channel_even_if_result_exists(self):
        manifest, batch, command = self.make_manifest()
        (self.root / "fail_cleanup").touch()
        with redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.run_batches(manifest, batch, command), 1)
        self.assertEqual(manifest["state"], "failed")
        self.assertTrue(all(not j["attempts"] for j in manifest["jobs"] if j["category"] == "weights"))

    def test_file_lock_rejects_another_process_and_releases_after_cleanup(self):
        model = self.aspen_root / "C2S_SS_simulation1.dynf"
        locks = self.root / "locks"
        code = (
            "import sys; from pathlib import Path; import run_distillation_phase2_repeats as launcher; "
            "guard=launcher.aspen_file_lock(Path(sys.argv[1]),Path(sys.argv[2])); "
            "guard.__enter__(); guard.__exit__(None,None,None)"
        )
        with launcher.aspen_file_lock(model, locks):
            checked = subprocess.run([sys.executable, "-c", code, str(model), str(locks)], cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(checked.returncode, 0)
            self.assertIn("reserved by another wrapper worker", checked.stderr)
        checked = subprocess.run([sys.executable, "-c", code, str(model), str(locks)], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(checked.returncode, 0, checked.stderr)

    def test_process_creation_failure_reserves_attempt_without_overwriting_on_resume(self):
        manifest, batch, command = self.make_manifest()
        with redirect_stdout(io.StringIO()), patch.object(launcher.subprocess, "Popen", side_effect=OSError("injected spawn failure")):
            with self.assertRaises(OSError):
                launcher.run_batches(manifest, batch, command)
        self.assertEqual(manifest["state"], "interrupted")
        first_attempt = Path(manifest["jobs"][0]["attempts"][0]["directory"])
        self.assertTrue((first_attempt / "job.json").is_file())
        with redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.run_batches(manifest, batch, command), 0)
        self.assertEqual(len(manifest["jobs"][0]["attempts"]), 2)
        self.assertTrue((first_attempt / "job.json").is_file())

    def test_preflights_stop_before_creating_output_or_launching_workers(self):
        output = self.root / "not_created"
        for failing_preflight in ("dependency_check", "aspen_preflight"):
            with redirect_stdout(io.StringIO()), patch.object(launcher, "dependency_check"), patch.object(launcher, "aspen_preflight"), \
                    patch.object(launcher, failing_preflight, side_effect=RuntimeError("injected preflight failure")), \
                    patch.object(launcher, "run_batches") as run:
                with self.assertRaises(RuntimeError):
                    launcher.main(["--output-dir", str(output)])
            run.assert_not_called()
            self.assertFalse(output.exists())
        jobs = self.plan()
        jobs[0]["aspen_path"] = str(self.root / "missing" / "C2S_SS_simulation1.dynf")
        with self.assertRaises(FileNotFoundError):
            launcher.check_aspen_files(jobs)

    def fake_worker_modules(self, fail_constructor=False, fail_cleanup=False):
        columns = []

        class FakeColumn:
            def __init__(self, path, **kwargs):
                self.ad = object()
                self.path = str(path)
                self.close_calls = []
                columns.append(self)
                if fail_constructor:
                    raise RuntimeError("injected constructor failure")

            def close(self, snaps_path=None, prefix="snp"):
                self.close_calls.append(str(snaps_path))
                if fail_cleanup:
                    raise RuntimeError("injected cleanup failure")

        distillation = types.ModuleType("systems.distillation")
        distillation.__path__ = []
        provider = lambda family: {"n_tests_override": None, "controller": {"z_bound": .7},
                                   "agent": {"gamma": .99}, "horizon_agent": {"gamma": .99}}
        distillation.get_distillation_notebook_defaults = provider
        distillation.DistillationColumnAspen = FakeColumn
        plant = types.ModuleType("systems.distillation.plant")
        plant.DistillationColumnAspen = FakeColumn
        plant.build_distillation_system = lambda **kwargs: plant.DistillationColumnAspen(**kwargs)
        distillation.plant = plant
        systems = types.ModuleType("systems")
        systems.__path__ = []
        systems.distillation = distillation
        seed_calls = []
        numpy = types.SimpleNamespace(random=types.SimpleNamespace(seed=lambda value: seed_calls.append(value)))
        torch = types.SimpleNamespace(manual_seed=lambda value: seed_calls.append(value), set_num_threads=lambda value: None,
                                      cuda=types.SimpleNamespace(is_available=lambda: False))
        return {"systems": systems, "systems.distillation": distillation, "systems.distillation.plant": plant,
                "numpy": numpy, "torch": torch}, columns, seed_calls, provider, FakeColumn

    def run_fixture(self, category="combined", fail_run=False, fail_constructor=False, fail_cleanup=False):
        family, entry = launcher.RUNNERS[category]
        program = self.root / entry
        program.write_text(
            "from pathlib import Path\nfrom systems.distillation import get_distillation_notebook_defaults\n"
            "from systems.distillation.plant import build_distillation_system\n"
            f"NB=get_distillation_notebook_defaults({family!r})\n"
            "TRAINING_PROFILE_NAME=NB['training_profile_override']\n"
            "COMBINED_AGENT_MODE=NB.get('combined_agent_mode')\nAGENT_MODE=NB.get('agent_mode')\n"
            "DYN_PATH=NB['aspen_path_override']\nSNAPS_PATH=NB['snaps_path_override']\n"
            "system=build_distillation_system(path=DYN_PATH)\n"
            + ("raise RuntimeError('injected runner failure')\n" if fail_run else "")
            + "result_bundle={'avg_rewards':[0.]*300}\n"
            "out_dir_rl=Path(NB['results_dir_override'])/'rl'\n"
            "out_dir_rl.mkdir()\n(out_dir_rl/'input_data.pkl').write_bytes(b'fixture')\n"
            "out_dir_cmp=None\n"
            # Some real entrypoints swallow close errors; the wrapper must detect them.
            "try:\n    system.close(SNAPS_PATH)\nexcept Exception:\n    pass\n",
            encoding="utf-8",
        )
        before = program.read_bytes()
        attempt = self.root / "attempt"
        attempt.mkdir()
        job = next(j for j in self.plan([category]) if j["mode"] == "plain" and j["seed"] == 37)
        job.update(threads_per_run=1, repo_root=str(self.root), results_directory=str(attempt),
                   batch_directory=str(self.root), entrypoint_sha256="fixture-hash")
        job_file = attempt / "job.json"
        launcher.write_json(job_file, job)
        modules, columns, seed_calls, provider, original_column = self.fake_worker_modules(fail_constructor, fail_cleanup)
        old_cwd, old_path = Path.cwd(), list(sys.path)
        old_signal = signal_handler = None
        if hasattr(launcher.signal, "SIGBREAK"):
            old_signal = launcher.signal.getsignal(launcher.signal.SIGBREAK)
            signal_handler = launcher.signal.SIGBREAK
        try:
            with patch.dict(sys.modules, modules), redirect_stderr(io.StringIO()):
                result = launcher.run_worker(job_file)
        finally:
            os.chdir(old_cwd)
            sys.path[:] = old_path
            if signal_handler is not None:
                launcher.signal.signal(signal_handler, old_signal)
        self.assertEqual(program.read_bytes(), before)
        self.assertIs(modules["systems.distillation"].get_distillation_notebook_defaults, provider)
        self.assertIs(modules["systems.distillation.plant"].DistillationColumnAspen, original_column)
        return result, attempt, columns, seed_calls

    def test_worker_executes_unchanged_entrypoint_seeds_and_closes_its_assigned_aspen_once(self):
        result, attempt, columns, seed_calls = self.run_fixture()
        self.assertEqual(result, 0)
        self.assertEqual(seed_calls, [37, 37])
        self.assertEqual(len(columns), 1)
        self.assertEqual(len(columns[0].close_calls), 1)
        self.assertEqual(Path(columns[0].path).name, "C2S_SS_simulation9.dynf")
        self.assertEqual(Path(columns[0].close_calls[0]).name, "AM_C2S_SS_simulation9")
        metadata = json.loads((attempt / "rl" / "replication_metadata.json").read_text())
        self.assertEqual(metadata["seed"], 37)
        self.assertEqual(metadata["episodes"], 300)
        self.assertEqual(metadata["mode"], "plain")
        self.assertEqual(metadata["aspen_number"], 9)
        self.assertTrue(launcher.saved_status(attempt, 0)["aspen_closed"])

    def test_worker_closes_aspen_on_runner_failure(self):
        result, attempt, columns, _ = self.run_fixture(fail_run=True)
        self.assertEqual(result, 1)
        self.assertEqual(len(columns[0].close_calls), 1)
        self.assertEqual(json.loads((attempt / "worker_status.json").read_text())["state"], "failed")

    def test_worker_closes_partial_aspen_construction(self):
        result, _, columns, _ = self.run_fixture(fail_constructor=True)
        self.assertEqual(result, 1)
        self.assertEqual(len(columns[0].close_calls), 1)

    def test_worker_cleanup_failure_cannot_be_reported_completed(self):
        result, attempt, _, _ = self.run_fixture(fail_cleanup=True)
        self.assertEqual(result, 1)
        self.assertEqual(launcher.saved_status(attempt, 0), {})
        status = json.loads((attempt / "worker_status.json").read_text())
        self.assertIn("Aspen cleanup failed", status["error"])

    def test_plain_standalone_worker_sets_agent_mode(self):
        result, attempt, _, _ = self.run_fixture(category="horizon")
        self.assertEqual(result, 0)
        self.assertEqual(json.loads((attempt / "rl" / "replication_metadata.json").read_text())["mode"], "plain")

    def test_dry_run_is_dependency_free_has_ten_slots_and_creates_no_batch(self):
        output = self.root / "not_created"
        with redirect_stdout(io.StringIO()) as captured, patch.object(launcher, "dependency_check") as dependencies, \
                patch.object(launcher, "aspen_preflight") as preflight:
            self.assertEqual(launcher.main(["--dry-run", "--output-dir", str(output)]), 0)
        dependencies.assert_not_called()
        preflight.assert_not_called()
        text = captured.getvalue()
        self.assertIn("Total repetitions: 50", text)
        for category in launcher.RUNNERS:
            self.assertIn(f"{category}: 10 parallel repetitions", text)
        self.assertEqual(text.count("C2S_SS_simulation1.dynf"), 5)
        self.assertEqual(text.count("C2S_SS_simulation10.dynf"), 5)
        self.assertNotIn("simulation11", text)
        self.assertFalse(output.exists())
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            launcher.main(["--dry-run", "--seeds", "1", "2", "3", "4", "5", "6"])


if __name__ == "__main__":
    unittest.main()
