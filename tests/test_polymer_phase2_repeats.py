"""Launcher contracts, using real short-lived processes instead of training."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_polymer_phase2_repeats as launcher


FAKE_WORKER = r'''
import json, pathlib, sys, time
job_file = pathlib.Path(sys.argv[1])
job = json.loads(job_file.read_text())
root = pathlib.Path(job['repo_root'])
group = root / job['category']
group.mkdir(exist_ok=True)
started = time.monotonic_ns()
(group / ('ready_' + job['mode'] + '_' + str(job['seed']))).touch()
deadline = time.monotonic() + 10
while len(list(group.glob('ready_*'))) < 10:
    if time.monotonic() > deadline:
        raise RuntimeError('All ten gated and ungated workers were not started together')
    time.sleep(.01)
time.sleep(.05)
failure = root / 'fail_horizon_sg_seed11'
if failure.exists() and job['category'] == 'horizon' and job['mode'] == 'sg' and job['seed'] == 11:
    (job_file.parent / 'worker_status.json').write_text(json.dumps({'state':'failed','error':'injected failure'}))
    sys.exit(1)
result = job_file.parent / 'input_data.pkl'
result.write_bytes(b'saved fixture result')
status = {'state':'completed','result_file':str(result),'started_ns':started,'finished_ns':time.monotonic_ns()}
(job_file.parent / 'worker_status.json').write_text(json.dumps(status))
'''


class LauncherTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / "_codex_tmp_phase2_wrapper_validation"
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(scratch.resolve()))

    def tearDown(self):
        self.temporary.cleanup()

    def make_manifest(self):
        worker = self.root / "fixture_worker.py"
        worker.write_text(FAKE_WORKER, encoding="utf-8")
        batch = self.root / "batch"
        batch.mkdir()
        manifest = {
            "repo_root": str(self.root), "python": sys.executable, "threads_per_run": 1,
            "source_hashes": {entry: "fixture-hash" for _, entry in launcher.RUNNERS.values()},
            "jobs": launcher.build_jobs(["horizon", "weights"], ["sg", "plain"], launcher.DEFAULT_SEEDS),
        }
        return manifest, batch, lambda python, job: [python, str(worker), str(job)]

    def test_default_plan_has_fifty_unique_runs_in_five_category_groups(self):
        jobs = launcher.build_jobs(launcher.RUNNERS, ["sg", "plain"], launcher.DEFAULT_SEEDS)
        self.assertEqual(len(jobs), 50)
        self.assertEqual(len({(j["category"], j["mode"], j["seed"]) for j in jobs}), 50)
        self.assertEqual([j["seed"] for j in jobs[:5]], list(launcher.DEFAULT_SEEDS))
        self.assertTrue(all(j["category"] == "horizon" and j["mode"] == "plain" for j in jobs[5:10]))
        self.assertEqual(len({j["category"] for j in jobs}), 5)
        for category in launcher.RUNNERS:
            group = [j for j in jobs if j["category"] == category]
            self.assertEqual(len(group), 10)
            self.assertEqual({j["mode"] for j in group}, {"sg", "plain"})

    def test_configuration_preserves_controller_learning_and_global_defaults(self):
        original = {
            "run_mode": "disturb", "n_tests_override": None,
            "episode_defaults": {"n_tests": 300, "warm_start": 10, "set_points_len": 400},
            "controller": {"z_bound": .7, "Q1_penalty": 5., "decision_interval": 4},
            "reward": {"movement_weight": 1.}, "supervisor_gate": {"advantage_margin": .5},
            "agent": {"gamma": .99, "hidden_layers": [512]*5},
            "td3_agent": {"gamma": .99, "actor_hidden": [512]*5, "seed": 7},
        }
        before = deepcopy(original)
        for category in launcher.RUNNERS:
            for mode in ("sg", "plain"):
                result = launcher.configure_defaults(original, {
                    "category": category, "mode": mode, "seed": 53,
                    "results_directory": str(self.root),
                })
                for key in ("controller", "episode_defaults", "reward", "supervisor_gate"):
                    self.assertEqual(result[key], original[key])
                self.assertEqual(result["agent"]["hidden_layers"], original["agent"]["hidden_layers"])
                self.assertEqual(result["td3_agent"]["gamma"], .99)
                self.assertEqual(result["td3_agent"]["seed"], 53)
                if category == "combined":
                    self.assertEqual(result["combined_agent_mode"], mode)
                else:
                    expected = ("sg_dqn" if mode == "sg" else "dqn") if category == "horizon" else ("sg_td3" if mode == "sg" else "td3")
                    self.assertEqual(result["agent_kind"], expected)
        self.assertEqual(original, before)
        with self.assertRaises(ValueError):
            launcher.configure_defaults({"n_tests_override": 20}, {"category":"horizon", "mode":"sg"})

    def test_ten_processes_overlap_and_next_category_waits_for_saved_results(self):
        manifest, batch, command = self.make_manifest()
        with redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.run_batches(manifest, batch, command), 0)
        self.assertEqual(manifest["state"], "completed")
        intervals = {}
        paths = []
        for job in manifest["jobs"]:
            status = json.loads((Path(job["attempts"][0]["directory"]) / "worker_status.json").read_text())
            intervals.setdefault(job["category"], []).append((status["started_ns"], status["finished_ns"]))
            paths.append(job["result_file"])
        for group in intervals.values():
            self.assertEqual(len(group), 10)
            self.assertLess(max(start for start, _ in group), min(end for _, end in group))
        self.assertLessEqual(max(end for _, end in intervals["horizon"]), min(start for start, _ in intervals["weights"]))
        self.assertEqual(len(set(paths)), 20)

    def test_failure_stops_next_category_and_resume_keeps_completed_runs(self):
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
        self.assertEqual(manifest["state"], "completed")

    def test_process_creation_failure_reserves_attempt_without_overwriting_on_resume(self):
        manifest, batch, command = self.make_manifest()
        with redirect_stdout(io.StringIO()):
            with patch.object(launcher.subprocess, "Popen", side_effect=OSError("injected spawn failure")):
                with self.assertRaises(OSError):
                    launcher.run_batches(manifest, batch, command)
        self.assertEqual(manifest["state"], "interrupted")
        self.assertEqual(len(manifest["jobs"][0]["attempts"]), 1)
        first_attempt = Path(manifest["jobs"][0]["attempts"][0]["directory"])
        self.assertTrue((first_attempt / "job.json").is_file())
        with redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.run_batches(manifest, batch, command), 0)
        self.assertEqual(len(manifest["jobs"][0]["attempts"]), 2)
        self.assertTrue((first_attempt / "job.json").is_file())

    def test_missing_dependencies_stop_before_creating_output_or_launching_workers(self):
        output = self.root / "not_created"
        with redirect_stdout(io.StringIO()), patch.object(launcher, "dependency_check", side_effect=RuntimeError("missing torch")), patch.object(launcher, "run_batches") as run:
            with self.assertRaises(RuntimeError):
                launcher.main(["--output-dir", str(output)])
        run.assert_not_called()
        self.assertFalse(output.exists())

    def test_worker_executes_an_unchanged_entrypoint_and_records_the_seed(self):
        category = "combined"
        family, entry = launcher.RUNNERS[category]
        program = self.root / entry
        program.write_text(
            "from pathlib import Path\nfrom systems.polymer import get_polymer_notebook_defaults\n"
            f"NB=get_polymer_notebook_defaults({family!r})\n"
            "TRAINING_PROFILE_NAME=NB['training_profile_override']\n"
            "result_bundle={'avg_rewards':[0.]*300}\n"
            "out_dir_rl=Path(NB['results_dir_override'])/'rl'\n"
            "out_dir_rl.mkdir()\n(out_dir_rl/'input_data.pkl').write_bytes(b'fixture')\n"
            "out_dir_cmp=None\n",
            encoding="utf-8",
        )
        before = program.read_bytes()
        defaults = {"n_tests_override":None, "controller":{"z_bound":.7}, "horizon_agent":{"gamma":.99}}
        original_provider = lambda requested: deepcopy(defaults)
        polymer = types.ModuleType("systems.polymer")
        polymer.get_polymer_notebook_defaults = original_provider
        systems = types.ModuleType("systems")
        systems.__path__ = []
        systems.polymer = polymer
        seed_calls = []
        numpy = types.SimpleNamespace(random=types.SimpleNamespace(seed=lambda value:seed_calls.append(value)))
        torch = types.SimpleNamespace(manual_seed=lambda value:seed_calls.append(value),
                                      set_num_threads=lambda value:None,
                                      cuda=types.SimpleNamespace(is_available=lambda:False))
        attempt = self.root / "attempt"
        attempt.mkdir()
        job_file = attempt / "job.json"
        launcher.write_json(job_file, {
            "category":category, "mode":"sg", "seed":37, "threads_per_run":1,
            "repo_root":str(self.root), "results_directory":str(attempt),
            "batch_directory":str(self.root), "entrypoint_sha256":"fixture-hash",
        })
        old_cwd, old_path = Path.cwd(), list(sys.path)
        try:
            with patch.dict(sys.modules, {"systems":systems, "systems.polymer":polymer, "numpy":numpy, "torch":torch}):
                self.assertEqual(launcher.run_worker(job_file), 0)
        finally:
            import os
            os.chdir(old_cwd)
            sys.path[:] = old_path
        self.assertEqual(program.read_bytes(), before)
        self.assertIs(polymer.get_polymer_notebook_defaults, original_provider)
        metadata = json.loads((attempt / "rl" / "replication_metadata.json").read_text())
        self.assertEqual(metadata["seed"], 37)
        self.assertEqual(metadata["episodes"], 300)
        self.assertEqual(seed_calls, [37,37])

    def test_dry_run_is_dependency_free_and_creates_no_batch(self):
        output = self.root / "not_created"
        with redirect_stdout(io.StringIO()) as captured:
            self.assertEqual(launcher.main(["--dry-run", "--output-dir", str(output)]), 0)
        self.assertIn("Total repetitions: 50", captured.getvalue())
        for category in launcher.RUNNERS:
            self.assertIn(f"{category}: 10 parallel repetitions", captured.getvalue())
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
