from __future__ import annotations

import pickle
import pathlib
import shutil
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import utils.plotting_core as plotting_core
from utils.plotting_core import disturbance_plot_items


def test_disturbance_plot_items_ignores_string_profile_name():
    assert disturbance_plot_items("fluctuation", ["Feed flow"]) == []


def test_disturbance_plot_items_accepts_1d_schedule():
    items = disturbance_plot_items(np.asarray([1.0, 2.0, 3.0]), ["Feed flow"])

    assert len(items) == 1
    key, label, series = items[0]
    assert key == "disturbance"
    assert label == "Feed flow"
    np.testing.assert_allclose(series, [1.0, 2.0, 3.0])


def test_disturbance_plot_items_accepts_2d_schedule():
    schedule = np.asarray([[1.0, 10.0], [2.0, 20.0]])
    items = disturbance_plot_items(schedule, ["d_a", "d_b"])

    assert [item[0] for item in items] == ["disturbance_1", "disturbance_2"]
    assert [item[1] for item in items] == ["d_a", "d_b"]
    np.testing.assert_allclose(items[0][2], [1.0, 2.0])
    np.testing.assert_allclose(items[1][2], [10.0, 20.0])


def test_combined_plotter_saves_input_bundle_before_figure_generation_failure():
    n_fe = 4
    y = np.array(
        [
            [5.0, 5.0],
            [5.1, 4.9],
            [5.2, 4.8],
            [5.3, 4.7],
            [5.4, 4.6],
        ],
        dtype=float,
    )
    u = np.array(
        [
            [2.0, 3.0],
            [2.1, 3.1],
            [2.2, 3.2],
            [2.3, 3.3],
        ],
        dtype=float,
    )
    bundle = {
        "y": y,
        "u": u,
        "avg_rewards": [1.0],
        "data_min": np.zeros(4),
        "data_max": np.ones(4) * 10.0,
        "nFE": n_fe,
        "delta_t": 1.0,
        "time_in_sub_episodes": n_fe,
        "y_sp": np.zeros((n_fe, 2)),
        "steady_states": {"y_ss": np.array([5.0, 5.0])},
        "active_agents": {},
        "system_metadata": {
            "output_labels": ["y1", "y2"],
            "input_labels": ["u1", "u2"],
            "time_label": "t",
        },
        "method_family": "combined",
    }

    original_save_fig = plotting_core._save_fig

    def fail_first_figure(*args, **kwargs):
        raise RuntimeError("forced figure failure")

    tmp_root = ROOT / "tests" / "_plot_tmp_combined"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir()
    try:
        plotting_core._save_fig = fail_first_figure
        try:
            try:
                plotting_core.plot_combined_results_core(
                    bundle,
                    {
                        "directory": tmp_root,
                        "prefix_name": "combined_smoke",
                        "include_baseline_compare": False,
                    },
                )
            except RuntimeError as exc:
                assert "forced figure failure" in str(exc)
            else:
                raise AssertionError("plot_combined_results_core should have failed during figure generation")
        finally:
            plotting_core._save_fig = original_save_fig

        out_dirs = list((tmp_root / "combined_smoke").iterdir())
        assert len(out_dirs) == 1
        input_data_path = out_dirs[0] / "input_data.pkl"
        assert input_data_path.exists()
        with input_data_path.open("rb") as handle:
            stored = pickle.load(handle)
        assert stored["method_family"] == "combined"
        np.testing.assert_allclose(stored["y"], y)
        np.testing.assert_allclose(stored["u"], u)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def run_direct():
    test_disturbance_plot_items_ignores_string_profile_name()
    test_disturbance_plot_items_accepts_1d_schedule()
    test_disturbance_plot_items_accepts_2d_schedule()
    test_combined_plotter_saves_input_bundle_before_figure_generation_failure()
    print("plotting disturbance item tests passed")


if __name__ == "__main__":
    run_direct()
