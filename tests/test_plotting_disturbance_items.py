from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def run_direct():
    test_disturbance_plot_items_ignores_string_profile_name()
    test_disturbance_plot_items_accepts_1d_schedule()
    test_disturbance_plot_items_accepts_2d_schedule()
    print("plotting disturbance item tests passed")


if __name__ == "__main__":
    run_direct()
