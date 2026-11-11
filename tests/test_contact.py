import json
import numpy as np
import pytest

from motion.cli import run
from motion.simulation import Config, Simulation


@pytest.mark.parametrize("task,force", [("press", 5), ("press", 10), ("slide", 10)])
def test_contact_tracking(tmp_path, task, force):
    output = tmp_path / task
    metrics = run(Config(task=task, target_force=force), output)
    assert metrics["passed"], metrics
    assert metrics["steps"] == 8000
    assert abs(metrics["force_mean_N"] - force) < 0.5
    if task == "slide":
        assert metrics["x_displacement_m"] > 0.055
    trace = np.loadtxt(output / "trace.csv", delimiter=",", skiprows=1)
    assert np.isfinite(trace).all()
    assert np.all(np.abs(trace[:, -7:]) <= np.array([87, 87, 87, 87, 12, 12, 12]) + 1e-9)
    assert json.loads((output / "metadata.json").read_text())["config"]["task"] == task


def test_step_size_sensitivity(tmp_path):
    coarse = run(Config(task="slide"), tmp_path / "coarse")
    fine = run(Config(task="slide", timestep=0.0005), tmp_path / "fine")
    assert coarse["passed"] and fine["passed"]
    assert abs(coarse["force_mean_N"] - fine["force_mean_N"]) < 0.2
    assert abs(coarse["x_displacement_m"] - fine["x_displacement_m"]) < 0.001
    assert abs(coarse["force_peak_N"] - fine["force_peak_N"]) < 2


@pytest.mark.parametrize("changes", [{"target_force": -1}, {"timestep": 0},
                                      {"duration": float("nan")}, {"friction": -1}])
def test_reject_invalid_physics(changes):
    with pytest.raises(ValueError):
        Simulation(Config(**changes))


def test_archive_without_git(tmp_path, monkeypatch):
    # Match the user's minimal local Docker image, where Git is not installed.
    monkeypatch.setenv("PATH", "")
    output = tmp_path / "without-git"
    metrics = run(Config(), output)
    assert metrics["passed"]
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["git_commit"] is None
    assert metadata["git_status"] is None
