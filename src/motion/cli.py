import argparse
import csv
import json
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

import numpy as np

from motion.simulation import Config, ROOT, Simulation

COLUMNS = (["time"] + [f"tcp_{a}" for a in "xyz"] + [f"target_{a}" for a in "xyz"]
           + [f"contact_f{a}" for a in "xyz"] + [f"wrist_{a}" for a in ("fx", "fy", "fz", "tx", "ty", "tz")]
           + [f"q{i}" for i in range(7)] + [f"dq{i}" for i in range(7)]
           + [f"tau{i}" for i in range(7)])


def run(config, output, viewer=False):
    sim = Simulation(config)
    handle = None
    if viewer:
        import mujoco.viewer
        handle = mujoco.viewer.launch_passive(sim.model, sim.data)
        handle.cam.lookat[:] = sim.initial
        handle.cam.distance = 1.5
    rows = []
    started = time.perf_counter()
    try:
        for _ in range(round(config.duration / config.timestep)):
            if handle is not None and not handle.is_running():
                raise RuntimeError("viewer closed before experiment completed")
            rows.append(sim.step())
            if handle is not None:
                handle.sync()
                delay = started + sim.data.time - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
    finally:
        if handle is not None:
            handle.close()
    rows = np.asarray(rows)
    steady = rows[rows[:, 0] >= config.duration - 1]
    metrics = {
        "steps": len(rows), "simulated_seconds": float(sim.data.time),
        "force_mean_N": float(steady[:, 9].mean()),
        "force_rmse_N": float(np.sqrt(np.mean((steady[:, 9] - config.target_force) ** 2))),
        "force_peak_N": float(rows[:, 9].max()),
        "contact_fraction_last_second": float(np.mean(steady[:, 9] > 0.5)),
        "final_xy_error_m": float(np.linalg.norm(rows[-1, 1:3] - rows[-1, 4:6])),
        "x_displacement_m": float(rows[-1, 1] - rows[0, 1]),
    }
    metrics["passed"] = bool(metrics["force_rmse_N"] < 1.5
                              and metrics["force_peak_N"] < 2 * config.target_force + 5
                              and metrics["contact_fraction_last_second"] > 0.95
                              and metrics["final_xy_error_m"] < 0.005)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "trace.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    metadata = sim.metadata()
    # Source archives/minimal Docker images may have no Git executable or metadata.
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
        metadata["git_commit"] = commit.stdout.strip() if commit.returncode == 0 else None
        metadata["git_status"] = status.stdout.splitlines() if status.returncode == 0 else None
    except FileNotFoundError:
        metadata["git_commit"] = None
        metadata["git_status"] = None
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Panda contact-control baseline")
    parser.add_argument("--task", choices=["press", "slide"], default="press")
    parser.add_argument("--duration", type=float, default=8)
    parser.add_argument("--timestep", type=float, default=0.001)
    parser.add_argument("--force", type=float, default=10)
    parser.add_argument("--distance", type=float, default=0.06)
    parser.add_argument("--friction", type=float, default=0.3)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = Config(task=args.task, duration=args.duration, timestep=args.timestep,
                 target_force=args.force, slide_distance=args.distance, friction=args.friction)
    output = args.output or ROOT / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    metrics = run(cfg, output, args.viewer)
    print(json.dumps({"output": str(output), **metrics}, indent=2))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
