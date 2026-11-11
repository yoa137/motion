"""SI units; world +Z is up. Contact force is the force ON the tool."""
from dataclasses import asdict, dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    task: str = "press"
    duration: float = 8.0
    timestep: float = 0.001
    target_force: float = 10.0
    slide_distance: float = 0.06
    friction: float = 0.3
    seed: int = 0

    def validate(self):
        if self.task not in ("press", "slide"):
            raise ValueError("task must be press or slide")
        values = (self.duration, self.timestep, self.target_force,
                  self.slide_distance, self.friction)
        if not all(np.isfinite(v) for v in values):
            raise ValueError("configuration values must be finite")
        if self.duration < 4 or not 0.0001 <= self.timestep <= 0.002:
            raise ValueError("duration >= 4 s; timestep in [0.0001, 0.002] s")
        if not 0 < self.target_force <= 20:
            raise ValueError("target force must be in (0, 20] N")
        if not 0 <= self.slide_distance <= 0.1 or self.friction < 0:
            raise ValueError("slide distance in [0, 0.1] m; friction >= 0")


class Simulation:
    def __init__(self, config: Config):
        config.validate()
        self.config = config
        model_dir = ROOT / "assets/panda"
        xml = ET.parse(model_dir / "panda_nohand.xml").getroot()
        xml.find("compiler").set("meshdir", str(model_dir / "assets"))
        option = xml.find("option")
        option.set("timestep", str(config.timestep))
        option.set("iterations", "100")
        option.set("tolerance", "1e-10")
        # Replace position servos with explicitly bounded torque motors.
        actuators = xml.find("actuator")
        actuators.clear()
        for i, limit in enumerate([87, 87, 87, 87, 12, 12, 12], 1):
            ET.SubElement(actuators, "motor", name=f"torque{i}",
                          joint=f"joint{i}", ctrlrange=f"{-limit} {limit}",
                          ctrllimited="true", gear="1")
        xml.find("keyframe/key").set("ctrl", "0 0 0 0 0 0 0")
        attachment = xml.find(".//body[@name='attachment']")
        tool = ET.SubElement(attachment, "body", name="tool")
        ET.SubElement(tool, "site", name="wrist", size="0.006")
        ET.SubElement(tool, "geom", type="capsule", size="0.006",
                      fromto="0 0 0 0 0 0.10", mass="0.1",
                      contype="0", conaffinity="0", rgba="0.3 0.3 0.3 1")
        ET.SubElement(tool, "geom", name="tip", type="sphere", size="0.012",
                      pos="0 0 0.10", mass="0.05", friction=f"{config.friction} 0.005 0.0001",
                      solref="0.01 1", solimp="0.95 0.99 0.001", rgba="0.9 0.5 0.1 1")
        ET.SubElement(tool, "site", name="tcp", pos="0 0 0.112", size="0.003")
        sensor = ET.SubElement(xml, "sensor")
        ET.SubElement(sensor, "force", name="wrist_force", site="wrist")
        ET.SubElement(sensor, "torque", name="wrist_torque", site="wrist")
        world = xml.find("worldbody")
        ET.SubElement(world, "geom", name="table", type="box", size="0.22 0.22 0.02",
                      pos="0.5 0 0", friction=f"{config.friction} 0.005 0.0001",
                      solref="0.01 1", solimp="0.95 0.99 0.001", rgba="0.35 0.5 0.6 1")
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self.tcp = self.model.site("tcp").id
        self.tip = self.model.geom("tip").id
        self.table = self.model.geom("table").id
        self.home = self.data.qpos.copy()
        self.initial = self.data.site_xpos[self.tcp].copy()
        self.rotation = self.data.site_xmat[self.tcp].reshape(3, 3).copy()
        self.surface_z = self.initial[2] - 0.015
        self.model.geom_pos[self.table] = [self.initial[0], self.initial[1], self.surface_z - 0.02]
        mujoco.mj_forward(self.model, self.data)
        self.reference = self.initial.copy()
        self.filtered_force = 0.0
        self.jp = np.zeros((3, self.model.nv))
        self.jr = np.zeros_like(self.jp)

    def contact_force(self):
        force = np.zeros(3)
        for c in range(self.data.ncon):
            contact = self.data.contact[c]
            if {contact.geom1, contact.geom2} != {self.tip, self.table}:
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, c, wrench)
            sign = 1 if contact.geom2 == self.tip else -1
            force += sign * contact.frame.reshape(3, 3).T @ wrench[:3]
        return force

    def step(self):
        m, d, cfg = self.model, self.data, self.config
        t = d.time
        force = self.contact_force()
        self.filtered_force += (1 - np.exp(-cfg.timestep / 0.02)) * (force[2] - self.filtered_force)
        speed = np.clip(0.002 * (self.filtered_force - cfg.target_force), -0.02, 0.02)
        self.reference[2] = np.clip(self.reference[2] + speed * cfg.timestep,
                                    self.surface_z - 0.04, self.initial[2])
        phase = np.clip((t - 2) / (cfg.duration - 3), 0, 1)
        smooth = phase * phase * (3 - 2 * phase)
        distance = cfg.slide_distance if cfg.task == "slide" else 0
        self.reference[0] = self.initial[0] + distance * smooth
        mujoco.mj_jacSite(m, d, self.jp, self.jr, self.tcp)
        rotation = d.site_xmat[self.tcp].reshape(3, 3)
        error_r = sum(np.cross(rotation[:, i], self.rotation[:, i]) for i in range(3)) * 0.5
        wrench = np.concatenate((1000 * (self.reference - d.site_xpos[self.tcp])
                                 - 65 * (self.jp @ d.qvel),
                                 60 * error_r - 10 * (self.jr @ d.qvel)))
        jac = np.vstack((self.jp, self.jr))
        null = np.eye(m.nv) - jac.T @ np.linalg.solve(jac @ jac.T + 1e-5 * np.eye(6), jac)
        torque = jac.T @ wrench + d.qfrc_bias + null @ (10 * (self.home - d.qpos) - 3 * d.qvel)
        d.ctrl[:] = np.clip(torque, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
        row = np.concatenate(([t], d.site_xpos[self.tcp], self.reference, force,
                              d.sensordata, d.qpos, d.qvel, d.ctrl)).copy()
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all() or any(w.number for w in d.warning):
            raise RuntimeError("simulation became unstable or emitted a MuJoCo warning")
        return row

    def metadata(self):
        return {"config": asdict(self.config), "initial_tcp": self.initial.tolist(),
                "surface_z": float(self.surface_z), "mujoco": mujoco.__version__,
                "numpy": np.__version__, "controller": "cartesian_impedance_normal_admittance_v1",
                "model_commit": "b0655057a9df2abe39ea7fc989914ba6f248301c",
                "force_frame": "world; contact force on tool; +Z up",
                "wrist_frame": "wrist site; raw MuJoCo internal wrench, includes tool dynamics/gravity",
                "seed_note": "deterministic baseline; no random perturbations enabled"}
