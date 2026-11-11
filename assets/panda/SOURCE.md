# Model provenance

Source: https://github.com/google-deepmind/mujoco_menagerie
Commit: b0655057a9df2abe39ea7fc989914ba6f248301c
Directory: franka_emika_panda

Upstream panda_nohand.xml and assets are copied without modification.
The upstream model license is included in LICENSE. Runtime changes (torque
actuators, tool, sensors and surface) are applied in memory in simulation.py.
This model uses the arm without the parallel gripper.
