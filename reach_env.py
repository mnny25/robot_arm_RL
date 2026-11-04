from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


class ReachEnv(gym.Env):
    def __init__(self):
        super().__init__()

        model_path = (
            Path(__file__).resolve().parent
            / "mujoco_menagerie"
            / "robotstudio_so101"
            / "scene.xml"
        )

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        arm_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ]

        joint_ids = np.array([
            self.model.joint(name).id for name in arm_names
        ])

        self.qpos_ids = self.model.jnt_qposadr[joint_ids]
        self.qvel_ids = self.model.jnt_dofadr[joint_ids]

        self.actuator_ids = np.array([
            self.model.actuator(name).id for name in arm_names
        ])

        # 관절 범위와 액추에이터 입력 범위의 공통 구간
        self.lower = np.maximum(
            self.model.jnt_range[joint_ids, 0],
            self.model.actuator_ctrlrange[self.actuator_ids, 0],
        )
        self.upper = np.minimum(
            self.model.jnt_range[joint_ids, 1],
            self.model.actuator_ctrlrange[self.actuator_ids, 1],
        )

        # 약 0.05초마다 새로운 action 적용
        self.frame_skip = max(
            1, round(0.05 / self.model.opt.timestep)
        )

        self.action_scale = 0.03  # 목표 각도 증분의 최대 크기 [rad]
        self.max_steps = 200

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(18,),
            dtype=np.float32,
        )

        self.goal = np.zeros(3)
        self.joint_target = np.zeros(5)
        self.steps = 0