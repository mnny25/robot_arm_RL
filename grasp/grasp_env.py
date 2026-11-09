from project_paths import PROJECT_ROOT

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


class GraspEnv(gym.Env):
    def __init__(self):
        super().__init__()

        model_path = (
            PROJECT_ROOT
            / "modelingfile"
            / "robot"
            / "scene_grasp.xml"
        )

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        # 팔 관절 5개와 그리퍼 관절 1개
        self.joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]

        joint_ids = np.array([
            self.model.joint(name).id
            for name in self.joint_names
        ])

        self.qpos_ids = self.model.jnt_qposadr[joint_ids]
        self.qvel_ids = self.model.jnt_dofadr[joint_ids]

        self.actuator_ids = np.array([
            self.model.actuator(name).id
            for name in self.joint_names
        ])

        self.lower = np.maximum(
            self.model.jnt_range[joint_ids, 0],
            self.model.actuator_ctrlrange[self.actuator_ids, 0],
        )
        self.upper = np.minimum(
            self.model.jnt_range[joint_ids, 1],
            self.model.actuator_ctrlrange[self.actuator_ids, 1],
        )

        cube_joint_id = self.model.joint("cube_free").id
        self.cube_qpos_id = int(
            self.model.jnt_qposadr[cube_joint_id]
        )
        self.cube_dof_id = int(
            self.model.jnt_dofadr[cube_joint_id]
        )

        self.cube_body_id = self.model.body("cube").id
        self.fixed_body_id = self.model.body("gripper").id
        self.moving_body_id = self.model.body(
            "moving_jaw_so101_v1"
        ).id

        self.frame_skip = max(
            1, round(0.05 / self.model.opt.timestep)
        )
        self.control_dt = (
            self.frame_skip * self.model.opt.timestep
        )

        self.action_scale = 0.03
        self.max_steps = 200

        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(6,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(35,), dtype=np.float32
        )

        self.joint_target = np.zeros(6)
        self.goal = np.zeros(3)
        self.hold_time = 0.0
        self.rest_height = 0.0
        self.steps = 0

    def _get_obs(self):
        ee_pos = self.data.site("grasp_center").xpos
        cube_pos = self.data.body("cube").xpos

        return np.concatenate([
            self.data.qpos[self.qpos_ids],       # 6
            self.data.qvel[self.qvel_ids],       # 6
            self.joint_target,                  # 6
            cube_pos - ee_pos,                  # 3
            self.goal - cube_pos,               # 3
            self.data.qpos[
                self.cube_qpos_id + 3:
                self.cube_qpos_id + 7
            ],                                  # 4
            self.data.qvel[
                self.cube_dof_id:
                self.cube_dof_id + 6
            ],                                  # 6
            np.array([self.hold_time]),          # 1
        ]).astype(np.float32)

    def _both_jaws_touch(self):
        touching = set()

        for contact in self.data.contact:
            if contact.dist > 0:
                continue

            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])

            if body1 == self.cube_body_id:
                touching.add(body2)

            if body2 == self.cube_body_id:
                touching.add(body1)

        return (
            self.fixed_body_id in touching
            and self.moving_body_id in touching
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # 고정 동작에서 확인한, 집게를 열고 내려간 자세
        initial_angles = np.array([
            -0.00618,
            -0.22093,
            0.62567,
            1.16609,
            -0.00603,
            0.8,
        ])

        self.data.qpos[self.qpos_ids] = initial_angles
        self.joint_target = initial_angles.copy()
        self.data.ctrl[self.actuator_ids] = self.joint_target

        # 큐브가 테이블에 안착하도록 잠깐 시뮬레이션
        for _ in range(round(0.2 / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)

        mujoco.mj_forward(self.model, self.data)

        self.steps = 0
        self.hold_time = 0.0

        cube_pos = self.data.body("cube").xpos.copy()
        self.rest_height = float(cube_pos[2])

        # 초기 큐브 위치보다 5cm 높은 지점
        self.goal = cube_pos.copy()
        self.goal[2] += 0.05

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(
            np.asarray(action, dtype=np.float64), -1.0, 1.0
        )

        self.joint_target = np.clip(
            self.joint_target + self.action_scale * action,
            self.lower,
            self.upper,
        )
        self.data.ctrl[self.actuator_ids] = self.joint_target

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        mujoco.mj_forward(self.model, self.data)
        self.steps += 1

        cube_pos = self.data.body("cube").xpos.copy()
        ee_pos = self.data.site("grasp_center").xpos.copy()

        cube_velocity = self.data.qvel[
            self.cube_dof_id:self.cube_dof_id + 6
        ]

        lift_height = float(cube_pos[2] - self.rest_height)
        both_contact = self._both_jaws_touch()

        # 높이 목표는 5cm, 허용 오차는 1cm
        stable = (
            lift_height >= 0.04
            and both_contact
            and np.linalg.norm(cube_velocity[:3]) < 0.05
            and np.linalg.norm(cube_velocity[3:]) < 0.5
        )

        if stable:
            self.hold_time = min(
                1.0, self.hold_time + self.control_dt
            )
        else:
            self.hold_time = 0.0

        success = bool(self.hold_time >= 1.0)
        fallen = bool(cube_pos[2] < -0.02)

        ee_distance = float(np.linalg.norm(cube_pos - ee_pos))
        height_error = max(0.0, self.goal[2] - cube_pos[2])

        # 접근, 높이, 접촉 상태를 평가하는 첫 실험용 보상
        reward = -2.0 * ee_distance
        reward -= 8.0 * height_error
        reward -= 0.2 * float(not both_contact)
        reward -= 0.01 * float(np.sum(action**2))
        reward -= 0.02

        if success:
            reward += 10.0
        if fallen:
            reward -= 10.0

        terminated = bool(success or fallen)
        truncated = bool(
            self.steps >= self.max_steps and not terminated
        )

        info = {
            "is_success": success,
            "lift_height": lift_height,
            "both_contact": bool(both_contact),
            "hold_time": self.hold_time,
            "fallen": fallen,
        }

        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            info,
        )


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    env = GraspEnv()
    check_env(env)

    obs, _ = env.reset(seed=0)
    print("관측 크기:", obs.shape)
    print("행동 크기:", env.action_space.shape)

    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )
        print(info)

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()