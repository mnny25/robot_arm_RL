from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import time
import mujoco.viewer


class ReachEnv(gym.Env):
    def __init__(self):
        super().__init__()

        model_path = (
            Path(__file__).resolve().parent
            / "modelingfile"
            / "robot"
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
        
    
    def _get_obs(self):
        q = self.data.qpos[self.qpos_ids]
        v = self.data.qvel[self.qvel_ids]

        ee_pos = self.data.site("gripperframe").xpos
        relative_goal = self.goal - ee_pos

        return np.concatenate([
            q,
            v,
            self.joint_target,
            relative_goal,
        ]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        self.steps = 0
        self.joint_target = self.data.qpos[self.qpos_ids].copy()
        self.data.ctrl[self.actuator_ids] = self.joint_target

        mujoco.mj_forward(self.model, self.data)

        # 초기 EE 위치 근처에 고정된 목표 설정
        initial_ee = self.data.site("gripperframe").xpos.copy()
        self.goal = initial_ee + np.array([-0.04, 0.02, -0.02])

        return self._get_obs(), {}
    
    def step(self, action):
        action = np.clip(
            np.asarray(action, dtype=np.float64),
            -1.0,
            1.0,
        )

        # action으로 관절 목표 각도 변경
        self.joint_target = np.clip(
            self.joint_target + self.action_scale * action,
            self.lower,
            self.upper,
        )

        self.data.ctrl[self.actuator_ids] = self.joint_target

        # 하나의 action을 일정 시간 유지
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # 적분 후 상태에 맞게 EE 위치 갱신
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1

        ee_pos = self.data.site("gripperframe").xpos
        distance = float(np.linalg.norm(self.goal - ee_pos))

        success = distance < 0.01

        # 거리와 과도한 행동에 비용 부여
        reward = -10.0 * distance
        reward -= 0.01 * float(np.sum(action**2))

        if success:
            reward += 10.0

        terminated = bool(success)
        truncated = bool(
            self.steps >= self.max_steps and not terminated
        )

        info = {
            "distance": distance,
            "is_success": success,
        }

        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            info,
        )
        
if __name__ == "__main__":
    env = ReachEnv()
    obs, info = env.reset(seed=0)

    control_dt = env.frame_skip * env.model.opt.timestep

    with mujoco.viewer.launch_passive(
        env.model, env.data
    ) as viewer:

        viewer.cam.lookat[:] = [0.15, 0.0, 0.10]
        viewer.cam.distance = 0.9
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25

        while viewer.is_running():
            start = time.perf_counter()

            # 아직 학습 전이므로 랜덤 명령 사용
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)

            viewer.sync()

            if terminated or truncated:
                print(
                    f"종료: 성공={info['is_success']}, "
                    f"거리={info['distance']:.4f} m"
                )
                obs, info = env.reset()

            # 화면에서 관찰할 수 있도록 실제 시간에 맞춤
            elapsed = time.perf_counter() - start
            time.sleep(max(0.0, control_dt - elapsed))

    env.close()