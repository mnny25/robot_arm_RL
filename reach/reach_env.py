from project_paths import PROJECT_ROOT

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
            PROJECT_ROOT
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
        self.max_steps = 200 # action 200회, 주기 20Hz, 에피소드 최대 길이 10초

        self.action_space = spaces.Box( # -1부터 1사이의 숫자 4개를 정책이 환경에 전달하는 행동
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
        
    
    def _get_obs(self): # observation: PPO가 로봇에 대해 아는 정보
        q = self.data.qpos[self.qpos_ids]
        v = self.data.qvel[self.qvel_ids]

        ee_pos = self.data.site("gripperframe").xpos
        relative_goal = self.goal - ee_pos

        return np.concatenate([
            q, # 관절 각도
            v, # 관절 속도
            self.joint_target, # 명령 각도
            relative_goal, # 목표 상대 위치
        ]).astype(np.float32)

    def reset(self, *, seed=None, options=None): # 한 에피소드의 시작 조건
        super().reset(seed=seed) # 목표는 reset 때 한 번 정해지고, 그 에피소드 동안 고정됨

        mujoco.mj_resetData(self.model, self.data) # 초기 상태로 되돌림

        self.steps = 0
        self.joint_target = self.data.qpos[self.qpos_ids].copy()
        self.data.ctrl[self.actuator_ids] = self.joint_target

        mujoco.mj_forward(self.model, self.data)

        # 초기 EE 위치 근처에 고정된 목표 설정
        initial_ee = self.data.site("gripperframe").xpos.copy()
        
        # 초기 EE를 기준으로 뒤쪽·아래쪽의 목표를 샘플링
        offset = self.np_random.uniform(
            low=np.array([-0.10, -0.04, -0.06]),
            high=np.array([-0.03, 0.04, -0.02]),
        )

        self.goal = initial_ee + offset

        return self._get_obs(), {}
    
    def step(self, action):
        
        # 물리 1스텝 = 0.005초

        # frame_skip = 0.05 / 0.005 = 10

        # action 1회
        # → 물리 계산 10회
        # → 시뮬레이션 시간 0.05초 진행
        # E따라서 정책의 제어 주기는 약 20Hz
        
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

        success = distance < 0.01 # 집게 끝의 기준점이 공중에 설정된 목표에서 1cm이내인가

        # 거리와 과도한 행동에 비용 부여
        reward = -10.0 * distance # 목표에 가까울수록 덜 부정적인 값을 받음
        reward -= 0.01 * float(np.sum(action**2)) # 불필요하게 큰 명령을 줄이려는 항

        if success:
            reward += 10.0 # 목표와 1cm 미만이면 보너스 +10을 받음

        terminated = bool(success) # 목표 도달로 종료
        truncated = bool( # 제한 시간까지 성공하지 못해 중단됨
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