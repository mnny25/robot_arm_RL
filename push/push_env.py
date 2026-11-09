from project_paths import PROJECT_ROOT

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import time
import mujoco.viewer


class PushEnv(gym.Env):
    def __init__(self, randomize_friction=False):
        super().__init__()

        model_path = (
            PROJECT_ROOT
            / "modelingfile"
            / "robot"
            / "scene.xml"
        )

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        
        self.randomize_friction = randomize_friction

        self.table_geom_id = self.model.geom("table_top").id
        self.cube_geom_id = self.model.geom("cube_geom").id

        arm_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ]
        
        cube_joint_id = self.model.joint("cube_free").id

        self.cube_qpos_id = int(
            self.model.jnt_qposadr[cube_joint_id]
        )

        self.cube_dof_id = int(
            self.model.jnt_dofadr[cube_joint_id]
        )

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
            shape=(31,),
            dtype=np.float32,
        )

        self.goal = np.zeros(3)
        self.joint_target = np.zeros(5)
        self.steps = 0
        
    
    def _get_obs(self):
        # 로봇 상태
        q = self.data.qpos[self.qpos_ids]
        v = self.data.qvel[self.qvel_ids]

        ee_pos = self.data.site("gripperframe").xpos
        cube_pos = self.data.body("cube").xpos

        # 물체 사이의 상대 위치
        ee_to_cube = cube_pos - ee_pos
        cube_to_goal = self.goal - cube_pos

        # 자유 관절의 qpos: [x, y, z, qw, qx, qy, qz]
        cube_quat = self.data.qpos[
            self.cube_qpos_id + 3:
            self.cube_qpos_id + 7
        ]

        cube_velocity = self.data.qvel[
            self.cube_dof_id:
            self.cube_dof_id + 6
        ]

        return np.concatenate([
            q,                  # 5
            v,                  # 5
            self.joint_target,  # 5
            ee_to_cube,         # 3
            cube_to_goal,       # 3
            cube_quat,          # 4
            cube_velocity,      # 6
        ]).astype(np.float32)

    def reset(self, *, seed=None, options=None): # 한 에피소드의 시작 조건
        super().reset(seed=seed) # 목표는 reset 때 한 번 정해지고, 그 에피소드 동안 고정됨
        self.stable_time = 0.0
        
        mujoco.mj_resetData(self.model, self.data) # 초기 상태로 되돌림
        
        # XML에 정의된 큐브 초기 위치를 기준으로 XY만 변경
        cube_xy_offset = self.np_random.uniform(
            low=-0.01,
            high=0.01,
            size=2,
        )

        self.data.qpos[
            self.cube_qpos_id:self.cube_qpos_id + 2
        ] += cube_xy_offset
        
        # 에피소드가 시작할 때만 마찰 변경
        if self.randomize_friction:
            friction = float(
                self.np_random.uniform(0.5, 1.5)
            )

            self.model.geom_friction[
                self.table_geom_id, 0
            ] = friction

            self.model.geom_friction[
                self.cube_geom_id, 0
            ] = friction

        self.steps = 0
        self.joint_target = self.data.qpos[self.qpos_ids].copy()
        self.data.ctrl[self.actuator_ids] = self.joint_target

        mujoco.mj_forward(self.model, self.data)

        self.goal = self.data.site("target").xpos.copy()
        
        cube_pos = self.data.body("cube").xpos

        self.prev_goal_distance = float(
            np.linalg.norm(self.goal[:2] - cube_pos[:2])
        )
        

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

        ee_pos = self.data.site("gripperframe").xpos.copy()
        cube_pos = self.data.body("cube").xpos.copy()

        # EE와 큐브 사이의 3차원 거리
        ee_distance = float(
            np.linalg.norm(cube_pos - ee_pos)
        )

        # 큐브와 목표 사이의 평면 거리
        goal_distance = float(
            np.linalg.norm(self.goal[:2] - cube_pos[:2])
        )
        
        # 양수: 목표에 가까워짐
        # 음수: 목표에서 멀어짐
        progress = self.prev_goal_distance - goal_distance

        self.prev_goal_distance = goal_distance

        # 테이블 윗면 z=0, 큐브 절반 높이 0.02m 기준
        on_table_height = abs(cube_pos[2] - 0.02) < 0.01
        
        cube_velocity = self.data.qvel[
            self.cube_dof_id:self.cube_dof_id + 6
        ]

        linear_speed = float(np.linalg.norm(cube_velocity[:3]))
        angular_speed = float(np.linalg.norm(cube_velocity[3:]))

        # 첫 실험용 기준: 목표 안에서 거의 정지
        stable = (
            goal_distance < 0.03
            and on_table_height
            and linear_speed < 0.02   # 2cm/s 미만
            and angular_speed < 0.2   # 0.2rad/s 미만
        )

        control_dt = self.frame_skip * self.model.opt.timestep

        if stable:
            self.stable_time += control_dt
        else:
            self.stable_time = 0.0

        # 위 조건을 연속 0.5초 유지해야 성공
        success = bool(self.stable_time >= 0.5)

        success = bool(
            goal_distance < 0.03 and on_table_height
        )

        fallen = bool(cube_pos[2] < -0.02)

        # 첫 실험용 보상
        reward = -1.0 * ee_distance
        reward -= 5.0 * goal_distance
        reward -= 0.01 * float(np.sum(action**2))
        
        # 추가: 목표 방향으로 이동한 정도에 대한 보상
        reward += 20.0 * progress

        if success:
            reward += 10.0

        if fallen:
            reward -= 10.0

        terminated = bool(success or fallen)

        truncated = bool(
            self.steps >= self.max_steps and not terminated
        )

        info = {
            "distance": goal_distance,
            "ee_distance": ee_distance,
            "is_success": success,
            "fallen": fallen,
            "cube_speed": linear_speed,
            "stable_time": self.stable_time,
            "friction": float(
                self.model.geom_friction[self.cube_geom_id, 0]
            ),
            "progress": float(progress),
        }

        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            info,
        )
        
if __name__ == "__main__":
    env = PushEnv()
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
                print("Push 관측 크기:", obs.shape)
                
            # 화면에서 관찰할 수 있도록 실제 시간에 맞춤
            elapsed = time.perf_counter() - start
            time.sleep(max(0.0, control_dt - elapsed))

    env.close()