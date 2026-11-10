"""Cube Push 환경 v2

기존 push_env.py는 그대로 두고, 아래 내용을 설정(config)으로 바꿀 수 있게 새로 만든 환경입니다.

1. 성공 판정: 목표 안에서 큐브가 거의 정지한 상태를 hold_time 동안 유지해야 성공
2. 태스크 랜덤화: 큐브 시작 위치와 목표 위치를 매 에피소드 랜덤 샘플링
   - range_scale=0 이면 기존 고정 위치(큐브 0.22,0 / 목표 0.30,0.06)와 같음
   - range_scale=1 이면 설정한 전체 범위 사용 (커리큘럼으로 0→1 증가 가능)
3. Domain Randomization: 마찰, 질량, 액추에이터 kp, 관절 damping,
   큐브 위치 관측 노이즈/바이어스, 제어 지연
4. 관측 모드: "full"(기존 31차원) / "real"(실물에서 얻기 쉬운 값만, 21차원)

reset(options=...)으로 큐브·목표 위치와 물리 조건을 직접 지정할 수 있어
평가 스크립트에서 같은 조건을 반복 재현할 수 있습니다.
"""

from __future__ import annotations

import copy
import math
from collections import deque

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from project_paths import PROJECT_ROOT


ARM_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

CUBE_HALF = 0.02  # 큐브 반 변 길이 [m], 테이블 윗면 z=0

DEFAULT_CONFIG = {
    # ---- 시간 / 행동 ----
    "max_steps": 200,          # 20Hz x 200 = 10초
    "control_dt": 0.05,        # 정책 주기 [s]
    "action_scale": 0.03,      # 한 스텝 관절 목표 증분 최대 [rad]

    # ---- 성공 판정 ----
    "success_radius": 0.03,    # 목표 원 반지름 [m]
    "require_stable": True,    # False면 기존처럼 원 안에 들어오는 순간 성공
    "hold_time": 0.5,          # 정지 유지 시간 [s]
    "stable_lin_speed": 0.03,  # 정지로 볼 큐브 선속도 상한 [m/s]
    "stable_ang_speed": 0.5,   # 정지로 볼 큐브 각속도 상한 [rad/s]

    # ---- 보상 가중치 ----
    # 주의: 성공 전까지의 스텝 보상은 항상 0 이하가 되도록 설계했습니다.
    # 원 안에 있을 때 +보상을 주면, 성공(종료)하지 않고 원 근처에 머물며
    # 보상을 계속 모으는 '보상 해킹'이 생길 수 있기 때문입니다.
    "w_ee": 1.0,               # EE-큐브(또는 밀 위치) 거리
    "ee_target": "cube",       # "cube": 큐브 중심 / "push_point": 큐브의 목표 반대편 지점
    "push_margin": 0.015,      # push_point가 큐브 면에서 떨어진 거리 [m]
    "w_goal": 5.0,             # 큐브-목표 거리
    "w_progress": 20.0,        # 목표 쪽으로 가까워진 정도 (+)
    "w_action": 0.01,          # 행동 크기
    "w_cube_speed": 1.0,       # 큐브가 cube_speed_free 보다 빠를 때 페널티 (쳐서 날리기 방지)
    "cube_speed_free": 0.15,   # 이 속도까지는 페널티 없음 [m/s]
    "w_zone_speed": 2.0,       # 목표 근처(2*반지름)에서는 큐브 속도 자체를 페널티 → 멈추도록 유도
    "time_penalty": 0.0,       # 매 스텝 상수 페널티
    "success_bonus": 10.0,
    "fall_penalty": 10.0,

    # ---- 태스크 랜덤화 ----
    "nominal_cube_xy": (0.22, 0.0),    # 기존 scene.xml 큐브 위치
    "nominal_goal_xy": (0.30, 0.06),   # 기존 scene.xml 목표 위치
    "cube_xy_half_range": (0.04, 0.06),  # range_scale=1 일 때 큐브 x, y 반범위 [m]
    "goal_dist_range": (0.06, 0.12),     # 큐브→목표 거리 범위 [m]
    "goal_angle_center_deg": 0.0,        # 큐브→목표 방향 중심 (0도 = +x, 로봇 바깥쪽)
    "goal_angle_half_range_deg": 60.0,   # 방향 반범위 [deg]
    "cube_yaw_half_range_deg": 0.0,      # 큐브 초기 yaw 반범위 [deg]
    "jitter": 0.01,                      # range_scale과 무관하게 항상 주는 큐브 위치 흔들림 [m]
    "range_scale": 1.0,                  # 0: 고정 위치, 1: 전체 범위
    "workspace_x": (0.14, 0.36),         # 큐브·목표가 놓일 수 있는 영역 [m]
    "workspace_y": (-0.14, 0.14),

    # ---- Domain Randomization ----
    # None 이면 DR 없음. 예시:
    # {"friction": (0.5, 1.5), "mass": (0.03, 0.08), "kp_scale": (0.8, 1.2),
    #  "damping_scale": (0.8, 1.2), "obs_noise_std": 0.003,
    #  "cube_pos_bias_std": 0.003, "action_delay_max": 2}
    "dr": None,

    # ---- 관측 ----
    "obs_mode": "full",        # "full" | "real"
}

# 물리 조건 기본값 (DR이 없을 때, 또는 평가에서 지정하지 않은 항목)
NOMINAL_PHYSICS = {
    "friction": None,          # None이면 XML 값 유지
    "mass": None,              # None이면 XML 값 유지
    "kp_scale": 1.0,
    "damping_scale": 1.0,
    "obs_noise_std": 0.0,
    "cube_pos_bias": (0.0, 0.0, 0.0),
    "action_delay": 0,
}


def make_config(overrides: dict | None = None) -> dict:
    """기본 설정에 overrides를 덮어쓴 설정을 만든다. 오타 키는 에러."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    for key, value in (overrides or {}).items():
        if key not in cfg:
            raise KeyError(f"알 수 없는 설정 키: {key}")
        cfg[key] = copy.deepcopy(value)
    return cfg


class PushEnvV2(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: dict | None = None):
        super().__init__()
        self.cfg = make_config(config)

        model_path = PROJECT_ROOT / "modelingfile" / "robot" / "scene.xml"
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        # ---- id / 인덱스 ----
        joint_ids = np.array([self.model.joint(n).id for n in ARM_JOINTS])
        self.qpos_ids = self.model.jnt_qposadr[joint_ids]
        self.qvel_ids = self.model.jnt_dofadr[joint_ids]
        self.actuator_ids = np.array([self.model.actuator(n).id for n in ARM_JOINTS])

        cube_joint_id = self.model.joint("cube_free").id
        self.cube_qpos_id = int(self.model.jnt_qposadr[cube_joint_id])
        self.cube_dof_id = int(self.model.jnt_dofadr[cube_joint_id])
        self.cube_body_id = self.model.body("cube").id
        self.cube_geom_id = self.model.geom("cube_geom").id
        self.table_geom_id = self.model.geom("table_top").id
        self.target_site_id = self.model.site("target").id

        self.lower = np.maximum(
            self.model.jnt_range[joint_ids, 0],
            self.model.actuator_ctrlrange[self.actuator_ids, 0],
        )
        self.upper = np.minimum(
            self.model.jnt_range[joint_ids, 1],
            self.model.actuator_ctrlrange[self.actuator_ids, 1],
        )

        self.frame_skip = max(1, round(self.cfg["control_dt"] / self.model.opt.timestep))
        self.control_dt = self.frame_skip * self.model.opt.timestep
        self.hold_steps = max(1, math.ceil(self.cfg["hold_time"] / self.control_dt - 1e-9))

        # ---- DR 복원을 위한 원래 물리값 저장 ----
        self._nom_friction = self.model.geom_friction[
            [self.table_geom_id, self.cube_geom_id], 0
        ].copy()
        self._nom_mass = float(self.model.body_mass[self.cube_body_id])
        self._nom_inertia = self.model.body_inertia[self.cube_body_id].copy()
        self._nom_gain = self.model.actuator_gainprm[self.actuator_ids, 0].copy()
        self._nom_bias1 = self.model.actuator_biasprm[self.actuator_ids, 1].copy()
        self._nom_damping = self.model.dof_damping[self.qvel_ids].copy()

        # ---- 공간 정의 ----
        self.action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        obs_dim = {"full": 31, "real": 21}[self.cfg["obs_mode"]]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        # ---- 상태 ----
        self.range_scale = float(self.cfg["range_scale"])
        self.goal = np.zeros(3)
        self.joint_target = np.zeros(5)
        self.steps = 0
        self.stable_count = 0
        self.prev_goal_distance = 0.0
        self.max_cube_speed = 0.0
        self.physics = dict(NOMINAL_PHYSICS)
        self._ctrl_queue: deque = deque(maxlen=1)

    # ------------------------------------------------------------------
    # 커리큘럼용
    # ------------------------------------------------------------------
    def set_range_scale(self, scale: float) -> None:
        self.range_scale = float(np.clip(scale, 0.0, 1.0))

    def get_range_scale(self) -> float:
        return self.range_scale

    # ------------------------------------------------------------------
    # 태스크 샘플링
    # ------------------------------------------------------------------
    def _in_workspace(self, xy) -> bool:
        x0, x1 = self.cfg["workspace_x"]
        y0, y1 = self.cfg["workspace_y"]
        return bool(x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1)

    def _sample_task(self):
        """range_scale에 따라 큐브·목표 위치를 샘플링."""
        rng = self.np_random
        s = self.range_scale
        cube_nom = np.array(self.cfg["nominal_cube_xy"], dtype=float)
        goal_nom = np.array(self.cfg["nominal_goal_xy"], dtype=float)
        d0 = goal_nom - cube_nom
        dist0 = float(np.linalg.norm(d0))
        ang0 = float(math.degrees(math.atan2(d0[1], d0[0])))

        half = np.maximum(np.array(self.cfg["cube_xy_half_range"]) * s, self.cfg["jitter"])
        a_c, a_h = self.cfg["goal_angle_center_deg"], self.cfg["goal_angle_half_range_deg"]
        a_lo = ang0 + s * ((a_c - a_h) - ang0)
        a_hi = ang0 + s * ((a_c + a_h) - ang0)
        d_lo = dist0 + s * (self.cfg["goal_dist_range"][0] - dist0)
        d_hi = dist0 + s * (self.cfg["goal_dist_range"][1] - dist0)

        for _ in range(100):
            cube_xy = cube_nom + rng.uniform(-half, half)
            ang = math.radians(rng.uniform(a_lo, a_hi))
            dist = rng.uniform(d_lo, d_hi)
            goal_xy = cube_xy + dist * np.array([math.cos(ang), math.sin(ang)])
            if self._in_workspace(cube_xy) and self._in_workspace(goal_xy):
                return cube_xy, goal_xy
        return cube_nom.copy(), goal_nom.copy()

    def _sample_physics(self) -> dict:
        p = dict(NOMINAL_PHYSICS)
        dr = self.cfg["dr"]
        if not dr:
            return p
        rng = self.np_random
        if "friction" in dr:
            p["friction"] = float(rng.uniform(*dr["friction"]))
        if "mass" in dr:
            p["mass"] = float(rng.uniform(*dr["mass"]))
        if "kp_scale" in dr:
            p["kp_scale"] = float(rng.uniform(*dr["kp_scale"]))
        if "damping_scale" in dr:
            p["damping_scale"] = float(rng.uniform(*dr["damping_scale"]))
        p["obs_noise_std"] = float(dr.get("obs_noise_std", 0.0))
        bias_std = float(dr.get("cube_pos_bias_std", 0.0))
        if bias_std > 0:
            b = rng.normal(0.0, bias_std, size=3)
            b[2] = 0.0
            p["cube_pos_bias"] = tuple(float(v) for v in b)
        if dr.get("action_delay_max", 0) > 0:
            p["action_delay"] = int(rng.integers(0, int(dr["action_delay_max"]) + 1))
        return p

    def _apply_physics(self, p: dict) -> None:
        """항상 원래 값에서 시작해서 적용 → 이전 에피소드 값이 남지 않음."""
        m = self.model
        ids = [self.table_geom_id, self.cube_geom_id]
        if p["friction"] is None:
            m.geom_friction[ids, 0] = self._nom_friction
        else:
            # MuJoCo는 두 geom의 마찰 중 큰 값을 쓰므로 둘 다 같은 값으로 설정
            m.geom_friction[ids, 0] = p["friction"]

        if p["mass"] is None:
            m.body_mass[self.cube_body_id] = self._nom_mass
            m.body_inertia[self.cube_body_id] = self._nom_inertia
        else:
            m.body_mass[self.cube_body_id] = p["mass"]
            m.body_inertia[self.cube_body_id] = self._nom_inertia * (p["mass"] / self._nom_mass)

        # position 액추에이터: force = kp*(ctrl - q) - kv*qdot
        # gainprm[0] = kp, biasprm[1] = -kp
        m.actuator_gainprm[self.actuator_ids, 0] = self._nom_gain * p["kp_scale"]
        m.actuator_biasprm[self.actuator_ids, 1] = self._nom_bias1 * p["kp_scale"]
        m.dof_damping[self.qvel_ids] = self._nom_damping * p["damping_scale"]

    # ------------------------------------------------------------------
    def _cube_touches_non_table(self) -> bool:
        c = self.data.contact
        n = self.data.ncon
        for g1, g2 in zip(c.geom1[:n], c.geom2[:n]):
            if self.cube_geom_id in (g1, g2):
                other = g2 if g1 == self.cube_geom_id else g1
                if other != self.table_geom_id:
                    return True
        return False

    def _place(self, cube_xy, goal_xy, yaw_deg: float) -> None:
        q = self.cube_qpos_id
        self.data.qpos[q:q + 2] = cube_xy
        self.data.qpos[q + 2] = CUBE_HALF + 0.0005
        yaw = math.radians(yaw_deg)
        self.data.qpos[q + 3:q + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        self.data.qvel[self.cube_dof_id:self.cube_dof_id + 6] = 0.0
        self.goal = np.array([goal_xy[0], goal_xy[1], 0.0])
        self.model.site_pos[self.target_site_id] = [goal_xy[0], goal_xy[1], 0.001]

    def reset(self, *, seed=None, options=None):
        """options 예시
        {"cube_xy": (0.22, 0.0), "goal_xy": (0.30, 0.06), "cube_yaw_deg": 0,
         "range_scale": 0.5,
         "physics": {"friction": 1.5, "mass": 0.1, "kp_scale": 0.8, ...}}
        physics를 주면 DR 샘플링 대신 그 값을 쓰고, 빠진 항목은 기본값.
        """
        super().reset(seed=seed)
        options = options or {}
        if "range_scale" in options:
            self.set_range_scale(options["range_scale"])

        if "physics" in options:
            self.physics = dict(NOMINAL_PHYSICS)
            self.physics.update(options["physics"])
        else:
            self.physics = self._sample_physics()
        self._apply_physics(self.physics)

        fixed_task = "cube_xy" in options and "goal_xy" in options
        yaw_h = self.cfg["cube_yaw_half_range_deg"] * self.range_scale

        for _ in range(20):
            mujoco.mj_resetData(self.model, self.data)
            if fixed_task:
                cube_xy = np.asarray(options["cube_xy"], dtype=float)
                goal_xy = np.asarray(options["goal_xy"], dtype=float)
            else:
                cube_xy, goal_xy = self._sample_task()
            yaw = options.get("cube_yaw_deg", self.np_random.uniform(-yaw_h, yaw_h) if yaw_h > 0 else 0.0)
            self._place(cube_xy, goal_xy, yaw)
            mujoco.mj_forward(self.model, self.data)
            if fixed_task or not self._cube_touches_non_table():
                break

        self.steps = 0
        self.stable_count = 0
        self.max_cube_speed = 0.0
        self.joint_target = self.data.qpos[self.qpos_ids].copy()
        self.data.ctrl[self.actuator_ids] = self.joint_target
        delay = int(self.physics["action_delay"])
        self._ctrl_queue = deque([self.joint_target.copy()] * (delay + 1), maxlen=delay + 1)

        cube_pos = self.data.body("cube").xpos
        self.prev_goal_distance = float(np.linalg.norm(self.goal[:2] - cube_pos[:2]))

        info = {"cube_xy": np.array(cube_xy), "goal_xy": np.array(goal_xy), "physics": dict(self.physics)}
        return self._get_obs(), info

    # ------------------------------------------------------------------
    def _get_obs(self):
        q = self.data.qpos[self.qpos_ids]
        v = self.data.qvel[self.qvel_ids]
        ee_pos = self.data.site("gripperframe").xpos
        cube_pos = self.data.body("cube").xpos.copy()

        # 관측용 큐브 위치: 카메라 추정이라고 가정하고 바이어스·노이즈 추가
        cube_obs = cube_pos + np.asarray(self.physics["cube_pos_bias"])
        std = self.physics["obs_noise_std"]
        if std > 0:
            cube_obs = cube_obs + self.np_random.normal(0.0, std, size=3)

        ee_to_cube = cube_obs - ee_pos
        cube_to_goal = self.goal - cube_obs

        if self.cfg["obs_mode"] == "real":
            parts = [q, v, self.joint_target, ee_to_cube, cube_to_goal]
        else:
            quat = self.data.qpos[self.cube_qpos_id + 3:self.cube_qpos_id + 7]
            cube_vel = self.data.qvel[self.cube_dof_id:self.cube_dof_id + 6]
            parts = [q, v, self.joint_target, ee_to_cube, cube_to_goal, quat, cube_vel]
        return np.concatenate(parts).astype(np.float32)

    def step(self, action):
        cfg = self.cfg
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        self.joint_target = np.clip(
            self.joint_target + cfg["action_scale"] * action, self.lower, self.upper
        )
        # 제어 지연: delay 스텝 전의 명령이 실제로 적용됨
        self._ctrl_queue.append(self.joint_target.copy())
        self.data.ctrl[self.actuator_ids] = self._ctrl_queue[0]

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1

        ee_pos = self.data.site("gripperframe").xpos.copy()
        cube_pos = self.data.body("cube").xpos.copy()
        cube_vel = self.data.qvel[self.cube_dof_id:self.cube_dof_id + 6]
        lin_speed = float(np.linalg.norm(cube_vel[:3]))
        ang_speed = float(np.linalg.norm(cube_vel[3:]))
        self.max_cube_speed = max(self.max_cube_speed, lin_speed)

        ee_distance = float(np.linalg.norm(cube_pos - ee_pos))
        goal_distance = float(np.linalg.norm(self.goal[:2] - cube_pos[:2]))
        progress = self.prev_goal_distance - goal_distance

        # 보상에 쓰는 EE 목표점
        ee_reward_distance = ee_distance
        if cfg["ee_target"] == "push_point" and goal_distance > cfg["success_radius"]:
            # 큐브 → 목표 방향의 반대편, 큐브 면에서 push_margin 떨어진 지점
            direction = (self.goal[:2] - cube_pos[:2]) / goal_distance
            push_point = cube_pos.copy()
            push_point[:2] -= direction * (CUBE_HALF + cfg["push_margin"])
            ee_reward_distance = float(np.linalg.norm(push_point - ee_pos))
        self.prev_goal_distance = goal_distance

        on_table = abs(cube_pos[2] - CUBE_HALF) < 0.01
        in_zone = goal_distance < cfg["success_radius"] and on_table
        stable = (
            in_zone
            and lin_speed < cfg["stable_lin_speed"]
            and ang_speed < cfg["stable_ang_speed"]
        )
        self.stable_count = self.stable_count + 1 if stable else 0

        if cfg["require_stable"]:
            success = self.stable_count >= self.hold_steps
        else:
            success = bool(in_zone)
        fallen = bool(cube_pos[2] < -0.02)

        # ---- 보상 ----
        reward = -cfg["w_ee"] * ee_reward_distance
        reward -= cfg["w_goal"] * goal_distance
        reward += cfg["w_progress"] * progress
        reward -= cfg["w_action"] * float(np.sum(action ** 2))
        reward -= cfg["w_cube_speed"] * max(0.0, lin_speed - cfg["cube_speed_free"])
        if goal_distance < 2.0 * cfg["success_radius"]:
            reward -= cfg["w_zone_speed"] * lin_speed
        reward -= cfg["time_penalty"]
        if success:
            reward += cfg["success_bonus"]
        if fallen:
            reward -= cfg["fall_penalty"]

        terminated = bool(success or fallen)
        truncated = bool(self.steps >= cfg["max_steps"] and not terminated)

        info = {
            "is_success": bool(success),
            "fallen": fallen,
            "distance": goal_distance,
            "ee_distance": ee_distance,
            "ee_reward_distance": ee_reward_distance,
            "in_zone": bool(in_zone),
            "stable_time": self.stable_count * self.control_dt,
            "cube_speed": lin_speed,
            "max_cube_speed": self.max_cube_speed,
            "progress": float(progress),
            "range_scale": self.range_scale,
            "friction": float(self.model.geom_friction[self.cube_geom_id, 0]),
            "mass": float(self.model.body_mass[self.cube_body_id]),
            "kp_scale": float(self.physics["kp_scale"]),
            "action_delay": int(self.physics["action_delay"]),
        }
        return self._get_obs(), float(reward), terminated, truncated, info
