"""테이블 배치를 거르기 위한 집게 위치·충돌 검사."""

import math

import mujoco
import numpy as np


class PushReach:
    """MuJoCo에서 집게 위치를 따라갈 관절 자세를 찾는다."""

    def __init__(self, model, qpos_ids, qvel_ids, lower, upper, cube_qpos_id):
        self.model = model
        self.data = mujoco.MjData(model)
        self.qpos_ids, self.qvel_ids = qpos_ids, qvel_ids
        self.lower, self.upper = lower, upper
        self.home = self.data.qpos[qpos_ids].copy()
        self.site_id = model.site("gripperframe").id
        self.jacobian = np.zeros((3, model.nv))
        self.damping = np.eye(3) * 0.002**2
        # 집게가 큐브를 밀어야 하므로 이 검사에서는 로봇과 테이블의 형상만 본다.
        self.data.qpos[cube_qpos_id + 2] = -0.5
        mujoco.mj_kinematics(model, self.data)
        self.shoulder_xy = self.data.body("shoulder").xpos[:2].copy()

    def _solve_position(self, target):
        for _ in range(30):
            mujoco.mj_kinematics(self.model, self.data)
            error = target - self.data.site_xpos[self.site_id]
            if np.linalg.norm(error) < 0.003:
                return self.data.qpos[self.qpos_ids].copy()

            mujoco.mj_comPos(self.model, self.data)
            mujoco.mj_jacSite(self.model, self.data, self.jacobian, None, self.site_id)
            j = self.jacobian[:, self.qvel_ids]
            delta = j.T @ np.linalg.solve(j @ j.T + self.damping, error)
            delta /= max(1.0, np.max(np.abs(delta)) / 0.2)
            current = self.data.qpos[self.qpos_ids]
            proposed = np.clip(current + delta, self.lower, self.upper)
            if np.max(np.abs(proposed - current)) < 1e-5:
                break
            self.data.qpos[self.qpos_ids] = proposed
        return None

    def _clear_between(self, start, goal):
        intervals = max(1, math.ceil(np.max(np.abs(goal - start)) / 0.12))
        for qpos in np.linspace(start, goal, intervals + 1)[1:]:
            self.data.qpos[self.qpos_ids] = qpos
            mujoco.mj_fwdPosition(self.model, self.data)
            if self.data.ncon:
                return False
        return True

    def can_push(self, behind_cube, behind_goal, height):
        """집게가 위에서 접근하고 밀기 선을 따라갈 수 있는지 근사한다."""
        start_xy = behind_cube - self.shoulder_xy
        self.data.qpos[self.qpos_ids] = self.home
        self.data.qpos[self.qpos_ids[0]] = np.clip(
            np.arctan2(start_xy[1], start_xy[0]), self.lower[0], self.upper[0]
        )

        intervals = max(1, math.ceil(np.linalg.norm(behind_goal - behind_cube) / 0.035))
        waypoints = [np.r_[behind_cube, height + 0.05]]
        waypoints.extend(np.r_[xy, height] for xy in np.linspace(behind_cube, behind_goal, intervals + 1))
        previous = self.home
        for point in waypoints:
            pose = self._solve_position(point)
            if pose is None or not self._clear_between(previous, pose):
                return False
            previous = pose
        return True
