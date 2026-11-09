from project_paths import PROJECT_ROOT
import time

import mujoco
import mujoco.viewer
import numpy as np


def smoothstep(start, end, elapsed, duration):
    s = np.clip(elapsed / duration, 0.0, 1.0)
    alpha = 3.0 * s**2 - 2.0 * s**3
    return start + alpha * (end - start)


if __name__ == "__main__":
    model_path = (
        PROJECT_ROOT
        / "modelingfile"
        / "robot"
        / "scene.xml"
    )

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    gripper_joint_id = model.joint("gripper").id
    gripper_actuator_id = model.actuator("gripper").id
    gripper_qpos_id = model.jnt_qposadr[gripper_joint_id]

    # 각 액추에이터에 연결된 관절의 위치 인덱스
    joint_ids = model.actuator_trnid[:, 0]
    actuator_qpos_ids = model.jnt_qposadr[joint_ids]

    # 초기 자세를 유지하기 위한 관절 목표 각도
    hold_targets = data.qpos[actuator_qpos_ids].copy()

    target_a = 0.0
    target_b = 0.8

    print("그리퍼 관절 범위:", model.jnt_range[gripper_joint_id])
    print("0.0 ↔ 0.8rad를 천천히 반복합니다.")

    next_print = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:

        
        viewer.cam.lookat[:] = [0.34, 0.0, 0.23]
        viewer.cam.distance = 0.55
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20

        while viewer.is_running():
            wall_start = time.perf_counter()
            t = data.time % 8.0

            if t < 1.0:
                target = target_a

            elif t < 3.0:
                target = smoothstep(
                    target_a, target_b, t - 1.0, 2.0
                )

            elif t < 5.0:
                target = target_b

            elif t < 7.0:
                target = smoothstep(
                    target_b, target_a, t - 5.0, 2.0
                )

            else:
                target = target_a

            # 나머지 관절은 초기 자세 유지
            data.ctrl[:] = hold_targets
            data.ctrl[gripper_actuator_id] = target

            # 명령과 같은 시점의 실제 관절 각도 확인
            if data.time >= next_print:
                actual = float(data.qpos[gripper_qpos_id])
                print(
                    f"목표={target:.3f} rad | "
                    f"실제={actual:.3f} rad"
                )
                next_print = data.time + 0.5

            mujoco.mj_step(model, data)
            viewer.sync()

            elapsed = time.perf_counter() - wall_start
            time.sleep(max(0.0, model.opt.timestep - elapsed))