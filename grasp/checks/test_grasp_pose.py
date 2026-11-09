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
    
    wrist_actuator_id = model.actuator("wrist_flex").id

    # 각 액추에이터에 연결된 관절의 위치 인덱스
    joint_ids = model.actuator_trnid[:, 0]
    actuator_qpos_ids = model.jnt_qposadr[joint_ids]

    # 초기 자세를 유지하기 위한 관절 목표 각도
    hold_targets = data.qpos[actuator_qpos_ids].copy()

    # 집게만 연 자세
    opened_targets = hold_targets.copy()
    opened_targets[gripper_actuator_id] = 0.8

    # 손목을 아래로 돌린 자세
    ready_targets = opened_targets.copy()
    ready_targets[wrist_actuator_id] = 1.57

    # 큐브 위에 정렬하는 후보 자세
    aligned_targets = ready_targets.copy()

    alignment_angles = {
        "shoulder_pan": -0.00561,
        "shoulder_lift": -0.21044,
        "elbow_flex": 0.23648,
        "wrist_flex": 1.54553,
        "wrist_roll": -0.00286,
    }

    for name, angle in alignment_angles.items():
        actuator_id = model.actuator(name).id
        aligned_targets[actuator_id] = angle
        
    next_print = 0.0
    
    # 집게를 연 채 큐브 양옆으로 내려가는 후보 자세
    lowered_targets = aligned_targets.copy()

    lowering_angles = {
        "shoulder_pan": -0.00618,
        "shoulder_lift": -0.22093,
        "elbow_flex": 0.62567,
        "wrist_flex": 1.16609,
        "wrist_roll": -0.00603,
    }

    for name, angle in lowering_angles.items():
        actuator_id = model.actuator(name).id
        lowered_targets[actuator_id] = angle
    

    with mujoco.viewer.launch_passive(model, data) as viewer:

        
        viewer.cam.lookat[:] = [0.22, 0.0, 0.12]
        viewer.cam.distance = 0.65
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20

        while viewer.is_running():
            wall_start = time.perf_counter()
            t = data.time

            if t < 1.0:
                # 0~1초: 집게 열기
                data.ctrl[:] = smoothstep(
                    hold_targets, opened_targets, t, 1.0
                )

            elif t < 4.0:
                # 1~4초: 손목을 아래로 회전
                data.ctrl[:] = smoothstep(
                    opened_targets, ready_targets, t - 1.0, 3.0
                )

            elif t < 7.0:
                # 4~7초: 큐브 위로 정렬
                data.ctrl[:] = smoothstep(
                    ready_targets, aligned_targets, t - 4.0, 3.0
                )

            else:
                # 7~10초: 열린 집게를 내려간 뒤 유지
                data.ctrl[:] = smoothstep(
                    aligned_targets, lowered_targets, t - 7.0, 3.0
                )

            # 기존 그리퍼 각도 출력에서 사용하는 변수
            target = float(data.ctrl[gripper_actuator_id])
                        
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