from pathlib import Path
import csv

import mujoco

import numpy as np
import torch
from stable_baselines3 import PPO, SAC

from push_env import PushEnv


def evaluate(policy, name, seeds, friction):
    env = PushEnv()
    # 질량은 기존 50g인지 확인
    cube_body_id = env.model.body("cube").id

    if not np.isclose(
        env.model.body_mass[cube_body_id], 0.05
    ):
        raise ValueError("큐브 질량이 50g인지 확인하세요.")

    table_geom_id = env.model.geom("table_top").id
    cube_geom_id = env.model.geom("cube_geom").id

    # 첫 번째 값인 미끄럼 마찰계수만 변경
    env.model.geom_friction[table_geom_id, 0] = friction
    env.model.geom_friction[cube_geom_id, 0] = friction

    rows = []
    control_dt = env.frame_skip * env.model.opt.timestep

    try:
        for episode, seed in enumerate(seeds, start=1):
            # 두 정책에 같은 seed 목록을 적용
            obs, _ = env.reset(seed=seed)

            cube_start = env.data.body("cube").xpos.copy()
            max_speed = 0.0
            max_height = float(cube_start[2])
            total_reward = 0.0

            while True:
                action, _ = policy.predict(
                    obs, deterministic=True
                )

                obs, reward, terminated, truncated, info = (
                    env.step(action)
                )

                total_reward += reward
                max_speed = max(max_speed, info["cube_speed"])
                max_height = max(
                    max_height,
                    float(env.data.body("cube").xpos[2]),
                )

                if terminated or truncated:
                    break

            rows.append({
                "policy": name,
                "seed": seed,
                "cube_start_x": float(cube_start[0]),
                "cube_start_y": float(cube_start[1]),
                "success": bool(info["is_success"]),
                "fallen": bool(info["fallen"]),
                "steps": env.steps,
                "sim_time_s": env.steps * control_dt,
                "final_distance_mm": info["distance"] * 1000,
                "max_speed_m_s": max_speed,
                "max_height_mm": max_height * 1000,
                "reward": total_reward,
                "friction": friction,
            })

            if episode % 10 == 0:
                print(
                    f"{name} / 마찰 {friction}: "
                    f"{episode}/{len(seeds)} 완료"
                )
    finally:
        env.close()

    return rows


def print_summary(name, rows):
    successful = [row for row in rows if row["success"]]

    print(f"\n{name}")
    print(f"성공률: {len(successful)}/{len(rows)}")

    if successful:
        mean_time = np.mean([
            row["sim_time_s"] for row in successful
        ])
        print(f"성공한 시도의 평균 시간: {mean_time:.2f}초")

    mean_distance = np.mean([
        row["final_distance_mm"] for row in rows
    ])
    mean_peak_speed = np.mean([
        row["max_speed_m_s"] for row in rows
    ])
    mean_peak_height = np.mean([
        row["max_height_mm"] for row in rows
    ])

    print(f"전체 평균 최종 거리: {mean_distance:.1f}mm")
    print(f"시도별 최고 속도의 평균: {mean_peak_speed:.3f}m/s")
    print(f"시도별 최고 높이의 평균: {mean_peak_height:.1f}mm")


if __name__ == "__main__":
    torch.set_num_threads(1)

    root = Path(__file__).resolve().parent
    seeds = list(range(100))

    policies = [
        (
            "PPO_DR",
            PPO.load(
                str(root / "results/ppo_push_progress_001/ppo_push.zip"),
                device="cpu",
            ),
        ),
    ]

    all_rows = []

    frictions = [1.0, 0.25, 0.5, 1.5]

    for friction in frictions:
        for name, policy in policies:
            rows = evaluate(policy, name, seeds, friction)

            label = f"{name} / 마찰 {friction}"
            print_summary(label, rows)

            all_rows.extend(rows)

    output = root / "results/push_friction_progress_comparison.csv"

    with output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(all_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print("\n평가 결과 저장:", output)