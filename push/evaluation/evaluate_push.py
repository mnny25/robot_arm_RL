from project_paths import PROJECT_ROOT
import csv

import numpy as np
import torch
from stable_baselines3 import PPO, SAC

from push.push_env import PushEnv


def evaluate(policy, name, seeds):
    env = PushEnv()
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
            })

            if episode % 10 == 0:
                print(f"{name}: {episode}/{len(seeds)} 완료")
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

    root = PROJECT_ROOT
    seeds = list(range(100))

    policies = [
        (
            "PPO",
            PPO.load(
                str(root / "results/ppo_push_003/ppo_push.zip"),
                device="cpu",
            ),
        ),
        (
            "SAC",
            SAC.load(
                str(root / "results/sac_push_002/ppo_push.zip"),
                device="cpu",
            ),
        ),
    ]

    all_rows = []

    for name, policy in policies:
        rows = evaluate(policy, name, seeds)
        print_summary(name, rows)
        all_rows.extend(rows)

    output = root / "results/push_comparison.csv"

    with output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(all_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print("\n평가 결과 저장:", output)