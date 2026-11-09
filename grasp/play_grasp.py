from project_paths import PROJECT_ROOT
import time

import mujoco.viewer
from stable_baselines3 import PPO

from grasp.grasp_env import GraspEnv


if __name__ == "__main__":
    root = PROJECT_ROOT

    policy = PPO.load(
        str(root / "results/ppo_grasp_001/ppo_grasp.zip"),
        device="cpu",
    )

    env = GraspEnv()
    obs, _ = env.reset(seed=0)

    episode = 1
    successes = 0

    max_lift = 0.0
    max_hold = 0.0
    ever_both_contact = False

    with mujoco.viewer.launch_passive(
        env.model, env.data
    ) as viewer:
        viewer.cam.lookat[:] = [0.22, 0.0, 0.10]
        viewer.cam.distance = 0.55
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -18

        while viewer.is_running():
            start = time.perf_counter()

            action, _ = policy.predict(
                obs, deterministic=True
            )

            obs, reward, terminated, truncated, info = env.step(action)

            max_lift = max(max_lift, info["lift_height"])
            max_hold = max(max_hold, info["hold_time"])
            ever_both_contact |= info["both_contact"]

            viewer.sync()

            if terminated or truncated:
                successes += int(info["is_success"])

                print(
                    f"시도 {episode} | "
                    f"성공={info['is_success']} | "
                    f"최대 상승={max_lift * 100:.2f}cm | "
                    f"최대 유지={max_hold:.2f}초 | "
                    f"양쪽 접촉 경험={ever_both_contact} | "
                    f"낙하={info['fallen']} | "
                    f"스텝={env.steps}"
                )

                time.sleep(0.8)

                if episode >= 3:
                    print(f"완료: {successes}/3회 성공")
                    break

                episode += 1
                max_lift = 0.0
                max_hold = 0.0
                ever_both_contact = False

                obs, _ = env.reset()

            elapsed = time.perf_counter() - start
            time.sleep(max(0.0, env.control_dt - elapsed))

    env.close()