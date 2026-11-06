from pathlib import Path
import time

import mujoco.viewer
from stable_baselines3 import SAC

from push_env import PushEnv


if __name__ == "__main__":
    model_path = (
        Path(__file__).resolve().parent
        / "results"
        / "sac_push_002"
        / "sac_push.zip"
    )

    env = PushEnv()
    policy = SAC.load(str(model_path), device="cpu") # 저장된 정책을 불러옴

    obs, info = env.reset(seed=0)
    control_dt = env.frame_skip * env.model.opt.timestep

    episode = 1
    episode_reward = 0.0
    success_count = 0

    print("Reach 목표 위치:", env.goal)

    with mujoco.viewer.launch_passive(
        env.model, env.data
    ) as viewer:

        viewer.cam.lookat[:] = [0.25, 0.0, 0.20]
        viewer.cam.distance = 0.7
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25

        while viewer.is_running():
            start = time.perf_counter()

            # 랜덤 action 대신 학습한 정책 사용
            action, _ = policy.predict(
                obs,
                deterministic=True, # 무작위 샘플링 없이 행동을 선택함
            )

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            viewer.sync()

            if terminated or truncated:
                success_count += int(info["is_success"])

                print(
                    f"시도 {episode} | "
                    f"성공={info['is_success']} | "
                    f"EE–큐브={info['ee_distance'] * 1000:.1f} mm | "
                    f"큐브–목표={info['distance'] * 1000:.1f} mm | "
                    f"낙하={info['fallen']} | "
                    f"스텝={env.steps} | "
                    f"누적 보상={episode_reward:.2f}"
                )

                # 마지막 자세를 잠깐 확인
                time.sleep(0.8)

                if episode >= 10:
                    print(f"완료: {success_count}/10회 성공")
                    break

                episode += 1
                episode_reward = 0.0
                obs, info = env.reset()

            elapsed = time.perf_counter() - start
            time.sleep(max(0.0, control_dt - elapsed))

    env.close()