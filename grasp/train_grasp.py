from project_paths import PROJECT_ROOT

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from grasp.grasp_env import GraspEnv


if __name__ == "__main__":
    torch.set_num_threads(1)

    output = (
        PROJECT_ROOT
        / "results"
        / "ppo_grasp_001"
    )
    output.mkdir(parents=True, exist_ok=True)

    env = Monitor(
        GraspEnv(),
        filename=str(output / "train"),
        info_keywords=(
            "is_success",
            "lift_height",
            "hold_time",
        ),
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        verbose=1,
        seed=0,
        device="cpu",
    )

    try:
        model.learn(total_timesteps=300_000)
    except KeyboardInterrupt:
        print("\n학습 중단 요청을 받았습니다.")
    finally:
        model.save(str(output / "ppo_grasp"))
        print("정책 저장 완료")
        env.close()