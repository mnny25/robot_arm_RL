from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from push_env import PushEnv


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "results" / "ppo_push_003"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Monitor가 에피소드 보상과 길이를 기록
    env = Monitor(
        PushEnv(),
        filename=str(output_dir / "train"),
        info_keywords=("is_success", "distance", "ee_distance"),
    )

    previous_model = (
        Path(__file__).resolve().parent
        / "results"
        / "ppo_push_001"
        / "ppo_push.zip"
    )

    model = PPO.load(
        str(previous_model),
        env=env,
        device="cpu",
    )

    try:
            
        model.learn(
            total_timesteps=200_000,
            reset_num_timesteps=False,
        )

        model.save(str(output_dir / "ppo_push"))
        print("Push 정책 저장 완료")

    finally:
        env.close()