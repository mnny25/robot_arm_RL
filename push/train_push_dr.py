from project_paths import PROJECT_ROOT

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from push.push_env import PushEnv


if __name__ == "__main__":
    torch.set_num_threads(1)

    root = PROJECT_ROOT

    output_dir = root / "results" / "ppo_push_dr_001"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = Monitor(
        PushEnv(randomize_friction=True),
        filename=str(output_dir / "train"),
        info_keywords=(
            "is_success",
            "distance",
            "ee_distance",
            "friction",
        ),
    )

    # 기존에 잘 작동한 PPO 정책에서 추가 학습
    model = PPO.load(
        str(root / "results/ppo_push_003/ppo_push.zip"),
        env=env,
        device="cpu",
    )

    try:
        model.learn(
            total_timesteps=200_000,
            reset_num_timesteps=False,
        )

    except KeyboardInterrupt:
        print("\n학습 중단 요청을 받았습니다.")

    finally:
        model.save(str(output_dir / "ppo_push"))
        print("학습 결과 저장:", output_dir / "ppo_push.zip")
        env.close()