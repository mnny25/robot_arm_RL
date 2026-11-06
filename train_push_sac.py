from pathlib import Path
import torch
import time

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from push_env import PushEnv


if __name__ == "__main__":
    torch.set_num_threads(1)
    print("PyTorch 스레드 수:", torch.get_num_threads())
    
    output_dir = (
        Path(__file__).resolve().parent
        / "results"
        / "sac_push_002"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    env = Monitor(
        PushEnv(),
        filename=str(output_dir / "train"),
        info_keywords=("is_success", "distance", "ee_distance"),
    )

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=256,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        verbose=1,
        seed=0,
        device="cpu",
    )

    start = time.perf_counter()

    try:
        model.learn(
            total_timesteps=200_000,
            reset_num_timesteps=False,
        )

    except KeyboardInterrupt:
        # 터미널에서 Ctrl+C를 누르면 여기로 이동
        print("\n학습 중단 요청을 받았습니다.")

    finally:
        model.save(str(output_dir / "sac_push"))
        print("정책 저장 완료:", output_dir / "sac_push.zip")

        env.close()