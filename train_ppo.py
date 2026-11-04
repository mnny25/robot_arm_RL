from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from reach_env import ReachEnv


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "results" / "ppo_reach"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Monitor가 에피소드 보상과 길이를 기록
    env = Monitor(
        ReachEnv(),
        filename=str(output_dir / "train"),
        info_keywords=("is_success", "distance"),
    )

    model = PPO(
        "MlpPolicy", # 관측 벡터를 입력받는 신경망 정책
        env,
        learning_rate=3e-4,
        n_steps=1024, # 환경과 1024번 상호작용한 뒤 정책 업데이트
        batch_size=64, # 수집한 데이터를 64개씩 묶어서 학습
        verbose=1,
        seed=0, # 난수 설정
        device="cpu",
    )

    try:
        model.learn(total_timesteps=100_000) # 총 10만번의 action으로 첫 학습

        model.save(str(output_dir / "ppo_reach"))
        print("학습 완료. 정책 저장:", output_dir / "ppo_reach.zip")

    finally:
        env.close()