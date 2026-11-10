"""Push v2 학습 스크립트 (PPO / SAC)

예시 (프로젝트 루트에서):
    python -m push.train_v2 --config fixed_hold --algo ppo --seed 0 --steps 500000 --n-envs 8
    python -m push.train_v2 --config rand --algo sac --seed 1 --steps 1000000

출력: results/v2/<config>_<algo>_s<seed>[_<tag>]/
    config.json               실행 설정·커밋·버전 기록
    best_model.zip            평가 성공률이 가장 높았던 정책
    best_vecnormalize.pkl     그때의 관측 정규화 통계
    final_model.zip / final_vecnormalize.pkl
    checkpoints/              주기 저장
    monitor/                  에피소드 로그 (환경별 csv)
    eval/evaluations.npz      학습 중 평가 기록
    tb/                       TensorBoard 로그 (tensorboard 설치 시)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import deque

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from project_paths import PROJECT_ROOT
from push.configs_v2 import get_config
from push.push_env_v2 import PushEnvV2, make_config

MONITOR_KEYS = (
    "is_success",
    "distance",
    "max_cube_speed",
    "range_scale",
    "friction",
    "mass",
)


# ----------------------------------------------------------------------
# 콜백
# ----------------------------------------------------------------------
class CurriculumCallback(BaseCallback):
    """최근 window 에피소드 성공률이 threshold 이상이면 range_scale을 increment만큼 증가."""

    def __init__(self, start=0.0, increment=0.1, threshold=0.8, window=100, verbose=1):
        super().__init__(verbose)
        self.scale = start
        self.increment = increment
        self.threshold = threshold
        self.results = deque(maxlen=window)

    def _on_training_start(self) -> None:
        self.training_env.env_method("set_range_scale", self.scale)

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                self.results.append(float(info.get("is_success", False)))
        if (
            self.scale < 1.0
            and len(self.results) == self.results.maxlen
            and np.mean(self.results) >= self.threshold
        ):
            self.scale = min(1.0, round(self.scale + self.increment, 4))
            self.training_env.env_method("set_range_scale", self.scale)
            self.results.clear()
            if self.verbose:
                print(f"[curriculum] step {self.num_timesteps}: range_scale -> {self.scale:.2f}")
        self.logger.record("curriculum/range_scale", self.scale)
        return True


class SaveVecNormalizeOnBest(BaseCallback):
    """EvalCallback에서 최고 성능이 갱신되면 관측 정규화 통계도 같이 저장."""

    def __init__(self, path):
        super().__init__(0)
        self.path = path

    def _on_step(self) -> bool:
        vec_norm = self.model.get_vec_normalize_env()
        if vec_norm is not None:
            vec_norm.save(str(self.path))
        return True


class EpisodeStatsCallback(BaseCallback):
    """TensorBoard에 학습 중 성공률·최대 큐브 속도 기록."""

    def __init__(self, window=100):
        super().__init__(0)
        self.success = deque(maxlen=window)
        self.speed = deque(maxlen=window)

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                self.success.append(float(info.get("is_success", False)))
                self.speed.append(float(info.get("max_cube_speed", 0.0)))
        if self.success:
            self.logger.record("rollout/success_rate_100", float(np.mean(self.success)))
            self.logger.record("rollout/max_cube_speed_100", float(np.mean(self.speed)))
        return True


# ----------------------------------------------------------------------
def make_env(env_cfg, rank, seed, monitor_dir):
    def _init():
        env = PushEnvV2(env_cfg)
        env = Monitor(env, filename=str(monitor_dir / f"env{rank}"), info_keywords=MONITOR_KEYS)
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        return env

    return _init


def git_info():
    def run(*args):
        try:
            return subprocess.check_output(
                ["git", *args], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            return None

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def versions():
    import gymnasium
    import mujoco
    import stable_baselines3

    return {
        "python": sys.version.split()[0],
        "mujoco": mujoco.__version__,
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }


def build_model(algo, venv, seed, device, n_envs, tb_dir):
    net = dict(net_arch=[256, 256])
    if algo == "ppo":
        return PPO(
            "MlpPolicy",
            venv,
            learning_rate=3e-4,
            n_steps=max(256, 8192 // n_envs),   # 업데이트당 약 8192 샘플
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            policy_kwargs=net,
            seed=seed,
            device=device,
            tensorboard_log=tb_dir,
            verbose=1,
        )
    return SAC(
        "MlpPolicy",
        venv,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=10_000,
        batch_size=256,
        train_freq=1,
        gradient_steps=n_envs,   # 환경 수와 맞춰 replay ratio 유지
        gamma=0.99,
        ent_coef="auto",
        policy_kwargs=net,
        seed=seed,
        device=device,
        tensorboard_log=tb_dir,
        verbose=1,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=None, help="기본: PPO 8, SAC 1")
    p.add_argument("--device", default=None, help="기본: PPO cpu, SAC auto(GPU 있으면 GPU)")
    p.add_argument("--eval-every", type=int, default=25_000, help="평가 주기 [총 스텝]")
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--tag", default="")
    p.add_argument("--out-root", default=str(PROJECT_ROOT / "results" / "v2"))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    torch.set_num_threads(1)
    n_envs = args.n_envs or (8 if args.algo == "ppo" else 1)
    device = args.device or ("cpu" if args.algo == "ppo" else "auto")

    exp = get_config(args.config)
    env_cfg = make_config(exp["env"])
    use_curriculum = bool(exp.get("curriculum", False))

    from pathlib import Path

    name = f"{args.config}_{args.algo}_s{args.seed}" + (f"_{args.tag}" if args.tag else "")
    out = Path(args.out_root) / name
    if out.exists():
        if not args.overwrite:
            sys.exit(f"이미 존재: {out}  (덮어쓰려면 --overwrite)")
        shutil.rmtree(out)
    (out / "monitor").mkdir(parents=True)

    meta = {
        "run_name": name,
        "config_name": args.config,
        "algo": args.algo,
        "seed": args.seed,
        "total_steps": args.steps,
        "n_envs": n_envs,
        "device": device,
        "curriculum": use_curriculum,
        "env_config": env_cfg,
        "git": git_info(),
        "versions": versions(),
        "host": platform.node(),
        "started": dt.datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
    }
    (out / "config.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("run_name", "n_envs", "device", "curriculum")}, ensure_ascii=False))

    # ---- 환경 ----
    fns = [make_env(env_cfg, i, args.seed, out / "monitor") for i in range(n_envs)]
    venv = SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=(args.algo == "ppo"), clip_obs=10.0)

    # 평가 환경: 항상 전체 범위(range_scale=1 설정값 그대로)에서 평가
    (out / "eval_monitor").mkdir()
    eval_venv = DummyVecEnv([make_env(env_cfg, 0, args.seed + 10_000, out / "eval_monitor")])
    eval_venv = VecNormalize(eval_venv, norm_obs=True, norm_reward=False, training=False, clip_obs=10.0)

    try:
        import tensorboard  # noqa: F401
        tb_dir = str(out / "tb")
    except ImportError:
        tb_dir = None
        print("tensorboard 미설치: TensorBoard 로그 생략 (pip install tensorboard)")

    model = build_model(args.algo, venv, args.seed, device, n_envs, tb_dir)

    callbacks = [
        EvalCallback(
            eval_venv,
            best_model_save_path=str(out),
            log_path=str(out / "eval"),
            eval_freq=max(args.eval_every // n_envs, 1),
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            callback_on_new_best=SaveVecNormalizeOnBest(out / "best_vecnormalize.pkl"),
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=max(250_000 // n_envs, 1),
            save_path=str(out / "checkpoints"),
            name_prefix="model",
            save_vecnormalize=True,
        ),
        EpisodeStatsCallback(),
    ]
    if use_curriculum:
        callbacks.append(CurriculumCallback())

    try:
        model.learn(total_timesteps=args.steps, callback=CallbackList(callbacks), progress_bar=False)
    except KeyboardInterrupt:
        print("\n학습 중단 요청 → 현재 정책 저장")
    finally:
        model.save(str(out / "final_model"))
        venv.save(str(out / "final_vecnormalize.pkl"))
        meta["finished"] = dt.datetime.now().isoformat(timespec="seconds")
        meta["trained_steps"] = int(model.num_timesteps)
        (out / "config.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        venv.close()
        eval_venv.close()
        print("저장 완료:", out)


if __name__ == "__main__":
    main()
