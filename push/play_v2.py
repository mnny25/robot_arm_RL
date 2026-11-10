"""Push v2 정책 재생 (뷰어 필요 → 로컬 PC에서 실행)

예시:
    python -m push.play_v2 --run results/v2/rand_ppo_s0
    python -m push.play_v2 --run results/v2/rand_ppo_s0 --nominal          # 기존 고정 위치
    python -m push.play_v2 --run results/v2/rand_dr_full_ppo_s0 --friction 1.8 --kp-scale 0.7
    python -m push.play_v2 --random                                         # 정책 없이 랜덤 행동 (환경 확인용)
"""

from __future__ import annotations

import argparse
import time

import mujoco.viewer
import numpy as np

from push.push_env_v2 import PushEnvV2, make_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None)
    p.add_argument("--model", choices=["best", "final"], default="best")
    p.add_argument("--random", action="store_true", help="랜덤 행동으로 환경만 확인")
    p.add_argument("--config", default="rand", help="--random일 때 사용할 설정 이름")
    p.add_argument("--nominal", action="store_true", help="기존 고정 위치에서 재생")
    p.add_argument("--friction", type=float, default=None)
    p.add_argument("--mass", type=float, default=None)
    p.add_argument("--kp-scale", type=float, default=1.0)
    p.add_argument("--delay", type=int, default=0)
    p.add_argument("--speed", type=float, default=1.0, help="재생 속도 배율")
    args = p.parse_args()

    if args.random:
        from push.configs_v2 import get_config

        env_cfg = make_config(get_config(args.config)["env"])
        policy, vec_norm = None, None
    else:
        from pathlib import Path

        from push.evaluation.evaluate_v2 import load_run

        if not args.run:
            raise SystemExit("--run 또는 --random 필요")
        meta, policy, env_cfg, vec_norm = load_run(Path(args.run), args.model)
        print("정책:", meta["run_name"])

    env = PushEnvV2(env_cfg)
    physics = {"friction": args.friction, "mass": args.mass, "kp_scale": args.kp_scale, "action_delay": args.delay}

    def reset(seed):
        opts = {"physics": physics}
        if args.nominal:
            opts.update(cube_xy=(0.22, 0.0), goal_xy=(0.30, 0.06))
        return env.reset(seed=seed, options=opts)

    episode = 0
    obs, info = reset(episode)
    n_success = 0
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = [0.25, 0.0, 0.05]
        viewer.cam.distance = 0.8
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -30
        while viewer.is_running():
            start = time.perf_counter()
            if policy is None:
                action = env.action_space.sample()
            else:
                o = vec_norm.normalize_obs(obs) if vec_norm is not None else obs
                action, _ = policy.predict(o, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            viewer.sync()
            if term or trunc:
                episode += 1
                n_success += int(info["is_success"])
                print(
                    f"[{episode}] 성공={info['is_success']} 거리={info['distance'] * 1000:.1f}mm "
                    f"최대속도={info['max_cube_speed']:.2f}m/s 시간={env.steps * env.control_dt:.1f}s "
                    f"| 누적 성공률 {n_success / episode:.2f}"
                )
                time.sleep(0.5)
                obs, info = reset(episode)
            elapsed = time.perf_counter() - start
            time.sleep(max(0.0, env.control_dt / args.speed - elapsed))
    env.close()


if __name__ == "__main__":
    main()
