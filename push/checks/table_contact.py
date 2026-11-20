"""정책이 로봇 몸체(집게 등)를 테이블에 대고 미는지 측정

    python -m push.checks.table_contact results/v2/rand_pp_sac_s0
    python -m push.checks.table_contact results/v2/rand_pp_sac_s0 results/v2/rand_pp_ppo_s0 --episodes 20

에피소드마다 출력
    테이블 접촉 스텝 비율 : 로봇 geom이 테이블과 닿아 있던 스텝 / 전체 스텝
    EE 최저 높이          : gripperframe의 최저 z [mm]
    최대 접촉 힘          : 로봇-테이블 접촉의 법선 방향 힘 최대값 [N]
마지막에 run별 평균과, 테이블에 닿은 로봇 부품(body) 이름별 횟수

판단 기준 (대략)
    접촉 0%                 → 공중에서 밀고 있음, 문제 없음
    10% 미만, 힘 수 N 이하  → 살짝 스침
    30% 이상 또는 힘이 큼   → 테이블을 짚거나 긁으며 밈 → 접촉 페널티 / push point 높이 조정 필요
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from push.evaluation.evaluate_v2 import load_run
from push.push_env_v2 import PushEnvV2


def run_one(run_dir: Path, episodes: int, which: str):
    meta, policy, cfg, vec_norm = load_run(run_dir, which)
    env = PushEnvV2({**cfg, "recovery_start_prob": 0.0})
    m, d = env.model, env.data
    parts = Counter()
    ratios, zmins, fmaxs, succ = [], [], [], []

    print(f"\n== {meta['run_name']} ({which})")
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        steps = touch = 0
        zmin, fmax = 1.0, 0.0
        while True:
            o = vec_norm.normalize_obs(obs) if vec_norm is not None else obs
            action, _ = policy.predict(o, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            steps += 1
            hit = False
            for i in range(d.ncon):
                c = d.contact[i]
                if env.table_geom_id not in (c.geom1, c.geom2):
                    continue
                other = c.geom2 if c.geom1 == env.table_geom_id else c.geom1
                if other == env.cube_geom_id:
                    continue
                hit = True
                f = np.zeros(6)
                mujoco.mj_contactForce(m, d, i, f)
                fmax = max(fmax, abs(f[0]))
                parts[m.body(m.geom_bodyid[other]).name] += 1
            touch += int(hit)
            zmin = min(zmin, float(d.site("gripperframe").xpos[2]))
            if term or trunc:
                break
        ratios.append(touch / steps)
        zmins.append(zmin)
        fmaxs.append(fmax)
        succ.append(bool(info["is_success"]))
        print(
            f"  [{ep:2d}] 성공={str(info['is_success']):<5} 테이블 접촉 {touch:3d}/{steps:3d}스텝 ({touch / steps:4.0%}) "
            f"| EE 최저 {zmin * 1000:5.1f}mm | 최대 접촉힘 {fmax:6.1f}N"
        )
    env.close()
    print(
        f"  평균: 성공률 {np.mean(succ):.2f} | 접촉 비율 {np.mean(ratios):.0%} | "
        f"EE 최저 {np.mean(zmins) * 1000:.1f}mm | 최대 접촉힘 {np.mean(fmaxs):.1f}N"
    )
    print("  테이블에 닿은 로봇 부품:", dict(parts.most_common()) if parts else "없음")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--model", choices=["best", "final"], default="best")
    args = p.parse_args()
    for r in args.runs:
        run_one(Path(r), args.episodes, args.model)


if __name__ == "__main__":
    main()
