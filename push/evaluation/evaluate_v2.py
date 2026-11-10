"""Push v2 정책 평가

고정된 평가 세트(큐브·목표 위치 조합)와 물리 조건에서 정책을 반복 재현 가능하게 평가합니다.
여러 run(시드)을 한 번에 넣으면 설정·알고리즘별로 평균±표준편차를 냅니다.

예시 (프로젝트 루트에서):
    python -m push.evaluation.evaluate_v2 --runs results/v2/rand_ppo_s0 results/v2/rand_ppo_s1
    python -m push.evaluation.evaluate_v2 --runs "results/v2/rand_*" --suite friction
    python -m push.evaluation.evaluate_v2 --runs "results/v2/*" --suite all --out results/v2/eval_all.csv

평가 세트
    nominal : 기존 고정 위치 (큐브 0.22,0 → 목표 0.30,0.06)
    grid    : 학습 범위 안의 27개 조합 (큐브 3x3 x 방향 3)
    ood     : 학습 범위 밖 5개 조합 (일반화 확인용)

suite (물리 조건)
    nominal / friction / mass / kp / delay / noise / all
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from project_paths import PROJECT_ROOT
from push.push_env_v2 import PushEnvV2, make_config


def polar_goal(cube_xy, angle_deg, dist):
    a = math.radians(angle_deg)
    return (cube_xy[0] + dist * math.cos(a), cube_xy[1] + dist * math.sin(a))


def build_tasks():
    tasks = [("nominal", (0.22, 0.0), (0.30, 0.06))]
    for x in (0.19, 0.22, 0.25):
        for y in (-0.04, 0.0, 0.04):
            for ang in (-40.0, 0.0, 40.0):
                tasks.append(("grid", (x, y), polar_goal((x, y), ang, 0.09)))
    ood = [
        ((0.22, 0.09), -30.0, 0.10),
        ((0.22, -0.09), 30.0, 0.10),
        ((0.20, 0.0), 75.0, 0.10),
        ((0.20, 0.0), -75.0, 0.10),
        ((0.22, 0.0), 0.0, 0.14),
    ]
    for cube, ang, dist in ood:
        tasks.append(("ood", cube, polar_goal(cube, ang, dist)))
    return tasks


SUITES = {
    "nominal": [("nominal", {})],
    "friction": [(f"friction={f}", {"friction": f}) for f in (0.25, 0.5, 1.0, 1.5, 2.0)],
    "mass": [(f"mass={m}", {"mass": m}) for m in (0.025, 0.05, 0.1, 0.15)],
    "kp": [(f"kp_scale={k}", {"kp_scale": k}) for k in (0.6, 0.8, 1.0, 1.2, 1.4)],
    "delay": [(f"delay={d}", {"action_delay": d}) for d in (0, 1, 2, 3)],
    "noise": [(f"noise={n}", {"obs_noise_std": n}) for n in (0.0, 0.003, 0.005, 0.01)],
}


def load_run(run_dir: Path, which: str):
    meta = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    algo_cls = {"ppo": PPO, "sac": SAC}[meta["algo"]]
    model_path = run_dir / f"{which}_model.zip"
    vn_path = run_dir / f"{which}_vecnormalize.pkl"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not vn_path.exists():
        vn_path = run_dir / "final_vecnormalize.pkl"
    policy = algo_cls.load(str(model_path), device="cpu")

    env_cfg = dict(meta["env_config"])
    env_cfg["dr"] = None  # 평가에서는 물리 조건을 직접 지정
    env_cfg = make_config(env_cfg)

    vec_norm = None
    if vn_path.exists():
        vec_norm = VecNormalize.load(str(vn_path), DummyVecEnv([lambda: PushEnvV2(env_cfg)]))
        vec_norm.training = False
        vec_norm.norm_reward = False
    else:
        print(f"경고: {run_dir} 관측 정규화 통계 없음 → 정규화 없이 평가")
    return meta, policy, env_cfg, vec_norm


def run_episode(env, policy, vec_norm, task, physics, seed):
    _, cube_xy, goal_xy = task
    obs, _ = env.reset(seed=seed, options={"cube_xy": cube_xy, "goal_xy": goal_xy, "physics": physics})
    ret = 0.0
    while True:
        o = vec_norm.normalize_obs(obs) if vec_norm is not None else obs
        action, _ = policy.predict(o, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        ret += r
        if term or trunc:
            return {
                "success": bool(info["is_success"]),
                "fallen": bool(info["fallen"]),
                "steps": env.steps,
                "time_s": env.steps * env.control_dt,
                "final_dist_mm": info["distance"] * 1000,
                "max_cube_speed": info["max_cube_speed"],
                "return": ret,
            }


def summarize(rows):
    # (설정, 알고리즘, 조건, split) → run별 성공률 → 평균±표준편차
    per_run = defaultdict(list)
    extra = defaultdict(list)
    for r in rows:
        key = (r["config"], r["algo"], r["condition"], r["split"])
        per_run[(key, r["run"])].append(r["success"])
        extra[key].append((r["max_cube_speed"], r["final_dist_mm"]))
    grouped = defaultdict(list)
    for (key, _run), succ in per_run.items():
        grouped[key].append(float(np.mean(succ)))

    out = []
    for key in sorted(grouped):
        rates = grouped[key]
        sp = [e[0] for e in extra[key]]
        di = [e[1] for e in extra[key]]
        out.append({
            "config": key[0], "algo": key[1], "condition": key[2], "split": key[3],
            "n_runs": len(rates),
            "success_mean": float(np.mean(rates)),
            "success_std": float(np.std(rates)),
            "max_cube_speed_mean": float(np.mean(sp)),
            "final_dist_mm_mean": float(np.mean(di)),
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="run 폴더 (glob 가능)")
    p.add_argument("--model", choices=["best", "final"], default="best")
    p.add_argument("--suite", choices=list(SUITES) + ["all"], default="nominal")
    p.add_argument("--splits", nargs="+", default=["nominal", "grid", "ood"])
    p.add_argument("--out", default=None, help="기본: results/v2/eval_<suite>.csv")
    args = p.parse_args()
    torch.set_num_threads(1)

    run_dirs = []
    for pattern in args.runs:
        run_dirs += [Path(x) for x in sorted(glob.glob(pattern)) if (Path(x) / "config.json").exists()]
    if not run_dirs:
        raise SystemExit("평가할 run 폴더가 없습니다 (config.json이 있는 폴더).")

    suites = list(SUITES) if args.suite == "all" else [args.suite]
    conditions = [(s, c, phys) for s in suites for c, phys in SUITES[s]]
    tasks = [t for t in build_tasks() if t[0] in args.splits]

    rows = []
    for run_dir in run_dirs:
        try:
            meta, policy, env_cfg, vec_norm = load_run(run_dir, args.model)
        except FileNotFoundError as e:
            print("건너뜀:", e)
            continue
        env = PushEnvV2(env_cfg)
        print(f"\n== {meta['run_name']} ({args.model}) : {len(conditions)} 조건 x {len(tasks)} 태스크")
        for suite, cond, phys in conditions:
            for i, task in enumerate(tasks):
                res = run_episode(env, policy, vec_norm, task, phys, seed=i)
                rows.append({
                    "run": meta["run_name"], "config": meta["config_name"], "algo": meta["algo"],
                    "seed": meta["seed"], "model": args.model, "suite": suite, "condition": cond,
                    "split": task[0], "task_id": i,
                    "cube_x": task[1][0], "cube_y": task[1][1],
                    "goal_x": task[2][0], "goal_y": task[2][1],
                    **res,
                })
            done = [r for r in rows if r["run"] == meta["run_name"] and r["condition"] == cond]
            print(f"  {cond:<16} 성공률 {np.mean([r['success'] for r in done]):.2f}")
        env.close()

    out = Path(args.out) if args.out else PROJECT_ROOT / "results" / "v2" / f"eval_{args.suite}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = summarize(rows)
    sum_path = out.with_name(out.stem + "_summary.csv")
    with sum_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print("\n설정 / 알고리즘 / 조건 / split : 성공률 평균±표준편차 (run 수) | 최대 큐브속도 | 최종거리")
    for s in summary:
        print(
            f"{s['config']:<18} {s['algo']:<4} {s['condition']:<16} {s['split']:<8}: "
            f"{s['success_mean']:.2f}±{s['success_std']:.2f} (n={s['n_runs']}) | "
            f"{s['max_cube_speed_mean']:.2f} m/s | {s['final_dist_mm_mean']:.1f} mm"
        )
    print("\n저장:", out, "/", sum_path)


if __name__ == "__main__":
    main()
