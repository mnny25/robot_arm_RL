"""평가 summary CSV를 '조건(행) × 설정(열)' 성공률 표로 출력

    python -m push.evaluation.table eval_phase3_sac
    python -m push.evaluation.table eval_phase3_ppo --split grid
    python -m push.evaluation.table results/v2/eval_phase2_summary.csv --split ood

인자는 results/v2/ 아래 파일 이름(확장자·_summary 생략 가능) 또는 경로.
열 이름에서 'rand_pp'는 base로 줄여서 표시하고, 알고리즘이 섞여 있으면 '설정/알고리즘'으로 표시.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from project_paths import PROJECT_ROOT


def resolve(name: str) -> Path:
    p = Path(name)
    if p.exists():
        return p
    base = PROJECT_ROOT / "results" / "v2"
    for cand in (name, f"{name}.csv", f"{name}_summary.csv"):
        if (base / cand).exists():
            return base / cand
    raise SystemExit(f"파일을 찾을 수 없음: {name}")


def short(cfg: str) -> str:
    s = cfg.replace("rand_pp", "base", 1)
    return s.replace("base_", "", 1) if s.startswith("base_") else s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("summary")
    p.add_argument("--split", default="grid", help="nominal | grid | ood")
    args = p.parse_args()

    path = resolve(args.summary)
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8-sig")) if r["split"] == args.split]
    if not rows:
        raise SystemExit(f"split='{args.split}' 행이 없음: {path}")

    algos = {r["algo"] for r in rows}
    key = (lambda r: f"{short(r['config'])}/{r['algo']}") if len(algos) > 1 else (lambda r: short(r["config"]))
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    cols = list(dict.fromkeys(key(r) for r in rows))
    tab = {(key(r), r["condition"]): (float(r["success_mean"]), float(r["success_std"]), r["n_runs"]) for r in rows}

    w = max(16, max(len(c) for c in cols) + 2)
    print(f"== {path.name}  ({args.split} 성공률 평균±표준편차)")
    print(f"{'조건':<16}" + "".join(f"{c:>{w}}" for c in cols))
    for cond in conds:
        cells = ""
        for c in cols:
            if (c, cond) in tab:
                m, s, _ = tab[(c, cond)]
                cells += f"{f'{m:.2f}±{s:.2f}':>{w}}"
            else:
                cells += f"{'-':>{w}}"
        print(f"{cond:<16}{cells}")
    runs = {c: tab[(c, conds[0])][2] for c in cols if (c, conds[0]) in tab}
    print("run 수:", ", ".join(f"{c}={n}" for c, n in runs.items()))


if __name__ == "__main__":
    main()
