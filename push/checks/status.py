"""학습 진행 상황 한눈에 보기

    python -m push.checks.status              # results/v2 아래 전체
    python -m push.checks.status rand_pp      # 이름에 rand_pp가 들어간 run만

열 설명
    진행      : 현재 스텝 / 목표 스텝
    fps       : 초당 학습 스텝 (로그의 마지막 값)
    남은시간  : (목표 - 현재) / fps  (평가 시간은 제외한 대략값)
    범위      : 커리큘럼 range_scale (0 → 1)
    평가 최근/최고 : 학습 중 평가 성공률
    상태      : 완료 / 진행 중 / 멈춤?(로그가 10분 넘게 갱신 안 됨)
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from project_paths import PROJECT_ROOT

RESULTS = PROJECT_ROOT / "results" / "v2"
LOGS = PROJECT_ROOT / "logs"


def last_number(text: str, key: str):
    m = re.findall(rf"\|\s+{key}\s+\|\s+([-\d.e+]+)", text)
    return float(m[-1]) if m else None


def fmt_time(sec):
    if sec is None:
        return "-"
    h, m = divmod(int(sec) // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    runs = sorted(p for p in RESULTS.glob("*") if (p / "config.json").exists() and pattern in p.name)
    if not runs:
        print("해당하는 run이 없습니다:", RESULTS)
        return

    header = f"{'run':<30} {'진행':>19} {'fps':>6} {'남은시간':>8} {'범위':>5} {'평가 최근/최고':>14}  상태"
    print(header)
    print("-" * 100)
    for run in runs:
        meta = json.loads((run / "config.json").read_text(encoding="utf-8"))
        total = int(meta["total_steps"])
        log = LOGS / f"{run.name}.log"
        text = log.read_text(encoding="utf-8", errors="ignore") if log.exists() else ""

        steps = last_number(text, "total_timesteps")
        fps = last_number(text, "fps")
        scales = re.findall(r"range_scale -> ([\d.]+)", text)
        scale = scales[-1] if scales else ("1.00" if not meta.get("curriculum") else "0.00")

        if "finished" in meta:
            steps = meta.get("trained_steps", steps)
            state = "완료"
        elif log.exists() and time.time() - log.stat().st_mtime > 600:
            state = "멈춤?"
        else:
            state = "진행 중"

        remain = (total - steps) / fps if (steps and fps and state == "진행 중") else None

        ev = run / "eval" / "evaluations.npz"
        if ev.exists():
            s = np.load(ev)["successes"].mean(axis=1)
            ev_text = f"{s[-1]:.2f} / {s.max():.2f}"
        else:
            ev_text = "-"

        n = int(steps) if steps else 0
        print(
            f"{run.name:<30} {n:>9,}/{total:<9,} {int(fps) if fps else '-':>6} "
            f"{fmt_time(remain):>8} {scale:>5} {ev_text:>14}  {state}"
        )


if __name__ == "__main__":
    main()
