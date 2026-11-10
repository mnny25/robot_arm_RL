"""로봇 EE(gripperframe)가 테이블 위 밀기 높이에서 닿을 수 있는 영역 확인

랜덤 관절 각도를 많이 뽑아 순기구학(mj_forward)으로 EE 위치를 계산하고,
높이가 밀기에 적당한 경우(z 0.5~3.5cm)만 모아 xy 범위를 보여줍니다.
충돌은 고려하지 않는 대략적인 확인이므로, 결과보다 조금 안쪽으로 범위를 잡으세요.

    python -m push.checks.check_workspace
    python -m push.checks.check_workspace --samples 200000
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np

from project_paths import PROJECT_ROOT
from push.push_env_v2 import ARM_JOINTS, DEFAULT_CONFIG


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=100_000)
    p.add_argument("--zmin", type=float, default=0.005)
    p.add_argument("--zmax", type=float, default=0.035)
    args = p.parse_args()

    model = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / "modelingfile" / "robot" / "scene.xml"))
    data = mujoco.MjData(model)
    jids = np.array([model.joint(n).id for n in ARM_JOINTS])
    qadr = model.jnt_qposadr[jids]
    lo, hi = model.jnt_range[jids, 0], model.jnt_range[jids, 1]
    site = model.site("gripperframe").id

    mujoco.mj_forward(model, data)
    print("초기 자세 gripperframe 위치 [m]:", np.round(data.site_xpos[site], 3))

    rng = np.random.default_rng(0)
    pts = []
    for _ in range(args.samples):
        data.qpos[qadr] = rng.uniform(lo, hi)
        mujoco.mj_kinematics(model, data)
        x = data.site_xpos[site]
        if args.zmin <= x[2] <= args.zmax:
            pts.append(x[:2].copy())
    pts = np.array(pts)
    print(f"밀기 높이 샘플: {len(pts)} / {args.samples}")
    if len(pts) == 0:
        return

    print("x 범위 (1~99%):", np.round(np.percentile(pts[:, 0], [1, 99]), 3))
    print("y 범위 (1~99%):", np.round(np.percentile(pts[:, 1], [1, 99]), 3))

    # 2cm 격자 텍스트 지도 (위쪽 = +x, 로봇은 아래쪽 x=0)
    cell = 0.02
    xs = np.arange(0.0, 0.46, cell)
    ys = np.arange(-0.30, 0.301, cell)
    grid = np.zeros((len(xs), len(ys)), dtype=int)
    ix = np.clip(((pts[:, 0] - xs[0]) / cell).astype(int), 0, len(xs) - 1)
    iy = np.clip(((pts[:, 1] - ys[0]) / cell).astype(int), 0, len(ys) - 1)
    np.add.at(grid, (ix, iy), 1)

    wx, wy = DEFAULT_CONFIG["workspace_x"], DEFAULT_CONFIG["workspace_y"]
    print("\n도달 지도 (#: 닿음, .: 안 닿음, 대문자: 설정된 workspace 안, C: 기존 큐브, G: 기존 목표)")
    print("y: -0.30 ... +0.30 (왼→오), x: 아래 0.00 → 위 0.44")
    for i in reversed(range(len(xs))):
        row = ""
        for j in range(len(ys)):
            x, y = xs[i] + cell / 2, ys[j] + cell / 2
            if abs(x - 0.22) < cell / 2 and abs(y - 0.0) < cell / 2:
                row += "C"
                continue
            if abs(x - 0.30) < cell / 2 and abs(y - 0.06) < cell / 2:
                row += "G"
                continue
            inside = wx[0] <= x <= wx[1] and wy[0] <= y <= wy[1]
            ch = "#" if grid[i, j] > 0 else "."
            row += ch.replace("#", "W").replace(".", "x") if inside else ch
        print(f"{xs[i]:5.2f} {row}")
    print("\nW: workspace 안 + 닿음, x: workspace 안인데 안 닿음 → x가 보이면 workspace를 줄이세요.")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(pts[:, 1], pts[:, 0], s=1, alpha=0.2)
        ax.add_patch(plt.Rectangle((wy[0], wx[0]), wy[1] - wy[0], wx[1] - wx[0], fill=False, color="r", label="workspace"))
        ax.plot(0.0, 0.22, "s", color="orange", label="nominal cube")
        ax.plot(0.06, 0.30, "o", color="green", label="nominal goal")
        ax.set_xlabel("y [m]")
        ax.set_ylabel("x [m]")
        ax.set_aspect("equal")
        ax.legend(loc="lower right")
        out = PROJECT_ROOT / "results" / "v2" / "workspace.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print("그림 저장:", out)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
