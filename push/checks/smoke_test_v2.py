"""Push v2 환경 동작 확인 (학습 전에 로컬/서버에서 한 번 실행)

    python -m push.checks.smoke_test_v2

확인 항목
 1. 모든 설정으로 환경 생성, SB3 check_env 통과, 랜덤 행동 100 스텝
 2. 태스크 랜덤화: range_scale=0이면 기존 위치, 1이면 퍼져 있는지 / 타깃 표시 이동
 3. hold 판정: 큐브를 목표에 올려놓고 가만히 두면 hold_time 후에 성공하는지
    require_stable=False면 첫 스텝에 바로 성공하는지
 4. DR: 물리값이 실제로 바뀌고, physics 미지정 시 원래 값으로 돌아오는지
"""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.env_checker import check_env

from push.configs_v2 import CONFIGS
from push.push_env_v2 import PushEnvV2, make_config

ok = True


def check(cond, msg):
    global ok
    print(("  [OK]   " if cond else "  [FAIL] ") + msg)
    ok = ok and bool(cond)


print("1) 설정별 생성 / check_env / 랜덤 100스텝")
for name, exp in CONFIGS.items():
    env = PushEnvV2(exp["env"])
    check_env(env, warn=False)
    obs, _ = env.reset(seed=0)
    finite = True
    for _ in range(100):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        finite &= bool(np.all(np.isfinite(obs)) and np.isfinite(r))
        if term or trunc:
            obs, _ = env.reset()
    check(finite and obs.shape == env.observation_space.shape, f"{name}: obs {obs.shape}, 값 정상")
    env.close()

print("\n2) 태스크 랜덤화")
env = PushEnvV2({"range_scale": 0.0})
cubes, goals = [], []
for s in range(50):
    _, info = env.reset(seed=s)
    cubes.append(info["cube_xy"])
    goals.append(info["goal_xy"])
cubes, goals = np.array(cubes), np.array(goals)
check(np.abs(cubes - [0.22, 0.0]).max() <= 0.0101, f"scale=0 큐브 최대 오차 {np.abs(cubes - [0.22, 0.0]).max() * 1000:.1f}mm (≤10mm)")
check(np.allclose(goals - cubes, [0.08, 0.06], atol=1e-6), "scale=0 목표 = 큐브 + (0.08, 0.06) (기존과 동일)")

env.set_range_scale(1.0)
cubes, goals = [], []
for s in range(300):
    _, info = env.reset(seed=s)
    cubes.append(info["cube_xy"])
    goals.append(info["goal_xy"])
cubes, goals = np.array(cubes), np.array(goals)
print(f"     scale=1 큐브 x {cubes[:, 0].min():.3f}~{cubes[:, 0].max():.3f}, y {cubes[:, 1].min():.3f}~{cubes[:, 1].max():.3f}")
print(f"     scale=1 목표 x {goals[:, 0].min():.3f}~{goals[:, 0].max():.3f}, y {goals[:, 1].min():.3f}~{goals[:, 1].max():.3f}")
check(np.ptp(cubes[:, 1]) > 0.08 and np.ptp(goals[:, 1]) > 0.1, "scale=1 위치가 넓게 퍼짐")
site = env.model.site_pos[env.target_site_id][:2]
check(np.allclose(site, goals[-1]), "뷰어 타깃 표시가 목표 위치로 이동")
env.close()


def place_on_goal_and_wait(env, n_steps):
    obs, _ = env.reset(seed=0, options={"cube_xy": (0.25, 0.0), "goal_xy": (0.25, 0.0)})
    for k in range(1, n_steps + 1):
        obs, r, term, trunc, info = env.step(np.zeros(5))
        if term:
            return k, info
    return None, info


print("\n3) hold 판정")
env = PushEnvV2({"range_scale": 0.0, "hold_time": 0.5})
k, info = place_on_goal_and_wait(env, 40)
check(k is not None and info["is_success"], f"hold 0.5s: {k} 스텝째 성공 (hold_steps={env.hold_steps}, 큐브 안착 시간 포함)")
check(k is not None and k >= env.hold_steps, "hold_steps보다 일찍 성공하지 않음")
env.close()

env = PushEnvV2({"range_scale": 0.0, "require_stable": False})
k, info = place_on_goal_and_wait(env, 5)
check(k == 1, f"require_stable=False: {k} 스텝째 성공 (기존 방식, 1이어야 함)")
env.close()

print("\n4) Domain Randomization")
env = PushEnvV2({"dr": CONFIGS["rand_dr_full"]["env"]["dr"]})
nom_f = env._nom_friction.copy()
nom_gain = env._nom_gain.copy()
fr, ms, kp, dl = [], [], [], []
for s in range(30):
    env.reset(seed=s)
    _, _, _, _, info = env.step(np.zeros(5))
    fr.append(info["friction"])
    ms.append(info["mass"])
    kp.append(info["kp_scale"])
    dl.append(info["action_delay"])
print(f"     friction {min(fr):.2f}~{max(fr):.2f}, mass {min(ms) * 1000:.0f}~{max(ms) * 1000:.0f}g, "
      f"kp_scale {min(kp):.2f}~{max(kp):.2f}, delay {sorted(set(dl))}")
check(np.ptp(fr) > 0.3 and np.ptp(ms) > 0.02 and np.ptp(kp) > 0.1, "물리값이 에피소드마다 바뀜")
g = env.model.actuator_gainprm[env.actuator_ids, 0]
b = env.model.actuator_biasprm[env.actuator_ids, 1]
check(np.allclose(g, -b), "kp 변경 시 gainprm[0] = -biasprm[1] 유지")
env.reset(seed=0, options={"physics": {}})
check(
    np.allclose(env.model.geom_friction[[env.table_geom_id, env.cube_geom_id], 0], nom_f)
    and np.allclose(env.model.actuator_gainprm[env.actuator_ids, 0], nom_gain),
    "physics={} 로 reset하면 원래 물리값으로 복원",
)
env.close()

print("\n전체 결과:", "통과" if ok else "실패 항목 있음 → 위 [FAIL] 확인")
