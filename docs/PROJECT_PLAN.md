# RL 기반 Robot Arm 조작 — 프로젝트 계획 · 환경 셋팅 · 설계 가이드

> **이 문서의 용도**
> 사람(학부연구생)과 AI 에이전트가 이 저장소에서 작업을 이어받을 때 **가장 먼저 읽는 문서**다.
> 과제 목표, 현재까지의 결과, 앞으로의 단계별 계획, 환경 셋팅, 설계 원칙, 작업 규칙을 한곳에 정리했다.
> 에이전트는 §8 "작업 규칙"을 반드시 지킨다.
>
> - 작성: 2026-11-16 · 기준 커밋 이후 결과는 `results/v2/*/config.json`의 `git.commit`으로 추적
> - 관련 문서: `push/README_v2.md` (v2 실행 방법), `README.md` (v1 기록)

---

## 목차
1. [과제 개요와 Subtask 현황](#1-과제-개요와-subtask-현황)
2. [현재까지의 결과 요약](#2-현재까지의-결과-요약)
3. [저장소 구조](#3-저장소-구조)
4. [환경 셋팅](#4-환경-셋팅)
5. [시뮬레이션 환경 설계 (Push v2)](#5-시뮬레이션-환경-설계-push-v2)
6. [학습 · 평가 설계](#6-학습--평가-설계)
7. [앞으로의 계획 (단계별 상세)](#7-앞으로의-계획-단계별-상세)
8. [작업 규칙 (사람·에이전트 공통)](#8-작업-규칙-사람에이전트-공통)
9. [자주 겪은 문제와 해결](#9-자주-겪은-문제와-해결)
10. [명령어 모음](#10-명령어-모음)
11. [용어 정리와 참고 자료](#11-용어-정리와-참고-자료)

---

## 1. 과제 개요와 Subtask 현황

**목표**: SO-101 로봇암으로 테이블 위 큐브를 목표 위치까지 미는(Push) 정책을 강화학습(PPO/SAC)으로 시뮬레이션(MuJoCo)에서 학습하고, Domain Randomization을 거쳐 실물 로봇에 배포(Sim-to-Real)한 뒤 Gap을 분석한다.

| # | Subtask | 상태 | 비고 |
|---|---|---|---|
| 1 | PPO / SAC 세미나 | ✅ 완료 | |
| 2 | MuJoCo 설치 · 기본 구조 이해 | ✅ 완료 | |
| 3 | MuJoCo 모델 구성 (SO-101 MJCF) | ✅ 완료 | `modelingfile/robot/` |
| 4 | 로봇암 기본 제어 | 🟡 부분 완료 | 단일 관절 제어 완료. **End-Effector(IK) 제어 미완** → §7 P7 |
| 5 | Task 환경 구성 (Robot/Table/Cube/Target) | ✅ 완료 | `scene.xml` |
| 6 | Gymnasium 환경 구현 | ✅ 완료 | `push/push_env_v2.py` |
| 7 | Action Space 설계 | ✅ 1차 완료 | 관절 목표 증분. wrist_roll 고정 검토 → §7 P5 |
| 8 | Observation Space 설계 | ✅ 1차 완료 | full 31차원 / real 21차원. 과거 행동 포함 검토 → §7 P4 |
| 9 | Reward 설계 · Shaping | ✅ 1차 완료 | hold 성공, 과속 페널티, push point |
| 10 | PPO/SAC 정책 학습 | ✅ 완료 | 위치 랜덤화에서 SAC 우세 |
| 11 | Domain Randomization | 🟡 진행 중 | 전체 DR 학습 실패 → 원인 분석 중 (§7 P4) |
| 12 | Sim-to-Real 배포 | ⬜ 대기 | **실물 로봇 사용 일정 미정** → 시뮬레이션에서 준비 (§7 P6, P8) |
| 13 | Sim-to-Real Gap 분석 · 개선 | ⬜ 대기 | §7 P9 |
| 14 | 결과 정리 (세미나) | ⬜ 대기 | §7 P10 |

> Grasp(`grasp/`)는 테스트용으로 만든 것이며 현재 범위 밖이다. Push에 집중한다.

---

## 2. 현재까지의 결과 요약

모든 수치는 `push/evaluation/evaluate_v2.py`의 고정 평가 세트 기준, 시드 3개 평균 ± 표준편차.

### Phase 1 — 고정 위치, 성공 조건 (PPO)
| 설정 | 학습량 | 성공 시드 | 성공 시 큐브 최고속도 |
|---|---|---|---|
| fixed_nohold (닿는 순간 성공) | 50만 | 2/3 | 0.27 m/s |
| fixed_hold (0.5초 정지 유지) | 50만 | 1/3 | — |
| fixed_hold | 150만 | 3/3 | 0.38 m/s |
| fixed_hold_pp (+push point) | 150만 | 3/3 | 0.31 m/s |

- 성공을 처음 발견하는 시점이 시드에 따라 **25만~115만 스텝**으로 크게 다름 → 학습량 부족이 50만 실패의 원인.
- 속도 감소(구버전 0.78 m/s → 0.3 m/s대)는 hold 판정 덕분이 **아님**(대조군 nohold도 0.27). hold는 "멈춰서 성공"이라는 정의를 현실에 맞춘 것.
- push point는 고정 위치에서 뚜렷한 효과 없음(최종 안정성만 약간 우세).

### Phase 2 — 큐브·목표 위치 랜덤화 + 커리큘럼 (`rand_pp`)
| | PPO (300만) | SAC (150만) |
|---|---|---|
| grid (학습 범위 안 27개) | 0.89 ± 0.08 | **0.98 ± 0.02** |
| ood (학습 범위 밖 5개) | 0.40 ± 0.16 | **0.73 ± 0.09** |
| nominal | 1.00 | 1.00 |
| 학습 시간 | **약 45분** | 약 5시간 |

- 고정 위치로만 학습한 정책의 grid 성공률은 0.06 → **위치 랜덤화가 일반화를 만든다.**
- SAC가 샘플 효율·일반화·시드 안정성 모두 우세. PPO는 `ent_coef=0` 기본 설정이라는 점 유의.

### Phase 3 — Domain Randomization
**커리큘럼 도달 범위 (0→1)**: rand_pp(DR 없음) 모두 1.00 / dr_friction PPO 1.0·0.8·1.0 / **dr_full PPO 0.4·0.4·0.3, SAC 0.7·0.2·0.4** / dr_full_real PPO 0.7·0.5·0.1, SAC 0.6·0.4·0.2
→ **전체 DR은 학습 자체를 막음** (grid 성공률 약 0.3, 조건과 무관하게 평평).

**DR 없는 SAC의 민감도 (grid 성공률, 기본 0.98)**
| 영향 거의 없음 (≥0.91) | 약점 |
|---|---|
| kp 0.6~1.4, 지연 0~2스텝, 질량 25~150g, 관측노이즈 ≤5mm | **마찰 2.0 → 0.64**, 마찰 1.5 → 0.80, **관측노이즈 1cm → 0.75**, 지연 3스텝 → 0.86, 마찰 0.25 → 0.86 |

**결과 대기**: Phase 3 PPO 평가 표(`eval_phase3_ppo*`), 원인 분석 실험 `phase3diag`(dr_delay / dr_obs / dr_dyn).

### 핵심 교훈 (세미나용)
1. RL은 시드 분산이 크다 → 시드 ≥3, 같은 조건의 대조군 필수.
2. 성공 전 스텝 보상 ≤ 0 원칙 → 보상 해킹 방지.
3. off-policy(SAC)는 샘플 효율, on-policy(PPO)는 시간 효율.
4. DR은 다 넣는 게 아니라 **민감도 분석 → 약점에만** 넣는다. 범위는 실측 근거로.

---

## 3. 저장소 구조

```
robot_arm_rl/
├─ project_paths.py              # PROJECT_ROOT 정의 (모든 경로의 기준)
├─ modelingfile/robot/
│   ├─ so101_new_calib.xml       # SO-101 MJCF (position 액추에이터 kp=998.22, kv=2.731)
│   ├─ scene.xml                 # Push 장면: table_top, cube(cube_free 관절, cube_geom), target site
│   └─ assets/                   # STL 메쉬
├─ push/
│   ├─ push_env.py               # v1 환경 (보존용, 수정 금지)
│   ├─ push_env_v2.py            # ★ v2 환경 (설정 기반)
│   ├─ configs_v2.py             # ★ 실험 설정 이름 → 환경 설정
│   ├─ train_v2.py               # ★ PPO/SAC 학습 (CLI)
│   ├─ play_v2.py                # 뷰어 재생 (로컬 전용)
│   ├─ run_experiments.sh        # 서버 일괄 실행 (phase 단위)
│   ├─ README_v2.md              # v2 사용법
│   ├─ evaluation/evaluate_v2.py # ★ 고정 평가 세트 + 물리 조건 스윕
│   └─ checks/
│       ├─ smoke_test_v2.py      # 환경 자동 검증 (학습 전 필수)
│       ├─ check_workspace.py    # EE 도달 영역 확인
│       └─ status.py             # 학습 진행 현황표
├─ reach/, grasp/                # 이전 작업 (범위 밖)
├─ results/                      # v1 결과 (보존)
│   └─ v2/<config>_<algo>_s<seed>[_<tag>]/   # v2 결과 (아래 참고)
├─ logs/                         # 서버 학습 로그 (git 제외)
└─ docs/PROJECT_PLAN.md          # 이 문서
```

**v2 run 폴더 구성**
| 파일 | 내용 |
|---|---|
| `config.json` | 설정 전체, git 커밋, 패키지 버전, 실행 명령, 학습 스텝 |
| `best_model.zip` + `best_vecnormalize.pkl` | 학습 중 평가 최고 정책 + **그때의 관측 정규화 통계 (항상 쌍으로 사용)** |
| `final_model.zip` + `final_vecnormalize.pkl` | 마지막 정책 |
| `eval/evaluations.npz` | 학습 중 평가 기록 (`successes`, `timesteps`) |
| `monitor/`, `eval_monitor/`, `checkpoints/`, `tb/` | 에피소드 로그, 체크포인트, TensorBoard (git 제외) |

---

## 4. 환경 셋팅

### 4.1 로컬 PC (Windows, 코드 편집 · 뷰어 재생)
```powershell
cd E:\Lab\robot_arm_rl
python -m venv .venv ; .venv\Scripts\activate
pip install mujoco gymnasium stable-baselines3 torch tensorboard matplotlib
python -m push.checks.smoke_test_v2        # 전부 [OK]
```
- 뷰어(`play_v2`)는 화면이 있는 로컬에서만 동작.
- 모든 명령은 **프로젝트 루트에서 `python -m 패키지.모듈`** 형식으로 실행 (`project_paths` import 때문).

### 4.2 GPU 서버 (Linux, 학습 · 평가)
- 접속: VS Code Remote-SSH (연구실 Wi-Fi에서만 접속 가능). 서버명 `devbox`, 32코어, RTX A6000 ×5 + RTX PRO 6000 ×2, 공유 서버.
```bash
cd ~ && git clone https://github.com/mnny25/robot_arm_RL.git robot_arm_rl   # 최초 1회
cd ~/robot_arm_rl
python3 -m venv .venv && source .venv/bin/activate
pip install mujoco gymnasium stable-baselines3 torch tensorboard
python -m push.checks.smoke_test_v2
```
- 패키지 버전은 로컬과 맞춘다(`pip freeze`로 확인). 특히 mujoco, stable-baselines3.
- conda `(base)`가 같이 켜져 있으면 `which python`이 `.venv`를 가리키는지 확인.

**GPU 번호 주의 (중요)**
`nvidia-smi` 번호와 CUDA 번호가 다르다 (CUDA는 빠른 GPU부터 번호를 매김).
| nvidia-smi | GPU | `CUDA_VISIBLE_DEVICES` |
|---|---|---|
| 0~4 | RTX A6000 | 2~6 (nvidia-smi 번호 + 2) |
| 5, 6 | RTX PRO 6000 | 0, 1 (추정, 사용 전 확인) |

확인: `CUDA_VISIBLE_DEVICES=3 python -c "import torch; print(torch.cuda.get_device_name(0))"`
대안: `CUDA_DEVICE_ORDER=PCI_BUS_ID`를 붙이면 nvidia-smi 번호와 일치.

**자원 사용 기준**
- PPO 학습 1개 ≈ `N_ENVS + 1` 코어, GPU 미사용(스크립트가 `CUDA_VISIBLE_DEVICES=`로 숨김).
- SAC 학습 1개 ≈ 1코어 + GPU 약 350MB. 여러 개가 GPU 1장을 공유해도 됨(6개까지 확인).
- 전체 사용량은 32코어의 절반~2/3 이내, `nice -n 19`로 우선순위를 낮춘다. 장시간 사용 시 선배에게 공유.
- 확인: `uptime`(load average), `htop -u $USER`, `nvidia-smi`.

**tmux 필수**: 10분 넘는 작업은 반드시 tmux 안에서 실행 (VS Code 연결이 끊기면 터미널 프로세스가 종료됨).
```bash
tmux new -s <이름>     # 새 세션 / 이미 있으면 tmux attach -t <이름>
Ctrl+b → d             # 빠져나오기 (작업은 계속)
tmux ls                # 세션 목록
```

**git 인증**: VS Code가 오래된 터미널·tmux 안에서는 인증 통로(`vscode-git-*.sock`)가 끊겨 push가 실패한다.
→ 근본 해결: 서버에 GitHub SSH 키 등록
```bash
ssh-keygen -t ed25519 -C "devbox" && cat ~/.ssh/id_ed25519.pub   # GitHub → Settings → SSH keys에 등록
git remote set-url origin git@github.com:mnny25/robot_arm_RL.git
ssh -T git@github.com
```

### 4.3 코드 · 결과 이동 흐름
```
[로컬] 코드 수정 → git commit → git push
[서버] git pull → (smoke test) → tmux에서 학습 → 평가
[서버] git add -f results/v2/<run>/{config.json,best_*} eval_*.csv → commit → pull → push
[로컬] git pull → play_v2로 재생
```
- 코드는 **로컬에서만** 수정, 서버는 pull만 (충돌 방지).
- 결과는 `.gitignore`로 제외되어 있으므로 `git add -f`로 필요한 파일만 올린다.

---

## 5. 시뮬레이션 환경 설계 (Push v2)

### 5.1 MDP 정의
| 요소 | 정의 |
|---|---|
| 제어 주기 | 20Hz (`control_dt=0.05`, `frame_skip = control_dt / model.opt.timestep`) |
| 에피소드 | 최대 200스텝(10초). 성공·낙하 시 terminated, 시간 초과 시 truncated |
| 행동 | 5차원 [-1,1]: 팔 관절(shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll) 목표각 **증분**, `joint_target += 0.03 × a` 후 관절 한계로 clip. 그리퍼 미사용 |
| 관측 full (31) | q(5), q̇(5), joint_target(5), EE→큐브(3), 큐브→목표(3), 큐브 쿼터니언(4), 큐브 속도(6) |
| 관측 real (21) | full에서 큐브 쿼터니언·속도 제외 (카메라로 얻기 어려운 값) |
| 할인율 | γ = 0.99 (유효 지평 ≈ 100스텝 ≈ 5초) |

**설계 원칙**
- 관측은 **상대 좌표**(EE→큐브, 큐브→목표) → 위치 일반화에 유리.
- 증분 액션이므로 `joint_target`을 관측에 포함 (누적 명령 상태).
- 증분 액션의 부작용: 보상과 무관한 관절(wrist_roll)은 작은 출력이 누적되어 **계속 회전**할 수 있음 → P5에서 고정 검토.

### 5.2 성공 · 종료 조건
- `in_zone`: 큐브-목표 xy 거리 < 3cm **그리고** 큐브 중심 높이 2cm ± 1cm
- `stable`: in_zone **그리고** 큐브 선속도 < 3cm/s **그리고** 각속도 < 0.5rad/s
- `success`: stable이 **연속** `hold_steps`(=ceil(0.5/0.05)=10) 스텝 (끊기면 0부터)
- `fallen`: 큐브 z < −2cm → terminated + 페널티
- `require_stable=False`로 구버전(닿는 순간 성공) 재현 가능

### 5.3 보상
```
r = − w_ee · d(EE, 목표점)            # 목표점: 큐브 중심 or push point(큐브의 목표 반대편 1.5cm)
    − w_goal · d(큐브, 목표)
    + w_progress · (이전 거리 − 현재 거리)   # potential-based, 경로와 무관
    − w_action · ‖a‖²
    − w_cube_speed · max(0, v_cube − 0.15)   # 쳐서 날리기 방지
    − w_zone_speed · v_cube   (목표 6cm 이내)  # 목표 근처에서 정지 유도
    − time_penalty
    + 10 (성공)   − 10 (낙하)
```
기본 가중치: `w_ee=1(push point 사용 시 2), w_goal=5, w_progress=20, w_action=0.01, w_cube_speed=1, w_zone_speed=2`

**보상 설계 규칙 (반드시 지킬 것)**
1. **성공 전 스텝 보상은 항상 ≤ 0.** 양수 구간(예: 원 안에서 +1)을 만들면 성공(종료)을 피하고 보상을 모으는 해킹이 생긴다.
2. 반대로 음수가 크면 **빨리 끝내려는 경향**이 생긴다. 낙하 페널티는 "남은 스텝 × 평균 스텝 페널티"보다 커야 "일부러 떨어뜨리기"가 생기지 않는다. 작업 범위를 테이블 가장자리 근처로 넓힐 때 재점검.
3. 새 항목을 추가하면 **항목별 크기를 표로 계산**해 다른 항과 비교한다(`info`에 항목별 값 기록 권장).
4. 보상 변경 후에는 반드시 고정 위치 짧은 학습(150만 스텝, 시드 3)으로 회귀 확인.

### 5.4 태스크 랜덤화 (초기 상태 분포)
- `range_scale` s ∈ [0,1]: 0이면 기존 고정 위치(큐브 0.22,0 / 목표 0.30,0.06), 1이면 전체 범위.
  - 큐브: x 0.18~0.26, y −0.06~0.06 (+ 항상 ±1cm jitter)
  - 목표: 큐브에서 6~12cm, 방향 ±60°(+x = 로봇 바깥쪽)
  - 큐브 yaw: 현재 0 (미적용) → P5
- 작업 영역(`workspace_x/y`) 밖이면 재샘플, 큐브가 로봇과 겹치면 재샘플.
- `reset(options={"cube_xy", "goal_xy", "physics", "range_scale"})`로 평가 시 직접 지정.

### 5.5 Domain Randomization
- `_sample_physics()`(샘플) / `_apply_physics()`(적용) 분리. **항상 원래 값에서 출발**해 적용 → 누적 버그 없음.
- 지원 항목: friction(테이블·큐브 동시), mass(+관성 비례), kp_scale(`gainprm[0]`과 `biasprm[1]`을 **함께** 변경), damping_scale, obs_noise_std(매 스텝), cube_pos_bias_std(에피소드 고정), action_delay_max(명령 큐).
- 물리값은 에피소드 단위로 샘플, 관측 노이즈만 매 스텝.
- **DR 범위 결정 원칙**: ① DR 없이 학습 → ② 민감도 분석(evaluate_v2 `--suite all`) → ③ 약점에만 DR → ④ 실측이 생기면 실측 중심값 ± 측정오차로 조정.

---

## 6. 학습 · 평가 설계

### 6.1 실험 설정 추가 방법
1. `push/configs_v2.py`의 `CONFIGS`에 이름 추가: `{"env": {...}, "curriculum": bool}`. `env` 키는 `DEFAULT_CONFIG`에 있는 것만 허용(오타 시 에러).
2. 환경 기능을 새로 넣었다면 `DEFAULT_CONFIG`에 **기본값은 기존 동작과 같게** 키를 추가.
3. `python -m push.checks.smoke_test_v2`로 새 설정까지 자동 검증.
4. 필요하면 `run_experiments.sh`에 phase 추가.

### 6.2 학습 기본값 (`train_v2.py`)
| | PPO | SAC |
|---|---|---|
| 환경 수 | 8 (서버에선 6 사용, **비교 실험끼리 동일하게**) | 1 |
| 장치 | CPU | GPU(auto) |
| 신경망 | MLP [256, 256] | MLP [256, 256] |
| 주요 값 | n_steps≈8192/n_envs, batch 256, epoch 10, clip 0.2, GAE 0.95, ent_coef 0 | buffer 1M, learning_starts 1만, batch 256, gradient_steps=n_envs, ent_coef auto |
| 권장 학습량 | 고정 위치 150만 / 랜덤화 300만 | 랜덤화 150만 |
| 소요 시간(서버) | 300만 ≈ 45~60분 | 150만 ≈ 5~8시간 |

공통: `VecNormalize`(PPO는 보상 정규화 포함), 2.5만 스텝마다 평가 50에피소드(평가 환경은 항상 전체 범위), 최고 성능 시 정규화 통계 함께 저장, 25만 스텝 체크포인트, 커리큘럼(최근 100에피소드 성공률 ≥ 80% → range_scale +0.1).

### 6.3 평가 프로토콜 (`evaluate_v2.py`)
- 태스크 세트: `nominal`(1) / `grid`(학습 범위 안 27) / `ood`(범위 밖 5). 결정적 정책(deterministic).
- 물리 조건 suite: `nominal`, `friction`(0.25~2.0), `mass`(25~150g), `kp`(0.6~1.4), `delay`(0~3), `noise`(0~1cm), `all`.
- 평가 시 DR은 끄고 조건을 **직접 지정** (학습은 랜덤하게, 평가는 통제해서).
- 출력: `<out>.csv`(에피소드별) + `<out>_summary.csv`(설정·알고리즘·조건·split별 평균±표준편차, 성공 에피소드만의 최고 속도).
- 주의: `best_model`은 여러 평가 중 최고를 고른 것(선택 편향) → 성능 보고는 반드시 evaluate_v2 결과로.

### 6.4 결과 해석 체크리스트
- [ ] 시드 3개 모두 보았는가 (평균만 보지 않기)
- [ ] 비교 대상과 학습량·환경 수·평가 세트가 같은가
- [ ] 커리큘럼이 1.00에 도달했는가 (미도달이면 grid 성능 비교 무의미)
- [ ] 성공률 외에 큐브 속도, 최종 거리, 소요 시간도 보았는가
- [ ] 재생으로 실패 유형을 눈으로 확인했는가 (도달 실패 / 밀고 포기 / 방향 오류 / 관절 회전 / 테이블 접촉)

---

## 7. 앞으로의 계획 (단계별 상세)

> 실물 로봇 사용 일정이 정해지지 않았으므로 **P4~P7은 시뮬레이션만으로 진행**하고, P8~P9는 실물 사용 가능 시점에 맞춘다.
> 각 단계는 "목적 → 할 일 → 설계 → 완료 기준" 순서. 완료 기준을 만족해야 다음 단계로.

### P4. DR 학습 저하 원인 분석 · DR 재설계 (Subtask 11) — **현재 단계**
**목적**: 전체 DR에서 학습이 막힌 원인을 찾고, 필요한 만큼만 DR을 넣은 정책을 만든다.

**할 일**
1. Phase 3 PPO 평가 표 확인: `dr_friction`이 마찰 1.5~2.0에서 base보다 나은지.
2. `phase3diag` 결과 확인 (dr_delay / dr_obs / dr_dyn, PPO 시드 0): 커리큘럼 도달 범위와 성공률.
3. 원인별 대응:
   | 막힌 실험 | 원인 | 대응 설계 |
   |---|---|---|
   | dr_delay | 지연을 관측할 수 없음(POMDP) | **관측에 직전 행동 k개(k=2~3) 추가** (`obs_history` 옵션). 지연 범위는 실측 전까지 0~1스텝으로 축소 |
   | dr_obs | 노이즈가 hold 판정 방해 | 노이즈 3mm 유지 + 관측 이력 추가, 또는 범위 축소 |
   | dr_dyn | 모터 응답 변화 과다 | kp·damping 범위 ±10%로 축소 (민감도상 불필요할 가능성 높음) |
   | 모두 정상 | 요소 결합 + 커리큘럼 기준 과엄격 | **커리큘럼 임계값을 인자로**(`--curriculum-threshold`, DR 시 0.6~0.7). 또는 DR 세기도 range_scale에 비례해 증가 |
4. 재설계 DR 설정 예: `rand_pp_dr_v2 = 마찰 0.5~2.0 + 관측노이즈 3mm + 바이어스 3mm (+ 관측 이력)`. **SAC로** 시드 3, 150만.
5. 평가: `--suite all`로 base(rand_pp_sac)와 비교.

**완료 기준**
- 커리큘럼 1.00 도달 (시드 3개 모두)
- 기본 조건 grid 성공률 ≥ 0.9
- 마찰 2.0 grid 성공률이 base(0.64)보다 유의하게 높음 (예: ≥ 0.85)
- 관측노이즈 1cm에서 base(0.75)보다 높음

### P5. 실물 조건을 고려한 환경 보완 (Subtask 7 · 9 보완)
**목적**: 실물에서 반드시 마주치지만 지금 학습 분포에 없는 상황을 추가한다.

| 항목 | 현재 | 설계 | 설정 키(제안) |
|---|---|---|---|
| 큐브 회전 | yaw 0° 고정 | ±45° 랜덤 (이미 구현, 값만 변경) | `cube_yaw_half_range_deg: 45` |
| 팔 초기 자세 | 항상 동일 | 관절별 ±0.1rad 랜덤 | `init_qpos_noise: 0.1` (신규) |
| wrist_roll 회전 | 행동에 포함 → 누적 회전 | 행동에서 제외, 초기각 고정 (행동 4차원) | `fixed_joints: ["wrist_roll"]` (신규, 실습 1) |
| EE–테이블 접촉 | 확인 안 됨 | 접촉 비율·힘 측정 → 많으면 접촉 페널티 + push point 높이 상향 | `w_table_contact`, `push_point_height` (신규) |
| 성공 후 동작 | 학습 안 됨 | 실물 실행 코드에서 성공 판정 시 정지(1차). 필요 시 성공 후 N스텝 유지 학습(2차) | 배포 코드 / `post_success_steps` |
| 에피소드 길이 | 10초 | 15초 비교 실험 | `max_steps: 300` |

**작업 순서**: ① 테이블 접촉 측정(스크립트로 접촉 비율·힘·부품) → ② 항목별 설정 추가 + smoke test 갱신 → ③ `rand_pp_v3`(위 항목 통합) SAC 시드 3 학습 → ④ rand_pp_sac 대비 성능 유지 확인 → ⑤ 회전된 큐브·다른 초기자세 평가 세트 추가(evaluate_v2 태스크에 yaw 포함).

**완료 기준**: grid 성공률 ≥ 0.9 유지, 큐브 yaw ±45° 평가 성공률 ≥ 0.85, 재생 시 관절 연속 회전·테이블 긁기 없음.

> 주의: 행동 차원을 바꾸면(4차원) **기존 정책과 호환되지 않는다**. 새 run으로만 비교.

### P6. 실물 관측 기반 정책 (Subtask 8 보완, Sim-to-Real 준비)
**목적**: 실물에서 얻을 수 있는 정보만으로 동작하는 최종 후보 정책을 만든다.

**설계**
- 관측: `obs_mode: "real"` + **관측 이력**(직전 행동 2~3개 또는 큐브 위치 2~3프레임). 큐브 속도는 위치 차분으로 정책이 추정.
- 관측 오차 모델(카메라 가정): 매 프레임 노이즈 σ, 에피소드 고정 바이어스 σ_b, 인식 지연 0~2스텝, **가림(occlusion)**: 확률 p로 이번 프레임 큐브 위치를 직전 값으로 유지(신규 `obs_dropout_prob`).
- 값은 실측 전 가정: σ=3mm, σ_b=3mm, p=0.1. 실측되면 교체.
- DR: P4에서 확정한 것 + 위 관측 오차.
- 알고리즘: SAC, 시드 3, 150만~300만.

**완료 기준**: full 관측 정책 대비 grid 성공률 하락 ≤ 0.05, 관측노이즈 5mm·가림 0.2 조건에서 ≥ 0.85.

### P7. End-Effector 제어 구현 (Subtask 4 잔여)
**목적**: 과제 요구사항 충족 + 디버깅·배포 도구 + 대안 액션 공간 실험.

**설계**
- 위치 IK: `mujoco.mj_jacSite`로 `gripperframe` 위치 Jacobian(3×5) → **감쇠 최소제곱(DLS)** `Δq = Jᵀ(JJᵀ + λ²I)⁻¹ Δx`, λ≈0.01~0.05, 관절 한계 clip, 스텝당 Δx 제한.
- 산출물: `push/ee_control.py` (함수 `ik_step(model, data, target_xyz)`), `push/checks/test_ee_control.py`(목표점 추종 오차·수렴 시간 그래프, Subtask 4의 관절 실험과 같은 형식).
- 활용 ① 스크립트 Push 기준선(규칙 기반: push point로 이동 → 목표 방향으로 직선 이동) — RL과 성공률 비교. ② 선택 실험: 액션을 EE 증분(Δx, Δy, Δz)으로 바꾼 환경(`action_mode: "ee"`)에서 학습해 관절 액션과 비교.

**완료 기준**: 작업 영역 내 임의 목표점 추종 오차 < 5mm, 스크립트 기준선 성공률 측정.

### P8. Sim-to-Real 배포 (Subtask 12) — 실물 사용 가능 시
**목적**: 학습한 정책을 SO-101 실물에서 실행.

**사전 확인 (연구실)**
- 실물 SO-101 사용 일정, LeRobot 제어 경험자(카메라·CNN 작업 중인 학부연구생), 카메라·마커 장비.

**코드 구조 (신규 `deploy/`)** — 시뮬레이션과 **같은 관측 생성 코드**를 쓰는 것이 핵심
```
deploy/
├─ backends.py      # Backend 인터페이스: read_joints(), send_targets(q), read_cube_xy(), now()
│                   #   SimBackend(PushEnvV2 래핑) / RealBackend(LeRobot + 카메라)
├─ obs_builder.py   # push_env_v2._get_obs와 동일한 순서·단위로 관측 생성 (단위 테스트로 일치 검증)
├─ perception.py    # 카메라 → ArUco → 카메라→로봇 좌표 변환 → 큐브 (x, y)
├─ run_policy.py    # 20Hz 제어 루프: 관측 → VecNormalize → 정책 → 증분 → 안전 검사 → 명령
└─ safety.py        # 관절 한계, 스텝당 최대 변화량, 작업영역 이탈, 타임아웃, 성공 시 정지, 비상정지
```
**절차**
1. **관절 매핑 검증**: 시뮬 관절각 ↔ 실물(LeRobot 캘리브레이션) 각도의 영점·부호·단위(rad/deg) 일치 확인. 같은 관절각을 보내 자세를 사진으로 비교.
2. **카메라 캘리브레이션**: 체스보드로 내부 파라미터, 로봇 원점 마커로 카메라→로봇 변환. 정지 마커로 노이즈 σ·바이어스 측정.
3. **SimBackend로 run_policy 실행** → 시뮬 학습 환경과 같은 성공률이 나오는지 (배포 코드 자체 검증).
4. 실물 단계적 실행: ① 큐브 없이 정책 실행(동작 범위·속도 확인) → ② 큐브 고정 위치 → ③ 랜덤 위치.
5. 모든 실행은 관측·행동·시간·이미지 타임스탬프를 CSV로 기록.

**안전 규칙**: 첫 실행은 속도 상한을 낮게, 사람이 비상정지 대기, 성공 판정 시 즉시 정지, 작업영역 밖 목표 금지.

**완료 기준**: 학습 범위 안 위치 20회 시도 중 성공률 측정·기록 (목표치는 시뮬과의 차이 분석이 목적이므로 사전에 고정하지 않음).

### P9. Sim-to-Real Gap 분석 · 개선 (Subtask 13)
**측정 (시스템 식별)**
| 항목 | 방법 | 사용처 |
|---|---|---|
| 서보 응답 | Subtask 4의 step 응답(0→0.1rad)을 실물에서 동일 실행, 시뮬과 겹쳐 그림 | kp/kv 보정, kp DR 범위 |
| 제어 지연 | 명령 타임스탬프 vs 엔코더 변화 시작 시점 | 지연 DR 범위 |
| 카메라 오차 | 정지 마커 위치의 표준편차, 자로 잰 위치와의 차 | 관측 노이즈·바이어스 |
| 마찰 | 경사판 미끄럼 각도 또는 밀기 실험 | 마찰 DR 중심값 |
| 궤적 비교 | 같은 행동 시퀀스를 시뮬·실물에서 재생, 관절 궤적 RMSE | 종합 gap 지표 |

**분석**: 실패 유형 분류(도달 실패 / 방향 오류 / 과속 / 정지 실패 / 인식 실패 / 가림), 시뮬 민감도 표와 대조해 원인 추정.
**개선 루프**: 실측 중심값으로 시뮬 파라미터 보정 → 실측 오차만큼 DR → 재학습 → 실물 재평가.

### P10. 결과 정리 · 세미나 (Subtask 14)
**스토리라인**: 문제 정의 → 환경·MDP 설계 → 보상·성공 조건 개선(Phase 1) → 일반화(Phase 2, PPO vs SAC) → DR과 민감도 분석(Phase 3~P4) → 실물 대비 보완(P5~P6) → Sim-to-Real 결과·Gap(P8~P9) → 교훈.

**필요한 그림·표**
- 학습 곡선: 설정별 시드 평균±표준편차 (x축 스텝, 그리고 x축 시간 — PPO/SAC 비교용)
- Phase 1 시드별 성공 발견 시점, Phase 2 grid/ood 막대그래프
- 민감도 히트맵: 조건(행) × 정책(열) 성공률
- 커리큘럼 range_scale 곡선 (DR 학습 실패 설명)
- 재생 영상/GIF: 성공·대표 실패 유형
- (실물) 시뮬 vs 실물 step 응답, 궤적 비교, 실물 성공률

---

## 8. 작업 규칙 (사람·에이전트 공통)

### 코드
1. **기존 파일(v1: `push_env.py`, `train_push*.py`, `results/` 루트)은 수정하지 않는다.** 새 기능은 v2 파일에 옵션으로 추가.
2. 새 옵션의 **기본값은 기존 동작과 동일**해야 한다 (기존 설정·결과 재현성 유지).
3. 실험 차이는 코드 분기·파일 복사가 아니라 `configs_v2.py`의 **설정 이름**으로 만든다.
4. 환경을 바꾸면 `smoke_test_v2.py`에 검증 항목을 추가하고 통과시킨다.
5. 모든 실행은 프로젝트 루트에서 `python -m ...`.
6. 사용자의 로컬 파일을 덮어쓸 때는 **수정 시각을 확인**하고(사용자가 실습으로 고쳤을 수 있음), 커밋되지 않은 변경이 되돌려지지 않았는지 diff로 확인.

### 실험
7. **비교 실험은 scratch에서 시작** (이어 학습 금지), **시드 3개**, 학습량·`N_ENVS`·평가 세트 동일.
8. 효과를 주장하려면 **그 요소 하나만 다른 대조군**이 있어야 한다.
9. 학습 전 **코드를 commit** (config.json의 `git.dirty`가 false가 되도록).
10. 같은 이름 run이 있으면 학습이 거부된다. 다시 돌릴 때는 `--tag`로 구분하거나, 확인 후 `rm -rf results/v2/<run>` (반드시 `ls`로 목록 먼저 확인).
11. 성능 수치는 `evaluate_v2` 결과로만 보고 (학습 중 best 값은 선택 편향).

### 서버
12. 10분 이상 작업은 tmux. 공유 자원은 절반~2/3 이내, `nice`. GPU는 실행 직전 nvidia-smi 확인 후 번호 매핑에 맞춰 지정.
13. 서버에서 코드 수정 금지. 결과만 `git add -f`로 선택 커밋.

### 에이전트 추가 규칙
14. 사용자는 학부연구생이며 **코드를 직접 이해하고 쓸 수 있게 되는 것**이 목표다. 변경 시 무엇을 왜 바꿨는지 설명하고, 작은 기능은 실습 과제로 제안할 수 있다.
15. 결론을 낼 때 근거 수치와 한계(시드 수, 학습량 차이, 튜닝 여부)를 함께 적는다. 틀린 결론은 즉시 정정한다.
16. 실물 로봇 관련 계획은 실물 사용 가능 여부를 확인하기 전까지 시뮬레이션 준비 작업으로 한정한다.

---

## 9. 자주 겪은 문제와 해결

| 증상 | 원인 | 해결 |
|---|---|---|
| 같은 설정인데 시드별 성공/실패 | 탐색이 우연에 의존, 학습량 부족 | 학습량 증가, 커리큘럼, push point, 시드 3개로 판단 |
| 같은 시드인데 이전과 결과가 다름 | `N_ENVS`가 달라 난수 흐름 변화 | 비교 실험끼리 `N_ENVS` 고정 |
| 집게가 계속 회전 | 증분 액션 + 보상과 무관한 관절 | wrist_roll 고정 (P5) |
| 커리큘럼이 중간에 멈춤 | DR로 성공률 하락 → 80% 기준 미달 | DR 축소, 임계값 인자화 (P4) |
| `git push` 인증 실패 (`vscode-git-*.sock`) | 오래된 터미널/tmux의 VS Code 인증 통로 | 새 터미널, Reload Window, 근본적으로 SSH 키 |
| 노트북 닫으니 학습 종료 | tmux 없이 실행 | tmux 안에서 실행 |
| `git pull` 시 local changes 에러 | 서버에서 파일 수정됨 | `git stash push -m "..."` 후 pull |
| `rm: Is a directory` | 폴더 삭제에 `-f`만 사용 | `rm -rf` (목록 먼저 확인) |
| PPO가 GPU 메모리를 잡음 | CUDA 초기화 | 스크립트가 PPO 실행 시 `CUDA_VISIBLE_DEVICES=` 적용 |
| 정책 재생이 엉뚱함 | 모델과 정규화 통계 불일치 | `best_model` ↔ `best_vecnormalize` 쌍으로 로드 (`load_run` 사용) |
| `--random` 재생 시 큐브를 못 밂 | `--random`은 랜덤 **행동** 옵션 | 정책 재생은 `--run <폴더>`만, 고정 위치는 `--nominal` |
| SAC fps가 계속 떨어짐 | 로그 fps는 누적 평균 (초기 1만 스텝은 업데이트 없음) | 정상. 실제 속도는 두 시점 스텝 차로 계산 |

---

## 10. 명령어 모음

```bash
# --- 확인 ---
python -m push.checks.smoke_test_v2
python -m push.checks.check_workspace
python -m push.checks.status [이름일부]          # 진행률, fps, 남은시간, 커리큘럼, 평가 성공률

# --- 학습 (단일) ---
python -m push.train_v2 --config rand_pp --algo sac --seed 0 --steps 1500000
python -m push.train_v2 --config rand_pp --algo ppo --seed 0 --steps 3000000 --n-envs 6
#   옵션: --tag, --device, --eval-every, --eval-episodes, --overwrite

# --- 학습 (일괄, 서버 tmux 안) ---
MAX_JOBS=2 N_ENVS=6 nice -n 19 bash push/run_experiments.sh <phase>
CUDA_VISIBLE_DEVICES=3 MAX_JOBS=6 nice -n 19 bash push/run_experiments.sh phase3sac
#   phase: smoke | phase1 | phase1b | phase2 | phase3 | phase3sac | phase3diag

# --- 평가 ---
python -m push.evaluation.evaluate_v2 --runs "results/v2/rand_pp_sac_*" --out results/v2/eval_x.csv
python -m push.evaluation.evaluate_v2 --runs "results/v2/A_*" "results/v2/B_*" --suite all --splits grid --out results/v2/eval_y.csv

# --- 재생 (로컬) ---
python -m push.play_v2 --run results/v2/rand_pp_sac_s0 [--nominal] [--friction 2.0] [--kp-scale 0.7] [--delay 2] [--speed 0.3]
python -m push.play_v2 --random --config rand_pp     # 정책 없이 환경 확인
#   뷰어에서 C: 접촉점, F: 접촉힘

# --- 결과 올리기 (서버) ---
git add -f results/v2/<pattern>/config.json results/v2/<pattern>/best_* results/v2/eval_*.csv
git commit -m "..." && git pull && git push
```

**summary CSV를 조건×설정 표로 보기**
```bash
python - <<'EOF'
import csv
f = "results/v2/eval_phase3_sac_summary.csv"
rows = [r for r in csv.DictReader(open(f, encoding="utf-8-sig")) if r["split"] == "grid"]
conds = list(dict.fromkeys(r["condition"] for r in rows)); cfgs = list(dict.fromkeys(r["config"] for r in rows))
tab = {(r["config"], r["condition"]): (float(r["success_mean"]), float(r["success_std"])) for r in rows}
print(f"{'조건':<16}" + "".join(f"{c:>24}" for c in cfgs))
for c in conds:
    print(f"{c:<16}" + "".join(f"{tab[(g,c)][0]:>18.2f}±{tab[(g,c)][1]:.2f}" if (g,c) in tab else f"{'-':>24}" for g in cfgs))
EOF
```

---

## 11. 용어 정리와 참고 자료

| 용어 | 뜻 |
|---|---|
| MDP / POMDP | 상태·행동·보상·전이로 정의한 문제 / 상태 일부만 관측되는 경우 (지연, 마찰값 등) |
| terminated / truncated | 작업이 끝남(성공·낙하) / 시간 제한으로 끊음 |
| on-policy / off-policy | 현재 정책 데이터만 사용(PPO) / 과거 데이터 재사용(SAC) |
| reward shaping / hacking | 학습을 돕는 중간 보상 / 의도와 다른 방식으로 보상만 올리는 것 |
| curriculum | 쉬운 분포에서 어려운 분포로 점진 확대 |
| in-distribution / OOD | 학습 때 본 범위 안 / 밖 |
| DR / System ID | 물리·센서 값을 랜덤화 / 실물을 측정해 시뮬레이터를 맞춤 |
| privileged info / teacher-student | 시뮬에서만 얻는 정보 / 특권 정보 정책을 실물 관측 정책이 모방 |
| ablation | 요소를 하나씩 넣고 빼서 효과 확인 |
| selection bias | 여러 번 중 최고를 고르면 실제보다 좋게 보임 |

**참고 자료**
- 강화학습 입문: 『바닥부터 배우는 강화 학습』(노승은, 유튜브 팡요랩), Hugging Face Deep RL Course, OpenAI Spinning Up(Key Concepts, PPO, SAC)
- 이론: 『단단한 강화학습』(Sutton & Barto 번역) 3·6·13장
- 로봇 기초: Modern Robotics (Coursera 한국어 강좌, 1~3장: 기구학·Jacobian·좌표 변환)
- MuJoCo 공식 문서 (Overview, Computation, camera), Stable-Baselines3 문서 (RL Tips and Tricks)
- Sim-to-Real: Lilian Weng "Domain Randomization for Sim2Real Transfer", Tobin et al. 2017, **Peng et al. 2018 (로봇팔 Push + 동역학 DR, 가장 유사한 선행연구)**, OpenAI 2019 (ADR)
- 카메라: OpenCV Camera Calibration · ArUco 튜토리얼, 『OpenCV 4로 배우는 컴퓨터 비전과 머신 러닝』(황선규)
- 비전 RL: RAD (Laskin 2020), DrQ (Kostrikov 2020)
- 실험 방법론: Henderson et al. 2018, Agarwal et al. 2021
