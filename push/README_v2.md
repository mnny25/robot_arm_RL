# Push v2 실험 가이드

기존 `push_env.py`와 기존 학습·평가 스크립트, `results/`의 기존 결과는 그대로 두었습니다.
v2 파일은 모두 새로 추가한 것이고, 결과는 `results/v2/` 아래에만 저장됩니다.

## 추가된 파일

| 파일 | 역할 |
|---|---|
| `push/push_env_v2.py` | 설정 기반 Push 환경 (hold 성공 판정, 큐브·목표 랜덤화, DR, 관측 모드) |
| `push/configs_v2.py` | 실험 설정 이름 모음 (`fixed_nohold`, `fixed_hold`, `rand`, `rand_dr_full` …) |
| `push/train_v2.py` | PPO/SAC 학습 (병렬 환경, 관측 정규화, 평가·최고 모델 저장, 커리큘럼, 실험 기록) |
| `push/evaluation/evaluate_v2.py` | 고정 평가 세트 + 물리 조건 스윕, 시드별 평균±표준편차 |
| `push/play_v2.py` | 뷰어 재생 (로컬 PC) |
| `push/checks/check_workspace.py` | 로봇이 밀기 높이에서 닿는 xy 영역 확인 |
| `push/checks/smoke_test_v2.py` | 환경 동작 자동 확인 |
| `push/run_experiments.sh` | 서버에서 단계별 실험 일괄 실행 |

## 기존 환경과 달라진 점

1. **성공 판정**: 목표 원(반지름 3cm) 안에서 큐브가 거의 멈춘 상태(선속도 < 3cm/s, 각속도 < 0.5rad/s)를
   0.5초(10스텝) 유지해야 성공합니다. `require_stable=False`면 기존처럼 들어오는 순간 성공합니다.
2. **보상**: 기존 항목(EE 거리, 목표 거리, progress, 행동 크기)에 두 항목을 추가했습니다.
   - 큐브 속도가 0.15m/s를 넘으면 페널티 → 쳐서 날리기 방지
   - 목표 근처(6cm 이내)에서는 큐브 속도 자체를 페널티 → 멈추도록 유도
   - 원 안에 있을 때 +보상은 일부러 넣지 않았습니다. 성공 전 스텝 보상이 양수가 되면
     정책이 성공(종료)하지 않고 원 근처에 머물며 보상을 계속 모으는 보상 해킹이 생길 수 있습니다.
3. **태스크 랜덤화**: `range_scale` 0이면 기존 고정 위치, 1이면 큐브 x 0.18~0.26 / y −0.06~0.06,
   목표는 큐브에서 6~12cm, 방향 ±60°(로봇 바깥쪽 기준). 뷰어의 초록 타깃도 같이 이동합니다.
   커리큘럼을 켜면 학습 중 최근 100 에피소드 성공률이 80%를 넘을 때마다 0.1씩 넓힙니다.
4. **DR**: 마찰, 큐브 질량, 액추에이터 kp, 관절 damping, 큐브 위치 관측 노이즈·바이어스, 제어 지연.
   매 reset마다 원래 값에서 다시 적용하므로 이전 에피소드 값이 남지 않습니다.
5. **관측 모드**: `full`(기존 31차원) / `real`(큐브 자세·속도 제외 21차원, 실물에서 얻기 쉬운 값만)

## 순서

### 0. 로컬에서 확인 (처음 한 번)
```powershell
cd E:\Lab\robot_arm_rl
python -m push.checks.smoke_test_v2        # 모두 [OK]인지
python -m push.checks.check_workspace      # 'x' 칸이 보이면 configs/env의 workspace를 줄이기
python -m push.play_v2 --random --config rand   # 큐브·목표가 매번 바뀌는지 눈으로 확인
```

### 1. 서버에서 학습
```bash
git pull                                   # 로컬에서 커밋·푸시한 코드 받기
bash push/run_experiments.sh smoke         # 몇 분: 전체 파이프라인 확인
bash push/run_experiments.sh phase1        # hold 효과 (고정 위치, PPO, 시드 3개)
bash push/run_experiments.sh phase2        # 랜덤화: PPO vs SAC (시드 3개)
bash push/run_experiments.sh phase3        # DR (phase2의 rand PPO가 비교 기준)
```
- 동시 실행 수는 `MAX_JOBS`, 환경 수는 `N_ENVS`로 조절 (`MAX_JOBS × N_ENVS ≤ nproc`)
- 접속이 끊겨도 계속 돌도록 `tmux` 안에서 실행하세요.
- 진행 확인: `tail -f logs/*.log`, `tensorboard --logdir results/v2`

### 2. 평가
```bash
python -m push.evaluation.evaluate_v2 --runs "results/v2/fixed_*" --suite nominal --out results/v2/eval_phase1.csv
python -m push.evaluation.evaluate_v2 --runs "results/v2/rand_ppo_*" "results/v2/rand_sac_*" --out results/v2/eval_phase2.csv
python -m push.evaluation.evaluate_v2 --runs "results/v2/rand_*ppo*" --suite all --out results/v2/eval_phase3.csv
```
`*_summary.csv`에 설정·알고리즘·조건·split별 성공률 평균±표준편차, 최대 큐브 속도, 최종 거리가 저장됩니다.

### 3. 로컬에서 재생
서버의 `results/v2/<run>/` 폴더를 통째로 로컬 같은 위치에 복사한 뒤:
```powershell
python -m push.play_v2 --run results/v2/rand_ppo_s0
python -m push.play_v2 --run results/v2/rand_dr_full_ppo_s0 --friction 1.8 --kp-scale 0.7
```

## 각 단계에서 볼 것

| 단계 | 비교 | 확인할 지표 |
|---|---|---|
| Phase 1 | fixed_nohold vs fixed_hold | 성공률, **최대 큐브 속도**(기존 0.78m/s), 최종 거리 |
| Phase 2 | rand PPO vs rand SAC | 학습 곡선(같은 스텝), grid/ood 성공률 평균±표준편차 |
| Phase 3 | rand vs rand_dr_friction vs rand_dr_full | 기본 조건 성공률(약간 떨어질 수 있음) vs 극단 조건 성공률(덜 무너져야 함) |

## 실험 기록
`results/v2/<run>/config.json`에 설정 전체, git 커밋, 패키지 버전, 실행 명령이 자동 저장됩니다.
코드를 고친 뒤에는 **커밋하고 나서** 학습을 돌려야 어떤 코드로 학습했는지 추적할 수 있습니다
(`git.dirty: true`면 커밋되지 않은 변경이 있었다는 뜻).
