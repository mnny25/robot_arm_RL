# SO-101 강화학습 실험

Reach, Push, Grasp 환경과 학습·재생·평가 코드를 Task별로 구분합니다.

## 폴더 구조

```text
robot_arm_rl/
├─ project_paths.py       # 공통 프로젝트 루트 경로
├─ reach/                 # EE 목표 위치 도달 환경·PPO 학습·재생
├─ push/                  # Cube Push 환경·PPO/SAC/DR 학습·재생
│  └─ evaluation/         # 기본 성능·질량·마찰 비교
├─ grasp/                 # 잡기·들어 올리기 환경·PPO 학습·재생
│  └─ checks/             # 그리퍼·잡기 자세·들어 올리기 확인
├─ checks/                # GPU 실험 정책 재생 등 보조 확인
├─ modelingfile/          # 기존 XML, 메시 및 기타 모델 자산
└─ results/               # 기존 정책, 학습 로그, 평가 CSV, 이미지
```

## 실행 방법

가상환경을 활성화하고 **robot_arm_rl 폴더에서** 실행합니다.
파일 경로로 실행하는 대신 `python -m`을 사용합니다.

```powershell
cd E:\Lab\robot_arm_rl

# Reach: 랜덤 동작 / 학습 / 학습한 정책 재생
python -m reach.reach_env
python -m reach.train_ppo
python -m reach.play_ppo

# Push: 랜덤 동작 / PPO 추가 학습 / SAC 학습 / 재생
python -m push.push_env
python -m push.train_push
python -m push.train_push_sac
python -m push.play_push
python -m push.play_push_sac

# 마찰 DR / progress 보상이 있는 환경에서 추가 학습
python -m push.train_push_dr
python -m push.train_push_progress

# 높은 마찰 조건에서 기존 정책과 DR 정책을 각각 재생
python -m push.play_high_friction --policy original
python -m push.play_high_friction --policy dr

# 정량 평가: 실행하면 기존 CSV를 같은 경로에 다시 씁니다.
python -m push.evaluation.evaluate_push
python -m push.evaluation.evaluate_mass
python -m push.evaluation.evaluate_friction
python -m push.evaluation.evaluate_friction_dr
python -m push.evaluation.evaluate_friction_progress

# Grasp: 학습 / 재생 / 수동 동작 확인
python -m grasp.train_grasp
python -m grasp.play_grasp
python -m grasp.checks.test_gripper
python -m grasp.checks.test_grasp_pose
python -m grasp.checks.test_grasp_lift

# GPU 서버에서 학습한 SAC 정책을 로컬에서 재생
python -m checks.play_sac_gpu_test
```

서버에서도 프로젝트 루트로 이동한 뒤 동일한 `python -m` 명령을 사용합니다.
모델과 결과 경로는 `project_paths.PROJECT_ROOT`를 기준으로 계산합니다.
뷰어를 사용하는 `play_*`, 환경 직접 실행, 수동 확인 코드는 화면이 있는 환경에서 실행합니다.

## 실험 폴더와 현재 스크립트의 연결

아래는 현재 코드가 지정한 경로입니다. 과거 체크포인트 생성 당시의 보상·성공 기준까지 증명하는 기록은 아닙니다.

| 결과 폴더 | 현재 코드에서의 용도 |
|---|---|
| `ppo_reach` | 초기 Reach 실험 보관 |
| `ppo_reach_random` | Reach 학습 출력 및 재생 입력 |
| `ppo_push_001` | `push.train_push`의 추가 학습 시작 정책 |
| `ppo_push_002` | 이전 Push 실험 보관 |
| `ppo_push_003` | PPO Push 추가 학습 출력, 기본 비교 및 DR 학습 입력 |
| `ppo_push_dr_001` | 마찰 무작위화 추가 학습 출력, progress 실험 입력 |
| `ppo_push_progress_001` | DR 정책에서 시작하는 추가 학습 출력 |
| `sac_push_001` | 이전 SAC Push 실험 보관 |
| `sac_push_002` | SAC 학습 출력 및 비교 대상 정책 |
| `ppo_grasp_001` | Grasp 학습 출력 및 재생 입력 |

SAC 학습 코드는 `sac_push.zip`으로 저장하지만 일부 기존 재생·평가 코드는
`sac_push_002/ppo_push.zip`을 읽습니다. 이번 정리에서는 기존 선택을 유지했습니다.
평가할 때 실제로 사용할 체크포인트 이름을 확인하세요.

## 파일 정리 범위와 실험 기록

- 기존 Python 파일 23개를 이동하고 import 및 공통 루트 경로만 조정했습니다.
- 모델 XML·메시, 저장된 정책, replay buffer, 학습 로그, 평가 CSV는 이동하거나 수정하지 않았습니다.
- 환경의 Action/Observation/Reward, 성공 조건, 학습량, 정책 선택은 변경하지 않았습니다.
- 기존 미커밋 Grasp 코드와 수동 확인 코드의 내용도 함께 보존했습니다.

`push_env.py`는 여러 실험이 공유합니다. 현재 들어 있는 progress 보상은 그 환경을 쓰는
모든 학습에 적용되며, `train_push_progress.py`에만 한정되지 않습니다.
또한 현재 성공 조건은 0.5초 유지 판정 뒤에 거리·높이 판정으로 다시 할당됩니다.
이번 파일 정리에서는 이 동작을 수정하지 않았습니다.

다음 실험부터 아래 항목을 결과 폴더에 별도 기록하면 과거 결과를 해석하기 쉽습니다.

```text
실험 이름 / 코드 커밋:
시작 체크포인트:
추가 학습 스텝 / seed:
초기 큐브·목표 위치 범위:
질량·마찰 및 무작위화 범위:
보상과 성공·실패 조건:
평가 조건 / 결과:
```

학습과 평가는 같은 실험 이름으로 실행하면 기존 정책·로그·CSV를 덮어쓸 수 있으므로,
새 실험을 보관할 때는 출력 폴더 이름도 구분합니다.
