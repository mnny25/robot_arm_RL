#!/usr/bin/env bash
# Push v2 실험 일괄 실행 (GPU 서버 등 리눅스용)
#
# 사용법 (프로젝트 루트에서):
#   bash push/run_experiments.sh smoke     # 1분짜리 동작 확인
#   bash push/run_experiments.sh phase1    # hold 효과: fixed_nohold vs fixed_hold (PPO, 시드 3개)
#   bash push/run_experiments.sh phase1b   # 학습량 150만: fixed_hold vs fixed_hold_pp (PPO, 시드 3개)
#   bash push/run_experiments.sh phase2    # 랜덤화: rand_pp PPO(300만) vs SAC(150만), 시드 3개
#   bash push/run_experiments.sh phase3    # DR: rand_pp_dr_friction / _dr_full / _dr_full_real (PPO 300만, 시드 3개)
#   bash push/run_experiments.sh phase3sac # DR을 SAC로: rand_pp_dr_full / _dr_full_real (SAC 150만, 시드 3개)
#
# 환경변수로 조절:
#   SEEDS="0 1 2"  STEPS=1000000  N_ENVS=8  MAX_JOBS=3
#   MAX_JOBS = 동시에 돌릴 학습 수. PPO 1개가 코어 N_ENVS개를 쓰므로
#              MAX_JOBS * N_ENVS 가 서버 코어 수(nproc)를 넘지 않게 하세요.
#
# 로그: logs/<run>.log  /  결과: results/v2/<run>/
# 실행 상황:  tail -f logs/*.log   또는   tensorboard --logdir results/v2

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

PHASE=${1:-smoke}
SEEDS=${SEEDS:-"0 1 2"}
N_ENVS=${N_ENVS:-8}
MAX_JOBS=${MAX_JOBS:-3}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

run() {  # run <config> <algo> <seed> <steps> [추가 인자]
  local cfg=$1 algo=$2 seed=$3 steps=$4; shift 4
  local name="${cfg}_${algo}_s${seed}"
  local args=("$@") i
  for ((i = 0; i < ${#args[@]}; i++)); do   # --tag가 있으면 로그 이름에도 붙임
    if [ "${args[i]}" = "--tag" ]; then name="${name}_${args[i+1]}"; fi
  done
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do sleep 10; done
  echo "[$(date '+%m-%d %H:%M')] 시작: $name ($steps steps)"
  if [ "$algo" = "ppo" ]; then
    # PPO는 CPU로 학습 → GPU를 아예 안 보이게 해서 GPU 메모리를 잡지 않도록 함
    CUDA_VISIBLE_DEVICES= python -m push.train_v2 --config "$cfg" --algo "$algo" --seed "$seed" \
      --steps "$steps" "$@" > "logs/${name}.log" 2>&1 &
  else
    python -m push.train_v2 --config "$cfg" --algo "$algo" --seed "$seed" --steps "$steps" "$@" \
      > "logs/${name}.log" 2>&1 &
  fi
}

case "$PHASE" in
  smoke)
    python -m push.checks.smoke_test_v2
    python -m push.train_v2 --config rand --algo ppo --seed 0 --steps 20000 --n-envs 2 \
      --eval-every 10000 --eval-episodes 5 --tag smoke --overwrite
    python -m push.train_v2 --config rand --algo sac --seed 0 --steps 12000 \
      --eval-every 6000 --eval-episodes 5 --tag smoke --overwrite
    python -m push.evaluation.evaluate_v2 --runs "results/v2/*_smoke" --out results/v2/eval_smoke.csv
    ;;
  phase1)
    STEPS=${STEPS:-500000}
    for s in $SEEDS; do
      run fixed_nohold ppo "$s" "$STEPS" --n-envs "$N_ENVS"
      run fixed_hold   ppo "$s" "$STEPS" --n-envs "$N_ENVS"
    done
    ;;
  phase1b)
    # 학습량을 늘려서 hold vs hold+push point 비교 (결과 폴더 이름 끝에 _long)
    STEPS=${STEPS:-1500000}
    for s in $SEEDS; do
      run fixed_hold    ppo "$s" "$STEPS" --n-envs "$N_ENVS" --tag long
      run fixed_hold_pp ppo "$s" "$STEPS" --n-envs "$N_ENVS" --tag long
    done
    ;;
  phase2)
    # 위치 랜덤화 + push point: PPO vs SAC
    # SAC는 스텝당 시간이 길어서 먼저 띄우고 스텝 수를 줄임 (코어 1개씩만 사용)
    PPO_STEPS=${PPO_STEPS:-3000000}
    SAC_STEPS=${SAC_STEPS:-1500000}
    for s in $SEEDS; do run rand_pp sac "$s" "$SAC_STEPS"; done
    for s in $SEEDS; do run rand_pp ppo "$s" "$PPO_STEPS" --n-envs "$N_ENVS"; done
    ;;
  phase3)
    # DR: phase2의 rand_pp PPO가 비교 기준
    STEPS=${STEPS:-3000000}
    for s in $SEEDS; do
      run rand_pp_dr_friction  ppo "$s" "$STEPS" --n-envs "$N_ENVS"
      run rand_pp_dr_full      ppo "$s" "$STEPS" --n-envs "$N_ENVS"
      run rand_pp_dr_full_real ppo "$s" "$STEPS" --n-envs "$N_ENVS"
    done
    ;;
  phase3sac)
    # DR을 SAC로: phase2의 rand_pp SAC가 비교 기준 (GPU 사용 → CUDA_VISIBLE_DEVICES 지정)
    STEPS=${STEPS:-1500000}
    for s in $SEEDS; do
      run rand_pp_dr_full      sac "$s" "$STEPS"
      run rand_pp_dr_full_real sac "$s" "$STEPS"
    done
    ;;
  *)
    echo "알 수 없는 단계: $PHASE (smoke | phase1 | phase1b | phase2 | phase3 | phase3sac)"; exit 1 ;;
esac
wait
echo "[$(date '+%m-%d %H:%M')] $PHASE 완료"
