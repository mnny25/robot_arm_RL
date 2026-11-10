#!/usr/bin/env bash
# Push v2 실험 일괄 실행 (GPU 서버 등 리눅스용)
#
# 사용법 (프로젝트 루트에서):
#   bash push/run_experiments.sh smoke     # 1분짜리 동작 확인
#   bash push/run_experiments.sh phase1    # hold 효과: fixed_nohold vs fixed_hold (PPO, 시드 3개)
#   bash push/run_experiments.sh phase2    # 랜덤화: rand PPO vs SAC (시드 3개)
#   bash push/run_experiments.sh phase3    # DR: rand / rand_dr_friction / rand_dr_full (PPO, 시드 3개)
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
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do sleep 10; done
  echo "[$(date '+%m-%d %H:%M')] 시작: $name ($steps steps)"
  python -m push.train_v2 --config "$cfg" --algo "$algo" --seed "$seed" --steps "$steps" "$@" \
    > "logs/${name}.log" 2>&1 &
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
  phase2)
    STEPS=${STEPS:-2000000}
    for s in $SEEDS; do
      run rand ppo "$s" "$STEPS" --n-envs "$N_ENVS"
      run rand sac "$s" "$STEPS"
    done
    ;;
  phase3)
    STEPS=${STEPS:-2000000}
    for s in $SEEDS; do
      run rand_dr_friction ppo "$s" "$STEPS" --n-envs "$N_ENVS"
      run rand_dr_full     ppo "$s" "$STEPS" --n-envs "$N_ENVS"
    done
    ;;
  *)
    echo "알 수 없는 단계: $PHASE (smoke | phase1 | phase2 | phase3)"; exit 1 ;;
esac
wait
echo "[$(date '+%m-%d %H:%M')] $PHASE 완료"
