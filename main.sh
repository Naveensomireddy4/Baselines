#!/bin/bash
# ============================================
# Parallel + Deterministic Hyperparameter Search
# EWC: ewc_lambda x ewc_gamma grid search
# ============================================
MODEL="ewc"
DATASET="seq-cifar100"
NETWORK="resnet18"
BATCH_SIZE=64
LR=0.005
NUM_CLIENTS=5
NUM_COMM_ROUNDS=20
NUM_EPOCHS=5
WANDB=False
DISTRIBUTION_ALPHA=10.0

# -------------------------------
# 🔑 SEED
# -------------------------------
SEED=42

# -------------------------------
# 🚀 AVAILABLE GPUs
# -------------------------------
DID_LIST=(0)
MAX_JOBS=${#DID_LIST[@]}
job_idx=0

OUTPUT_DIR="./results"
mkdir -p "$OUTPUT_DIR"
BEST_LOG="$OUTPUT_DIR/best_so_far.log"
ALL_LOG="$OUTPUT_DIR/all_results.log"
RAW_RESULTS="$OUTPUT_DIR/raw_unsorted_results.tmp"

echo "All results log initialized." > "$ALL_LOG"
echo "Best-so-far log initialized." > "$BEST_LOG"
> "$RAW_RESULTS"

# -------------------------------
# 🔥 HYPERPARAMETERS
# Based on CIFAR-100 10-task recommendations
# -------------------------------
ewc_lambda_values=( 1 10 25 50 100 )
ewc_gamma_values=(0.5 0.75 0.9 1.0)

BEST_SCORE=-999
BEST_COMBO=""

# -------------------------------
# 🔁 Hyperparameter Sweep (Grid)
# -------------------------------
for LAMBDA in "${ewc_lambda_values[@]}"; do
  for GAMMA in "${ewc_gamma_values[@]}"; do

    DID=${DID_LIST[$job_idx]}
    RUN_NAME="seed${SEED}_lambda${LAMBDA}_gamma${GAMMA}_gpu${DID}"
    RUN_DIR="${OUTPUT_DIR}/${RUN_NAME}"
    LOG_FILE="${RUN_DIR}/log.txt"

    echo "============================================="
    echo "Launching: $RUN_NAME on GPU $DID"
    echo "  ewc_lambda = $LAMBDA"
    echo "  ewc_gamma  = $GAMMA"
    echo "============================================="

    mkdir -p "$RUN_DIR"

    (
      export PYTHONHASHSEED=$SEED
      export CUBLAS_WORKSPACE_CONFIG=:4096:8
      CUDA_VISIBLE_DEVICES=$DID python3 -u main.py \
        --model $MODEL \
        --dataset $DATASET \
        --network $NETWORK \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --num_clients $NUM_CLIENTS \
        --num_comm_rounds $NUM_COMM_ROUNDS \
        --num_epochs $NUM_EPOCHS \
        --wandb $WANDB \
        --distribution_alpha $DISTRIBUTION_ALPHA \
        --ewc_lambda $LAMBDA \
        --ewc_gamma $GAMMA \
        > "$LOG_FILE" 2>&1
    ) &

    # ---------------------------
    # Scheduler (max jobs = num GPUs)
    # ---------------------------
    job_idx=$((job_idx + 1))
    if (( job_idx >= MAX_JOBS )); then
      wait
      job_idx=0
    fi

  done
done

# Wait for remaining jobs
wait

# -------------------------------
# 📊 Collect Results
# -------------------------------
for log in ${OUTPUT_DIR}/*/log.txt; do
  RUN_NAME=$(basename "$(dirname "$log")")
  SCORE=$(grep -E "Average Accuracy" "$log" | grep -oE '[0-9]+(\.[0-9]+)?' | tail -n 1)
  if [[ -n "$SCORE" ]]; then
    echo "$RUN_NAME  :  $SCORE" >> "$ALL_LOG"
    echo "$SCORE $RUN_NAME" >> "$RAW_RESULTS"
  fi
done

# -------------------------------
# 🏆 Best Result
# -------------------------------
while read score name; do
  cmp=$(awk -v a="$score" -v b="$BEST_SCORE" 'BEGIN {print (a>b)?1:0}')
  if [ "$cmp" -eq 1 ]; then
    BEST_SCORE=$score
    BEST_COMBO=$name
  fi
done < <(sort -nr "$RAW_RESULTS")

{
  echo "============================================="
  echo " FINAL BEST EWC COMBINATION "
  echo "============================================="
  echo "🏆 Best Run  : $BEST_COMBO"
  echo "📊 Best Score: $BEST_SCORE"
  echo "🌱 Seed      : $SEED"
  echo "============================================="
} > "$BEST_LOG"

# Print sorted leaderboard to ALL_LOG
echo "" >> "$ALL_LOG"
echo "===== SORTED LEADERBOARD =====" >> "$ALL_LOG"
sort -nr "$RAW_RESULTS" | while read score name; do
  echo "$score  :  $name" >> "$ALL_LOG"
done

rm -f "$RAW_RESULTS"

echo "✅ Parallel sweep finished (${#ewc_lambda_values[@]} x ${#ewc_gamma_values[@]} = $((${#ewc_lambda_values[@]} * ${#ewc_gamma_values[@]})) runs)"
echo "📄 Logs : $ALL_LOG"
echo "🏆 Best : $BEST_LOG"
