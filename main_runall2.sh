#!/bin/bash
# ============================================
# Multi-Model Runner (Final Version)
# ============================================

DATASET="seq-tinyimagenet"
NETWORK="resnet18"
BATCH_SIZE=64
LR=0.005
NUM_CLIENTS=5
NUM_COMM_ROUNDS=10
NUM_EPOCHS=5
WANDB=False
# DISTRIBUTION_ALPHA=10.0
# DISTRIBUTION_ALPHA=100000
DISTRIBUTION_ALPHA=0.001

# -------------------------------
#  SEED
# -------------------------------
SEED=44

# -------------------------------
#  GPU
# -------------------------------
DID_LIST=(0)
MAX_JOBS=${#DID_LIST[@]}
job_idx=0

# -------------------------------
#  MODELS LIST
# -------------------------------
MODELS=(fedavg ewc fedproto lwf )

# -------------------------------
#  OUTPUT
# -------------------------------
# OUTPUT_DIR="./results2"
OUTPUT_DIR="./${DATASET}_Seed${SEED}_Alpha${DISTRIBUTION_ALPHA}_hetero"
mkdir -p "$OUTPUT_DIR"

ALL_LOG="$OUTPUT_DIR/all_results.log"
RAW_RESULTS="$OUTPUT_DIR/raw_results.tmp"

echo "All results log initialized." > "$ALL_LOG"
> "$RAW_RESULTS"

# -------------------------------
#  RUN MODELS
# -------------------------------
for MODEL in "${MODELS[@]}"; do

  DID=${DID_LIST[$job_idx]}
  RUN_DIR="${OUTPUT_DIR}/${MODEL}"
  LOG_FILE="${RUN_DIR}/log.txt"

  echo "============================================="
  echo "Launching: $MODEL on GPU $DID"
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
      --random_seed $SEED \
      > "$LOG_FILE" 2>&1

  ) &

  # Scheduler (1 GPU → sequential)
  job_idx=$((job_idx + 1))
  if (( job_idx >= MAX_JOBS )); then
    wait
    job_idx=0
  fi

done

# Wait remaining jobs
wait

# -------------------------------
#  Collect Results
# -------------------------------
for MODEL in "${MODELS[@]}"; do
  LOG_FILE="${OUTPUT_DIR}/${MODEL}/log.txt"

  if [[ -f "$LOG_FILE" ]]; then

    #  Extract FINAL Global Mean Accuracy
    SCORE=$(grep "\[Global\] Mean accuracy up to task" "$LOG_FILE" \
      | grep -oE '[0-9]+(\.[0-9]+)?(?= ?%)' \
      | tail -n 1)

    if [[ -n "$SCORE" ]]; then
      echo "$MODEL  :  $SCORE" >> "$ALL_LOG"
      echo "$SCORE $MODEL" >> "$RAW_RESULTS"
    else
      echo "$MODEL  :  FAILED (no score)" >> "$ALL_LOG"
    fi
  fi
done

# -------------------------------
#  Best Model
# -------------------------------
BEST_SCORE=-999
BEST_MODEL=""

while read score name; do
  cmp=$(awk -v a="$score" -v b="$BEST_SCORE" 'BEGIN {print (a>b)?1:0}')
  if [ "$cmp" -eq 1 ]; then
    BEST_SCORE=$score
    BEST_MODEL=$name
  fi
done < <(sort -nr "$RAW_RESULTS")

# -------------------------------
#  Leaderboard + Best
# -------------------------------
{
  echo ""
  echo "============================================="
  echo "🏆 BEST MODEL"
  echo "============================================="
  echo "Model : $BEST_MODEL"
  echo "Score : $BEST_SCORE"
  echo "Seed  : $SEED"
  echo "============================================="

  echo ""
  echo "===== SORTED LEADERBOARD ====="
  sort -nr "$RAW_RESULTS" | while read score name; do
    echo "$score  :  $name"
  done

} >> "$ALL_LOG"

rm -f "$RAW_RESULTS"

echo "✅ Finished running ${#MODELS[@]} models"
echo "📄 Logs : $ALL_LOG"