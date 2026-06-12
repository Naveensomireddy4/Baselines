#!/bin/bash

RESULTS_DIR="./results_tuning_ewc_lambda_gamma"
ALL_LOG="$RESULTS_DIR/all_results.log"
BEST_LOG="$RESULTS_DIR/best_so_far.log"
RAW_TMP="$RESULTS_DIR/raw_unsorted_results.tmp"

echo "Rebuilding results..."
> "$ALL_LOG"
> "$BEST_LOG"
> "$RAW_TMP"

BEST_SCORE=-999
BEST_COMBO=""

# ----------------------------------------------------
# Scan through all run folders and extract accuracies
# ----------------------------------------------------
for RUN_DIR in "$RESULTS_DIR"/seed*; do
    RUN_NAME=$(basename "$RUN_DIR")
    LOG_FILE="$RUN_DIR/log.txt"

    if [ ! -f "$LOG_FILE" ]; then
        echo "Skipping $RUN_NAME (no log.txt)"
        continue
    fi

    # Extract numeric accuracy (float)
    SCORE=$(grep "\[GLOBAL\]" "$LOG_FILE" | grep -oE '[0-9]+(\.[0-9]+)?' | tail -n 1)

    if [[ -z "$SCORE" ]]; then
        echo "⚠️ No accuracy found in $RUN_NAME"
        continue
    fi

    echo "$RUN_NAME  :  $SCORE" >> "$ALL_LOG"
    echo "$SCORE $RUN_NAME" >> "$RAW_TMP"

    # Compare using awk
    cmp=$(awk -v a="$SCORE" -v b="$BEST_SCORE" 'BEGIN {print (a>b)?1:0}')
    if [ "$cmp" -eq 1 ]; then
        BEST_SCORE="$SCORE"
        BEST_COMBO="$RUN_NAME"
    fi
done

# ----------------------------------------------------
# Sort results (best first)
# ----------------------------------------------------
echo "" >> "$ALL_LOG"
echo "=============================================" >> "$ALL_LOG"
echo "       SORTED RESULTS (BEST FIRST)           " >> "$ALL_LOG"
echo "=============================================" >> "$ALL_LOG"

sort -nr "$RAW_TMP" | while read score name; do
    printf "%s  :  %s\n" "$name" "$score" >> "$ALL_LOG"
done

# Remove temp file
rm -f "$RAW_TMP"

# ----------------------------------------------------
# Write BEST to file
# ----------------------------------------------------
{
echo "============================================="
echo " FINAL BEST SNR COMBINATION "
echo "============================================="
echo "🏆 Best Run:  $BEST_COMBO"
echo "📊 Best Score: $BEST_SCORE"
echo "🌱 Seed: 42"
echo "============================================="
} | tee "$BEST_LOG"

echo ""
echo "✅ Post-processing complete!"
echo "📄 All results → $ALL_LOG"
echo "🏆 Best result → $BEST_LOG"
