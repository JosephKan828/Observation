#!/bin/bash

BASE_PATH="/work/DATA/Satellite/CloudSat"
LOG_FILE="collect_log.csv"

[ -f "$LOG_FILE" ] && rm "$LOG_FILE"

TIME_FORMAT="\n--- Statistics ---\nExecution Time: %E\nMax RAM Usage: %M KB\n------------------"

# 1. Initialize Log File if it doesn't exist (CSV Header)
if [ ! -f "$LOG_FILE" ]; then
    printf "%-10s %-10s %-15s %-15s\n" "YEAR" "DAY" "EXEC_TIME" "MAX_RAM_KB" > "$LOG_FILE"
    echo "------------------------------------------------------------" >> "$LOG_FILE"
fi

# 2. Main Loop
for YEAR_DIR in $BASE_PATH/*; do
    [ -d "$YEAR_DIR" ] || continue
    YEAR=$(basename "$YEAR_DIR")
    
    for DAY_DIR in $YEAR_DIR/*; do
        [ -d "$DAY_DIR" ] || continue
        DAY=$(basename "$DAY_DIR")
        
        echo "Processing $YEAR - $DAY"

        /usr/bin/time -a -o "$LOG_FILE" \
        -f "$YEAR       $DAY       %E           %M" \
        python -u CloudSat_collect.py "$YEAR" "$DAY"
    done
done