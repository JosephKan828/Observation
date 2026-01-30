#!/bin/bash

BASE_PATH="/data92/b11209013/CloudSat_tmp"
LOG_FILE="regrid_log.csv"

[ -f "$LOG_FILE" ] && rm "$LOG_FILE"

TIME_FORMAT="\n--- Statistics ---\nExecution Time: %E\nMax RAM Usage: %M KB\n------------------"

# 1. Initialize Log File if it doesn't exist (CSV Header)
if [ ! -f "$LOG_FILE" ]; then
    printf "%-10s %-10s %-15s %-15s\n" "YEAR" "DAY" "EXEC_TIME" "MAX_RAM_KB" > "$LOG_FILE"
    echo "------------------------------------------------------------" >> "$LOG_FILE"
fi

# 2. Main Loop
for file in $BASE_PATH/*; do
    filename=$(basename "$file")

    echo "Processing $filename"

    /usr/bin/time -a -o "$LOG_FILE" \
    -f "$filename       %E           %M" \
    python -u CloudSat_regrid.py "$filename"

    echo "Finished $filename"
done