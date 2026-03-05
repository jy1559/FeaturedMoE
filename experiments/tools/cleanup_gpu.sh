#!/bin/bash
# GPU cleanup script for user jy1559
# Usage: ./tools/cleanup_gpu.sh [--kill]

set -e

USER="jy1559"

echo "=== GPU processes for user $USER ==="
echo ""

# Get GPU processes for this user
nvidia-smi --query-compute-apps=pid,gpu_name,used_memory --format=csv,noheader 2>/dev/null | while read line; do
    if [ -n "$line" ]; then
        PID=$(echo "$line" | cut -d',' -f1 | tr -d ' ')
        if [ -n "$PID" ] && [ "$PID" != "[N/A]" ]; then
            PROC_USER=$(ps -o user= -p "$PID" 2>/dev/null | tr -d ' ')
            if [ "$PROC_USER" = "$USER" ]; then
                GPU=$(echo "$line" | cut -d',' -f2)
                MEM=$(echo "$line" | cut -d',' -f3)
                CMD=$(ps -o cmd= -p "$PID" 2>/dev/null | head -c 60)
                echo "PID: $PID | GPU:$GPU | Mem:$MEM | $CMD"
            fi
        fi
    fi
done

echo ""

# Kill mode
if [ "$1" = "--kill" ]; then
    echo "Killing all GPU processes for $USER..."
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read PID; do
        PID=$(echo "$PID" | tr -d ' ')
        if [ -n "$PID" ] && [ "$PID" != "[N/A]" ]; then
            PROC_USER=$(ps -o user= -p "$PID" 2>/dev/null | tr -d ' ')
            if [ "$PROC_USER" = "$USER" ]; then
                echo "Killing PID $PID..."
                kill -9 "$PID" 2>/dev/null || true
            fi
        fi
    done
    echo "Done!"
else
    echo "To kill all processes: $0 --kill"
fi
