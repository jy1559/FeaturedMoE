#!/usr/bin/env bash
# 실시간 P0 grid 진행 상황 모니터
# Usage: bash watch_p0_grid.sh          # 5초마다 갱신
#        bash watch_p0_grid.sh 10       # 10초마다 갱신

INTERVAL=${1:-5}
LOG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../logs/cue_perturb_p0_grid"

while true; do
    clear
    echo "══════════════════════════════════════════════════════════════════"
    printf "  P0 Grid Monitor  —  %s  (refresh every ${INTERVAL}s)\n" "$(date '+%H:%M:%S')"
    echo "══════════════════════════════════════════════════════════════════"
    echo ""

    # GPU 상태
    echo "── GPU 상태 ────────────────────────────────────────────────────"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
               --format=csv,noheader,nounits 2>/dev/null | \
    awk -F', ' '{printf "  GPU %s  util=%3s%%  mem=%5s/%5s MiB  temp=%s°C\n", $1,$3,$4,$5,$6}'
    echo ""

    # 각 로그별 진행 상황
    echo "── 학습 진행 ───────────────────────────────────────────────────"
    for log in "$LOG_DIR"/*.log; do
        [ -f "$log" ] || continue
        name=$(basename "$log" .log)

        # 마지막 epoch 라인 추출
        last_ep=$(grep -E "^Ep +[0-9]+" "$log" 2>/dev/null | tail -1)

        if [ -z "$last_ep" ]; then
            # epoch 시작 전 — 데이터 로딩 중
            last_status=$(tail -3 "$log" 2>/dev/null | grep -v "^$" | tail -1 | cut -c1-80)
            printf "  %-38s  [로딩중] %s\n" "$name" "$last_status"
        else
            # epoch 정보 파싱
            ep=$(echo "$last_ep" | grep -oP 'Ep\s+\K[0-9]+(?=/)')
            max_ep=$(echo "$last_ep" | grep -oP '/\K[0-9]+')
            eta=$(echo "$last_ep" | grep -oP 'ETA \K\S+')
            mrr=$(echo "$last_ep" | grep -oP 'best M@20\s+\K[0-9.]+')
            pat=$(echo "$last_ep" | grep -oP 'pat\s+\K[0-9]+(?=/)')
            max_pat=$(echo "$last_ep" | grep -oP 'pat\s+[0-9]+/\K[0-9]+')
            t=$(echo "$last_ep" | grep -oP 'time\s+\K[0-9.]+')

            # 완료 여부
            if grep -q "Results ->" "$log" 2>/dev/null; then
                final_mrr=$(grep -oP 'mrr@20.*?([0-9.]+)' "$log" 2>/dev/null | tail -1 | grep -oP '[0-9.]+$')
                printf "  %-38s  ✅ DONE  best MRR@20=%s\n" "$name" "$mrr"
            elif grep -q "ERROR\|Traceback\|error_rc" "$log" 2>/dev/null; then
                printf "  %-38s  ❌ ERROR\n" "$name"
            else
                bar_filled=$(( ep * 20 / (max_ep > 0 ? max_ep : 1) ))
                bar=$(printf '%0.s█' $(seq 1 $bar_filled 2>/dev/null))
                bar_empty=$(printf '%0.s░' $(seq 1 $((20 - bar_filled)) 2>/dev/null))
                printf "  %-38s  [%s%s] %3s/%s  ETA %-8s  best=%s  pat %s/%s  %.1fs/ep\n" \
                    "$name" "$bar" "$bar_empty" \
                    "$ep" "$max_ep" "$eta" "$mrr" "$pat" "$max_pat" "${t:-0}"
            fi
        fi
    done

    # 아직 로그 없는 job (foursquare — 대기 중)
    echo ""
    missing=0
    for ds in foursquare KuaiRec; do
        for i in 0 1 2 3; do
            found=0
            for log in "$LOG_DIR"/${ds}_c${i}_*.log; do
                [ -f "$log" ] && found=1 && break
            done
            if [ $found -eq 0 ]; then
                [ $missing -eq 0 ] && echo "── 대기 중 (미시작) ────────────────────────────────────────────"
                printf "  %s c%s\n" "$ds" "$i"
                missing=1
            fi
        done
    done

    echo ""
    echo "  Ctrl+C 로 종료"
    sleep "$INTERVAL"
done
