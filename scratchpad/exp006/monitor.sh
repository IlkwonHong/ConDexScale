#!/bin/bash
# EXP-005/006 진행 감시: 새 이벤트만 한 줄씩 출력. EXP006_DONE 이면 종료.
cd "$(dirname "$0")/../.."
seen="f0.25_s0 f0.25_s1 f0.5_s0 f0.5_s1 $*"
while true; do
  for d in checkpoints/exp005_*; do
    t=${d#checkpoints/exp005_}
    n=$(grep -c '"epoch"' $d/log.jsonl 2>/dev/null)
    if [ "$n" = "30" ] && [[ " $seen " != *" $t "* ]]; then echo "EXP-005 완료 $t"; seen="$seen $t"; fi
  done
  for f in scratchpad/exp006/*.done; do
    b=$(basename "$f")
    if [ -s "$f" ] && [[ " $seen " != *" $b "* ]]; then echo "EXP-006 단계 끝: $b $(cut -c1-200 "$f")"; seen="$seen $b"; fi
  done
  for f in scratchpad/exp005_all.log scratchpad/exp006/run_pilot.log scratchpad/exp006/search_records_*.log scratchpad/exp006/gate_*.log; do
    [ -f "$f" ] || continue
    b=ERR$(basename "$f")
    if grep -qE "Traceback|Error:" "$f" && [[ " $seen " != *" $b "* ]]; then echo "오류 의심: $f"; seen="$seen $b"; fi
  done
  if grep -q EXP006_DONE scratchpad/exp006/run_pilot.log 2>/dev/null; then echo EXP006_DONE; exit 0; fi
  sleep 30
done
