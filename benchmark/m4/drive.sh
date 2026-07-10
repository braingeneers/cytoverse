#!/bin/bash
# drive.sh COOLDOWN_S  "cells:device:rep" "cells:device:rep" ...
# Cools down BEFORE each run (so we start cold), runs it, appends a line to progress.log
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
COOL=$1; shift
PROG="$DIR/results/progress.log"
stamp(){ date '+%H:%M:%S'; }
echo "$(stamp) DRIVER START cooldown=${COOL}s runs=$*" >> "$PROG"
for spec in "$@"; do
  IFS=: read -r C D R <<< "$spec"
  echo "$(stamp) cooldown ${COOL}s before ${C}/${D}/r${R}" >> "$PROG"
  sleep "$COOL"
  echo "$(stamp) RUN ${C}/${D}/r${R} start" >> "$PROG"
  if OUT=$(MAX_MS="${MAX_MS:-14400000}" bash "$DIR/run_one.sh" "$C" "$D" "$R" 2>&1); then
    echo "$(stamp) DONE ${OUT#  }" >> "$PROG"
  else
    echo "$(stamp) FAIL ${C}/${D}/r${R}: ${OUT}" >> "$PROG"
  fi
done
echo "$(stamp) DRIVER DONE" >> "$PROG"
