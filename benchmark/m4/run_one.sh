#!/bin/bash
# run_one.sh CELLS DEVICE REP  — one Phase A/B point, saves results/<cells>_<device>_r<rep>.json
set -euo pipefail
CELLS=$1; DEVICE=$2; REP=${3:-1}
DIR="$(cd "$(dirname "$0")" && pwd)"
H5AD="/Users/rcurrie/data/cytoverse.main/metaatlas_cells_${CELLS}.h5ad"
OUT="$DIR/results/${CELLS}_${DEVICE}_r${REP}.json"
[ -f "$H5AD" ] || { echo "missing $H5AD"; exit 1; }
# m4bench.mjs drives a separately-launched `vite preview`; it does not start one. Check it is up
# before the caller pays a multi-minute cooldown only to fail on ERR_CONNECTION_REFUSED.
BASE="${BASE:-http://localhost:4173/}"
curl -sfI --max-time 5 "$BASE" > /dev/null || {
  echo "no preview server at $BASE - run 'npm run preview' in the repo root first"; exit 1; }
# Always headed: headless Chromium truncates the h5ad read to 1000 cells (verified), so
# headless CPU runs silently under-process. GPU also needs headed for WebGPU on macOS.
HEADED_FLAG=1
LOG="$(mktemp)"
HEADED=$HEADED_FLAG DEVICE=$DEVICE MODEL=scimilarity H5AD="$H5AD" \
  MAX_MS="${MAX_MS:-14400000}" node "$DIR/m4bench.mjs" > "$LOG" 2>&1 || { echo "RUN FAILED"; tail -20 "$LOG"; exit 1; }
# extract the JSON block after the M4_RESULT marker
sed -n '/^M4_RESULT/,$p' "$LOG" | sed '1s/^M4_RESULT //' > "$OUT"
python3 - "$OUT" "$CELLS" "$DEVICE" "$REP" <<'PY'
import json,sys
o=json.load(open(sys.argv[1]))
print(f"  {sys.argv[2]:>7} {sys.argv[3]:>3} r{sys.argv[4]}: wall={o['wall_s']}s peak={o['peak_renderer_rss_mb']}MB "
      f"at_rest={o['at_rest_renderer_rss_mb']}MB finished={o['finished']} fell_back={o['fell_back_to_wasm']} "
      f"ep={'WebGPU' if 'WebGPU' in o['embedding_ep_log'] else 'WASM'}")
PY
