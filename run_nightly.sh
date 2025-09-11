#!/usr/bin/env bash
set -euo pipefail

# --- CONFIG (edit these paths for your machine) ---
PROJECT_DIR="$HOME/world-model-poc"
PY_BIN="$PROJECT_DIR/.venv/bin/python"     # or "python3" if you didn't make a venv
DATE="$(date +%F)"                         # e.g., 2025-09-04
LOG_DIR="$PROJECT_DIR/logs"
REPORT_DIR="$PROJECT_DIR/reports"
RAW_DIR="$PROJECT_DIR/data/raw/$DATE"      # drop today's raw .mp4 here (or change)
CLIP_SECONDS=3
FRAME_RATE=1

mkdir -p "$LOG_DIR" "$REPORT_DIR" "$RAW_DIR"

# --- environment hygiene for launchd (no inherited shell state) ---
export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export PYTHONUNBUFFERED=1

# --- sanity checks ---
command -v ffmpeg >/dev/null || { echo "ffmpeg not found in PATH"; exit 1; }
[ -x "$PY_BIN" ] || { echo "Python not found at $PY_BIN"; exit 1; }

# --- optional: activate venv if you prefer that style ---
# source "$PROJECT_DIR/.venv/bin/activate"

# --- ingest (segmentation + frames + audio) ---
# Will NO-OP if there are no mp4s in data/raw/$DATE
"$PY_BIN" "$PROJECT_DIR/scripts/ingest_day.py" \
  --date "$DATE" --clip_seconds "$CLIP_SECONDS" --frame_rate "$FRAME_RATE" \
  >>"$LOG_DIR/$DATE.ingest.log" 2>&1 || true

# --- nightly "sleep" (vision, audio, alignment, report) ---
"$PY_BIN" "$PROJECT_DIR/sleep.py" \
  --date "$DATE" --config "$PROJECT_DIR/configs/default.yaml" \
  >>"$LOG_DIR/$DATE.sleep.log" 2>&1

# --- summarize location of report (makes grepping logs easy) ---
echo "Report: $REPORT_DIR/$DATE.md" >>"$LOG_DIR/$DATE.sleep.log"

# Weekly anchor on Sundays (1=Mon ... 7=Sun in BSD date; adjust if needed)
DOW=$(date +%u)
if [ "$DOW" = "7" ]; then
  ANCHOR_DIR="$PROJECT_DIR/checkpoints/weekly"
  mkdir -p "$ANCHOR_DIR"
  cp "$PROJECT_DIR/checkpoints/daily/${DATE}_teacher.pt" "$ANCHOR_DIR/${DATE}_teacher.pt" 2>/dev/null || true
  echo "Weekly anchor saved: $ANCHOR_DIR/${DATE}_teacher.pt" >>"$LOG_DIR/$DATE.sleep.log"
fi
