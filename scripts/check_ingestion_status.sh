#!/bin/bash
# Quick status check for CLEVRER ingestion

cd "$(dirname "$0")/.."

echo "🔍 CLEVRER Ingestion Status Check"
echo "=================================="
echo ""

# Check if process is running
if ps aux | grep -v grep | grep -q "ingest_clevrer"; then
    echo "✅ Process is RUNNING"
    ps aux | grep -v grep | grep "ingest_clevrer" | head -1 | awk '{print "   PID: " $2 " | CPU: " $3 "% | Memory: " $4 "%"}'
else
    echo "❌ Process is NOT running"
fi

echo ""

# Check progress
python3 scripts/check_clevrer_progress.py

echo ""
echo "📝 Recent log output:"
tail -10 /tmp/clevrer_ingest_full.log 2>/dev/null | grep -v "UserWarning" | tail -5 || echo "   (No log output yet)"

echo ""
echo "💡 To watch live: tail -f /tmp/clevrer_ingest_full.log"

