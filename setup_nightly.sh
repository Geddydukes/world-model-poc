#!/usr/bin/env bash
set -euo pipefail

echo "🌙 Setting up nightly world model training..."

# Get the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Project directory: $PROJECT_DIR"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/reports"
mkdir -p "$PROJECT_DIR/data/raw"
mkdir -p "$PROJECT_DIR/checkpoints/daily"
mkdir -p "$PROJECT_DIR/checkpoints/weekly"

# Check for ffmpeg
echo "🔍 Checking for ffmpeg..."
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "❌ ffmpeg not found. Please install it with:"
    echo "   brew install ffmpeg"
    exit 1
else
    echo "✅ ffmpeg found: $(which ffmpeg)"
fi

# Check for Python virtual environment
echo "🐍 Checking Python environment..."
if [ -f "$PROJECT_DIR/.venv/bin/python" ]; then
    echo "✅ Virtual environment found"
    PY_BIN="$PROJECT_DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    echo "✅ Python3 found: $(which python3)"
    PY_BIN="python3"
else
    echo "❌ Python not found. Please install Python 3.8+ or create a virtual environment"
    exit 1
fi

# Test the nightly script
echo "🧪 Testing nightly script..."
if [ -x "$PROJECT_DIR/run_nightly.sh" ]; then
    echo "✅ Nightly script is executable"
else
    echo "❌ Nightly script is not executable. Fixing..."
    chmod +x "$PROJECT_DIR/run_nightly.sh"
fi

# Install LaunchAgent
echo "📋 Installing LaunchAgent..."
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
mkdir -p "$LAUNCH_AGENTS_DIR"

# Copy the plist file
cp "$PROJECT_DIR/com.worldmodel.sleep.plist" "$LAUNCH_AGENTS_DIR/"

# Unload any existing agent
launchctl unload "$LAUNCH_AGENTS_DIR/com.worldmodel.sleep.plist" 2>/dev/null || true

# Load the new agent
launchctl load "$LAUNCH_AGENTS_DIR/com.worldmodel.sleep.plist"

echo "✅ LaunchAgent installed and loaded"

# Show status
echo ""
echo "📊 LaunchAgent status:"
launchctl list | grep com.worldmodel.sleep || echo "   (not running yet - will start at 2:00 AM)"

echo ""
echo "🎯 Setup complete! Here's what to do next:"
echo ""
echo "1. Drop your daily video files in:"
echo "   $PROJECT_DIR/data/raw/YYYY-MM-DD/*.mp4"
echo ""
echo "2. The system will automatically:"
echo "   - Run every night at 2:00 AM"
echo "   - Process any new videos"
echo "   - Train the models"
echo "   - Save reports to $PROJECT_DIR/reports/"
echo ""
echo "3. Check logs:"
echo "   tail -f $PROJECT_DIR/logs/\$(date +%F).sleep.log"
echo ""
echo "4. Test manually:"
echo "   $PROJECT_DIR/run_nightly.sh"
echo ""
echo "5. To stop the nightly runs:"
echo "   launchctl unload $LAUNCH_AGENTS_DIR/com.worldmodel.sleep.plist"
echo ""
echo "6. To wake your Mac before training (optional):"
echo "   sudo pmset repeat wakeorpoweron MTWRFSU 01:55:00"
echo ""
echo "🌙 Happy sleeping! Your AI will learn while you dream."
