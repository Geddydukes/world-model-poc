# Nightly Training Setup Guide

This guide will help you set up automated nightly training for your world model system.

## Quick Setup

Run the automated setup script:

```bash
cd ~/world-model-poc
./setup_nightly.sh
```

This will:
- Create necessary directories
- Check for dependencies (ffmpeg, Python)
- Install the LaunchAgent for scheduling
- Test the nightly script

## Manual Setup

If you prefer to set up manually:

### 1. Install Dependencies

```bash
# Install ffmpeg for video processing
brew install ffmpeg

# Verify Python environment
python3 --version  # Should be 3.8+
```

### 2. Create Directories

```bash
mkdir -p ~/world-model-poc/{logs,reports,data/raw,checkpoints/{daily,weekly}}
```

### 3. Install LaunchAgent

```bash
# Copy the plist file to LaunchAgents directory
cp ~/world-model-poc/com.worldmodel.sleep.plist ~/Library/LaunchAgents/

# Load the agent
launchctl load ~/Library/LaunchAgents/com.worldmodel.sleep.plist
```

### 4. Test the Setup

```bash
# Test the nightly script manually
~/world-model-poc/run_nightly.sh

# Check the logs
tail -f ~/world-model-poc/logs/$(date +%F).sleep.log
```

## Daily Workflow

### 1. Drop Video Files

Place your daily video files in:
```
~/world-model-poc/data/raw/YYYY-MM-DD/*.mp4
```

Example:
```bash
# For today's videos
mkdir -p ~/world-model-poc/data/raw/$(date +%F)
cp ~/Desktop/daily_videos/*.mp4 ~/world-model-poc/data/raw/$(date +%F)/
```

### 2. Automatic Processing

The system will automatically:
- Run every night at 2:00 AM
- Process any videos from `data/raw/YYYY-MM-DD/`
- Extract frames and audio segments
- Train vision, audio, and alignment models
- Save embeddings to episodic memory
- Generate daily reports

### 3. Check Results

```bash
# View today's report
cat ~/world-model-poc/reports/$(date +%F).md

# Check training logs
tail -n 50 ~/world-model-poc/logs/$(date +%F).sleep.log

# Query your memories
python3 ~/world-model-poc/scripts/query_memory.py --query_path path/to/image.jpg
```

## Configuration

### Adjust Training Time

Edit `~/Library/LaunchAgents/com.worldmodel.sleep.plist`:

```xml
<key>StartCalendarInterval</key>
<dict>
  <key>Hour</key><integer>3</integer>    <!-- Change to 3 AM -->
  <key>Minute</key><integer>30</integer> <!-- Change to 3:30 AM -->
</dict>
```

Then reload:
```bash
launchctl unload ~/Library/LaunchAgents/com.worldmodel.sleep.plist
launchctl load ~/Library/LaunchAgents/com.worldmodel.sleep.plist
```

### Adjust Video Processing

Edit `~/world-model-poc/run_nightly.sh`:

```bash
CLIP_SECONDS=5        # Longer clips
FRAME_RATE=2          # Higher frame rate
RAW_DIR="$PROJECT_DIR/data/raw/$DATE"  # Different input directory
```

### Adjust Model Parameters

Edit `~/world-model-poc/configs/default.yaml`:

```yaml
train:
  steps_vision: 10000  # More training steps
  batch_size: 32       # Larger batch size
  lr: 2.0e-4          # Higher learning rate
```

## Monitoring

### Check LaunchAgent Status

```bash
# List all agents
launchctl list | grep com.worldmodel.sleep

# Check if it's running
launchctl list com.worldmodel.sleep
```

### View Logs

```bash
# Today's training log
tail -f ~/world-model-poc/logs/$(date +%F).sleep.log

# Today's ingestion log
tail -f ~/world-model-poc/logs/$(date +%F).ingest.log

# LaunchAgent system logs
tail -f ~/world-model-poc/logs/launchd.out.log
tail -f ~/world-model-poc/logs/launchd.err.log
```

### Check Memory Usage

```bash
# View episodic memory database
sqlite3 ~/world-model-poc/memory/episodic.sqlite "SELECT COUNT(*) FROM clips;"

# List all stored embeddings
ls -la ~/world-model-poc/memory/embeddings/
```

## Troubleshooting

### Common Issues

1. **"ffmpeg not found"**
   ```bash
   brew install ffmpeg
   ```

2. **"Python not found"**
   ```bash
   # Create virtual environment
   python3 -m venv ~/world-model-poc/.venv
   source ~/world-model-poc/.venv/bin/activate
   pip install -r requirements.txt
   ```

3. **"Permission denied"**
   ```bash
   chmod +x ~/world-model-poc/run_nightly.sh
   ```

4. **LaunchAgent not running**
   ```bash
   # Reload the agent
   launchctl unload ~/Library/LaunchAgents/com.worldmodel.sleep.plist
   launchctl load ~/Library/LaunchAgents/com.worldmodel.sleep.plist
   ```

5. **Mac sleeps during training**
   ```bash
   # Wake Mac before training
   sudo pmset repeat wakeorpoweron MTWRFSU 01:55:00
   ```

### Debug Mode

Run the nightly script manually to see detailed output:

```bash
# Run with verbose output
~/world-model-poc/run_nightly.sh 2>&1 | tee debug.log
```

### Check System Resources

```bash
# Monitor CPU and memory during training
top -pid $(pgrep -f "sleep.py")

# Check disk space
df -h ~/world-model-poc/
```

## Advanced Features

### Weekly Snapshots

The system automatically creates weekly anchor snapshots on Sundays in `checkpoints/weekly/`.

### Custom Wake Schedule

Set your Mac to wake before training:

```bash
# Wake at 1:55 AM every day
sudo pmset repeat wakeorpoweron MTWRFSU 01:55:00

# Cancel wake schedule
sudo pmset repeat cancel
```

### Multiple Date Processing

Process videos from multiple dates:

```bash
# Process yesterday's videos
~/world-model-poc/run_nightly.sh
# Then manually run for specific date
cd ~/world-model-poc
python3 sleep.py --date 2025-01-14
```

## Stopping the System

```bash
# Stop nightly runs
launchctl unload ~/Library/LaunchAgents/com.worldmodel.sleep.plist

# Remove the agent completely
rm ~/Library/LaunchAgents/com.worldmodel.sleep.plist
```

## Security Notes

- The system needs Full Disk Access to read videos from protected folders
- Grant Terminal/iTerm Full Disk Access in System Settings → Privacy & Security
- The system only processes local files and doesn't send data anywhere

## Performance Tips

1. **Use SSD storage** for faster I/O during training
2. **Close other applications** during training to free up memory
3. **Use smaller batch sizes** if you have limited GPU memory
4. **Process videos in smaller chunks** if you have very long recordings

## Support

If you encounter issues:

1. Check the logs first: `~/world-model-poc/logs/`
2. Run the setup script again: `./setup_nightly.sh`
3. Test manually: `~/world-model-poc/run_nightly.sh`
4. Check system resources and permissions
