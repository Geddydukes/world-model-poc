import argparse, os, glob, subprocess

def run(cmd):
    print("+", " ".join(cmd)); subprocess.run(cmd, check=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--clip_seconds", type=int, default=3)
    ap.add_argument("--frame_rate", type=int, default=1)
    args = ap.parse_args()

    raw_dir = f"data/raw/{args.date}"
    clip_dir = f"data/clips/{args.date}"
    frame_dir = f"data/frames/{args.date}"
    audio_dir = f"data/audio/{args.date}/segments"
    for d in [clip_dir, frame_dir, audio_dir]: os.makedirs(d, exist_ok=True)

    videos = sorted(glob.glob(f"{raw_dir}/*.mp4"))
    if not videos: raise SystemExit(f"No videos in {raw_dir}")

    for i, v in enumerate(videos, 1):
        # split into 3s clips
        run(["ffmpeg","-i",v,"-c","copy","-map","0","-segment_time",str(args.clip_seconds),
             "-f","segment","-reset_timestamps","1", f"{clip_dir}/clip_{i:04d}_%04d.mp4"])

    for i, c in enumerate(sorted(glob.glob(f"{clip_dir}/*.mp4")), 1):
        # extract frames
        out_frames = f"{frame_dir}/clip_{i:04d}"
        os.makedirs(out_frames, exist_ok=True)
        run(["ffmpeg","-i",c,"-vf",f"fps={args.frame_rate}", f"{out_frames}/frame_%04d.jpg"])
        # extract audio @16kHz mono
        run(["ffmpeg","-i",c,"-vn","-acodec","pcm_s16le","-ar","16000","-ac","1", f"{audio_dir}/clip_{i:04d}.wav"])
