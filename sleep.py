import argparse, yaml, os, datetime               # arg parsing, YAML config, filesystem, dates
from src.trainers.vision_trainer import train_vision
from src.trainers.audio_trainer import train_audio
from src.trainers.align_trainer import train_align

if __name__ == "__main__":                         # standard Python entry point
    ap = argparse.ArgumentParser()                 # CLI interface
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--date", default=None)        # allow manual date override
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))        # load hyperparameters & paths
    date = args.date or cfg["data"].get("today_date") or str(datetime.date.today())
    os.makedirs(cfg["outputs"]["report_dir"], exist_ok=True)
    report = os.path.join(cfg["outputs"]["report_dir"], f"{date}.md")

    v = train_vision(cfg, date)                    # run the vision JEPA pass
    a = train_audio(cfg, date)                     # run the speech CPC pass
    al = train_align(cfg, date)                    # align audio/video embeddings

    with open(report,"w") as f:                    # write a tiny nightly report
        f.write(f"# Nightly Report {date}\n\n")
        f.write(f"- Vision loss: **{v:.4f}**\n")
        f.write(f"- Audio (CPC) loss: **{a:.4f}**\n")
        f.write(f"- A/V align loss: **{al:.4f}**\n")
    print("Report:", report)
