"""Build clean TTS training lists from the JML transcript CSV and audio.

Usage (from kanbior-tts/):
    python prepare_data.py                          # report + multi-speaker lists
    python prepare_data.py --speakers JML_S082      # single-voice lists
    python prepare_data.py --speakers JML_S082,JML_S521

Steps:
  1. Keep clips whose WAV exists, with no digits, and clean the text (kanbior-mcore/fon_text.py).
  2. Measure each clip (cached in <out-dir>/audio_stats.csv, so re-runs are fast):
     loudness, an SNR estimate, clipping, share of speech frames, speaking rate.
  3. Drop clips that are too short/long, noisy, clipped, mostly silence, or whose
     speaking rate is far from their speaker's median (likely text/audio mismatch).
  4. Write <out-dir>/speakers.csv (speakers ranked by clean hours, to pick a voice)
     and <out-dir>/train.csv, val.csv (id, audio_path, speaker_id, text, duration).

Unlike the ASR splits, val uses the same speakers as train: the goal is to learn
these voices, not to generalize to new ones. Audio paths stay relative to --data-dir.
"""

import argparse
import csv
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kanbior-mcore"))
from fon_text import clean_text  # noqa: E402

_DIGIT_RE = re.compile(r"\d")
STAT_FIELDS = ["id", "sample_rate", "duration", "rms_db", "snr_db", "clip_frac", "speech_frac"]


def audio_stats(path: Path) -> dict:
    """Cheap quality measures from 25 ms frames (10 ms hop)."""
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    frame, hop = int(0.025 * sr), int(0.010 * sr)
    if len(audio) < frame:
        return {"sample_rate": sr, "duration": len(audio) / sr, "rms_db": -100.0,
                "snr_db": 0.0, "clip_frac": 0.0, "speech_frac": 0.0}
    n = 1 + (len(audio) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    energy_db = 10 * np.log10(np.mean(audio[idx] ** 2, axis=1) + 1e-10)
    noise_db, speech_db = np.percentile(energy_db, [10, 90])
    return {
        "sample_rate": sr,
        "duration": len(audio) / sr,
        "rms_db": float(10 * np.log10(np.mean(audio ** 2) + 1e-10)),
        "snr_db": float(speech_db - noise_db),  # speech level vs noise floor
        "clip_frac": float(np.mean(np.abs(audio) >= 0.99)),
        "speech_frac": float(np.mean(energy_db > noise_db + 0.5 * (speech_db - noise_db))),
    }


def load_stats(rows, data_dir: Path, cache_path: Path) -> dict:
    cache = {}
    if cache_path.is_file():
        with open(cache_path, newline="", encoding="utf-8") as f:
            cache = {r["id"]: {k: float(v) for k, v in r.items() if k != "id"} for r in csv.DictReader(f)}
    todo = [r for r in rows if r["id"] not in cache]
    for i, r in enumerate(todo, 1):
        cache[r["id"]] = audio_stats(data_dir / r["audio_path"])
        if i % 1000 == 0 or i == len(todo):
            print(f"  measured {i:,}/{len(todo):,} new clips")
    if todo:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=STAT_FIELDS)
            w.writeheader()
            w.writerows({"id": k, **v} for k, v in sorted(cache.items()))
    return cache


def chars_per_second(text: str, duration: float) -> float:
    return sum(ch.isalpha() for ch in text) / max(duration, 1e-3)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("../kanbior-asr/sample-data"),
                        help="folder holding transcribed.csv and audio/")
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--speakers", default="all",
                        help="'all' (multi-speaker) or comma-separated speaker ids for a single/few voices")
    parser.add_argument("--min-speaker-minutes", type=float, default=10.0,
                        help="with --speakers all: drop speakers with less clean audio than this")
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=15.0)
    parser.add_argument("--min-snr", type=float, default=25.0, help="dB, speech level vs noise floor")
    parser.add_argument("--max-clip-frac", type=float, default=0.001, help="share of clipped samples")
    parser.add_argument("--min-speech-frac", type=float, default=0.4, help="share of frames that are speech")
    parser.add_argument("--max-rate-ratio", type=float, default=1.5,
                        help="drop clips whose chars/s is > ratio x or < 1/ratio x their speaker's median")
    parser.add_argument("--val-clips", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def load_clips(data_dir: Path, cache_path: Path):
    """Usable clips (audio on disk, no digits, non-empty text) with audio stats and speaking rate."""
    with open(data_dir / "transcribed.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    stats = defaultdict(int, total=len(rows))
    clips = []
    for row in rows:
        if not (data_dir / row["audio_path"]).is_file():
            stats["no audio"] += 1
        elif _DIGIT_RE.search(row["transcription"]):
            stats["digits"] += 1
        elif not (text := clean_text(row["transcription"])):
            stats["empty"] += 1
        else:
            clips.append({"id": row["id"], "audio_path": row["audio_path"],
                          "speaker_id": row["speaker_id"], "text": text})

    print(f"Measuring audio ({len(clips):,} clips with audio)...")
    measures = load_stats(clips, data_dir, cache_path)
    rates = defaultdict(list)
    for c in clips:
        c.update(measures[c["id"]])
        c["rate"] = chars_per_second(c["text"], c["duration"])
        rates[c["speaker_id"]].append(c["rate"])
    median_rate = {s: statistics.median(v) for s, v in rates.items()}
    for c in clips:
        c["rate_ratio"] = c["rate"] / median_rate[c["speaker_id"]]
    return clips, stats, median_rate


def drop_reason(c: dict, args) -> str | None:
    """Why a clip is rejected for TTS, or None if it is kept."""
    if not args.min_duration <= c["duration"] <= args.max_duration:
        return "duration"
    if c["snr_db"] < args.min_snr:
        return "noisy"
    if c["clip_frac"] > args.max_clip_frac:
        return "clipped"
    if c["speech_frac"] < args.min_speech_frac:
        return "mostly silence"
    if not 1 / args.max_rate_ratio <= c["rate_ratio"] <= args.max_rate_ratio:
        return "rate mismatch"
    return None


def main():
    args = build_parser().parse_args()
    clips, stats, median_rate = load_clips(args.data_dir, args.out_dir / "audio_stats.csv")

    kept = []
    for c in clips:
        reason = drop_reason(c, args)
        if reason:
            stats[reason] += 1
        else:
            kept.append(c)

    # Speaker report: all speakers with audio, ranked by clean hours.
    by_speaker = defaultdict(list)
    for c in kept:
        by_speaker[c["speaker_id"]].append(c)
    all_speakers = {c["speaker_id"] for c in clips}
    report = []
    for s in all_speakers:
        ks = by_speaker.get(s, [])
        n_all = sum(1 for c in clips if c["speaker_id"] == s)
        report.append({
            "speaker_id": s,
            "clean_hours": round(sum(c["duration"] for c in ks) / 3600, 3),
            "clean_clips": len(ks),
            "kept_frac": round(len(ks) / n_all, 3),
            "median_snr_db": round(statistics.median(c["snr_db"] for c in ks), 1) if ks else "",
            "median_rms_db": round(statistics.median(c["rms_db"] for c in ks), 1) if ks else "",
            "median_chars_per_s": round(median_rate[s], 2),
        })
    report.sort(key=lambda r: -r["clean_hours"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "speakers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0]))
        w.writeheader()
        w.writerows(report)

    # Select speakers.
    if args.speakers == "all":
        chosen = {r["speaker_id"] for r in report if r["clean_hours"] * 60 >= args.min_speaker_minutes}
    else:
        chosen = set(args.speakers.split(","))
        missing = chosen - all_speakers
        if missing:
            sys.exit(f"No downloaded audio for speaker(s): {', '.join(sorted(missing))}")
    selected = sorted((c for c in kept if c["speaker_id"] in chosen), key=lambda c: c["id"])

    random.Random(args.seed).shuffle(selected)
    val, train = selected[:args.val_clips], selected[args.val_clips:]
    fields = ["id", "audio_path", "speaker_id", "text", "duration"]
    for name, split in (("train", train), ("val", val)):
        with open(args.out_dir / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(sorted(split, key=lambda c: c["id"]))

    srs = sorted({int(c["sample_rate"]) for c in clips})
    dropped = ", ".join(f"{k} {v:,}" for k, v in stats.items() if k != "total")
    print(f"Rows in CSV: {stats['total']:,}  | dropped: {dropped}")
    print(f"Clean: {len(kept):,} clips, {sum(c['duration'] for c in kept) / 3600:.2f} h, "
          f"{len(by_speaker)} speakers  | sample rate(s): {srs}")
    print("Top speakers by clean hours (full list in speakers.csv):")
    for r in report[:5]:
        print(f"  {r['speaker_id']}  {r['clean_hours']:5.2f} h  {r['clean_clips']:5,} clips  "
              f"SNR {r['median_snr_db']} dB  kept {r['kept_frac']:.0%}")
    print(f"Selected {len(chosen)} speaker(s) -> train {len(train):,} clips "
          f"({sum(c['duration'] for c in train) / 3600:.2f} h), val {len(val):,}  -> {args.out_dir}/")


if __name__ == "__main__":
    main()
