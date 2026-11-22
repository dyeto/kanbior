"""Build clean, speaker-disjoint train/val/test splits from the JML transcript CSV.

Usage (from kanbior-asr/):
    python prepare_data.py
    python prepare_data.py --data-dir sample-data --out-dir sample-data/splits --seed 42

Only clips whose WAV file exists on disk are kept, so re-running this after
downloading more audio grows the splits. Speakers never appear in more than
one split.
"""

import argparse
import csv
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

# Character fixes found by auditing transcribed.csv.
CHAR_MAP = {
    "‚": ",",  # ‚ SINGLE LOW-9 QUOTATION MARK, used as a comma in ~430 rows
    "‘": "'",  # ‘
    "’": "'",  # ’
    "ε": "ɛ",  # Greek ε -> Fon ɛ
    "Ε": "Ɛ",  # Greek Ε -> Fon Ɛ
    "Ð": "Ɖ",  # Latin Ð (eth) -> Fon Ɖ
    "“": "",  # “
    "”": "",  # ”
    '"': "",  # quotes around French loanwords are not spoken
    "«": "",  # «
    "»": "",  # »
    "*": "",
}
_CHAR_RE = re.compile("|".join(re.escape(k) for k in CHAR_MAP))
_SPACE_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;:!?])")
_DIGIT_RE = re.compile(r"\d")


def clean_text(text: str) -> str:
    """Normalize a transcription for training (keeps case, tones and punctuation)."""
    text = unicodedata.normalize("NFC", text)
    text = _CHAR_RE.sub(lambda m: CHAR_MAP[m.group(0)], text)
    text = _SPACE_RE.sub(" ", text)  # also joins multi-line transcriptions
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)  # "nɛ́ ?" -> "nɛ́?"
    return unicodedata.normalize("NFC", text).strip()


def normalize_for_wer(text: str, keep_tones: bool = True) -> str:
    """Normalize text before scoring: lowercase, no punctuation, optionally no tones."""
    text = clean_text(text).lower()
    if not keep_tones:
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = "".join(ch if (ch.isalnum() or ch in " '" or unicodedata.category(ch) == "Mn") else " " for ch in text)
    return unicodedata.normalize("NFC", _SPACE_RE.sub(" ", text)).strip()


def read_split(path) -> list[dict]:
    """Load a split CSV written by this script."""
    with open(path, newline="", encoding="utf-8") as f:
        return [{**row, "duration": float(row["duration"])} for row in csv.DictReader(f)]


def split_speakers(clips_by_speaker, val_frac, test_frac, seed, max_eval_speaker_share):
    """Assign whole speakers to val/test until each reaches its share of total hours.

    Speakers holding more than `max_eval_speaker_share` of all audio always go to
    train, so one heavy speaker cannot dominate an evaluation set.
    """
    hours = {s: sum(r["duration"] for r in rows) for s, rows in clips_by_speaker.items()}
    total = sum(hours.values())
    speakers = sorted(hours)
    random.Random(seed).shuffle(speakers)

    split_of = {}
    targets = {"test": test_frac * total, "val": val_frac * total}
    filled = {"test": 0.0, "val": 0.0}
    for s in speakers:
        if hours[s] > max_eval_speaker_share * total:
            split_of[s] = "train"
            continue
        for name in ("test", "val"):
            if filled[name] + hours[s] <= targets[name] * 1.05:
                split_of[s] = name
                filled[name] += hours[s]
                break
        else:
            split_of[s] = "train"
    return split_of


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("sample-data"), help="folder holding transcribed.csv and audio/")
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <data-dir>/splits")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-eval-speaker-share", type=float, default=0.01,
                        help="speakers above this share of total audio stay in train")
    args = parser.parse_args()
    out_dir = args.out_dir or args.data_dir / "splits"

    with open(args.data_dir / "transcribed.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    stats = defaultdict(int, total=len(rows))
    clips_by_speaker = defaultdict(list)
    for row in rows:
        if not (args.data_dir / row["audio_path"]).is_file():
            stats["no_audio"] += 1
            continue
        if _DIGIT_RE.search(row["transcription"]):
            stats["digits"] += 1  # audio says the number in words; text does not
            continue
        text = clean_text(row["transcription"])
        if not text:
            stats["empty"] += 1
            continue
        clips_by_speaker[row["speaker_id"]].append({
            "id": row["id"],
            "audio_path": row["audio_path"],
            "text": text,
            "speaker_id": row["speaker_id"],
            "duration": float(row["duration"]),
        })

    split_of = split_speakers(clips_by_speaker, args.val_frac, args.test_frac, args.seed, args.max_eval_speaker_share)

    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["id", "audio_path", "text", "speaker_id", "duration"]
    print(f"Rows in CSV: {stats['total']:,}  | skipped: no audio {stats['no_audio']:,}, "
          f"digits {stats['digits']:,}, empty {stats['empty']:,}")
    for name in ("train", "val", "test"):
        split_rows = sorted(
            (r for s, rs in clips_by_speaker.items() if split_of[s] == name for r in rs),
            key=lambda r: r["id"],
        )
        with open(out_dir / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(split_rows)
        n_spk = sum(1 for s in split_of if split_of[s] == name)
        hrs = sum(r["duration"] for r in split_rows) / 3600
        print(f"{name:5s}: {len(split_rows):6,} clips  {hrs:6.2f} h  {n_spk:4d} speakers  -> {out_dir / (name + '.csv')}")


if __name__ == "__main__":
    main()
