# Kanbior

Voice assistant in Fon: ask a question by voice in Fon, hear the answer in Fon.
This repo currently holds **kanbior-asr**, the Fon speech recognition component.
See `Kanbior state of the art for Fon speech AI.pdf` for the background and plan.

## 1. Set up the environment

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv venv .kanbior-env --python 3.12
source .kanbior-env/bin/activate
uv pip install -r requirements.txt
```

Run `source .kanbior-env/bin/activate` in each new terminal before the commands below.
In VS Code / Jupyter, select `.kanbior-env` as the notebook kernel.

## 2. Download the data

Data comes from [JML-fon-speech-v1](https://huggingface.co/datasets/JaimeMaLangue/JML-fon-speech-v1)
(J'aime Ma Langue, CC-BY-SA-4.0). The transcribed audio is split into 8 archives of ~2 GB each;
start with the first one (~8,300 clips, ~17 h).

From the project root:

```bash
cd kanbior-asr/sample-data
hf download JaimeMaLangue/JML-fon-speech-v1 \
    transcribed.csv audio_transcribed_000.tar --repo-type dataset --local-dir .
tar -xf audio_transcribed_000.tar   # creates audio/transcribed/tr_*.wav
rm audio_transcribed_000.tar
```

Expected layout:

```
kanbior-asr/sample-data/
├── transcribed.csv          # all 64.6k transcripts (only some have audio locally)
└── audio/transcribed/       # tr_000001.wav, tr_000002.wav, ...
```

For more data, download and extract `audio_transcribed_001.tar`, `002`, ... the same way.

## 3. Prepare the dataset

From the project root:

```bash
cd kanbior-asr
python prepare_data.py
```

This keeps only clips whose audio is downloaded, cleans the text (Unicode NFC, removes quotes
around French words, fixes stray characters, drops clips containing digits) and writes
speaker-disjoint splits to `sample-data/splits/{train,val,test}.csv` (about 80/10/10 by hours).
Re-run it after downloading more audio.

## 4. Notebooks (in `kanbior-asr/`)

| Notebook | What it does |
|---|---|
| `whisper.ipynb` | Plain Whisper-small on a Fon clip (baseline: it doesn't know Fon) |
| `fon-asr-v6.ipynb` | Runs [finanfa-fon-asr-v6](https://huggingface.co/finanfa-tech/finanfa-fon-asr-v6), plays the clip, compares with the reference transcript |
| `01-first_finetuning.ipynb` | Fine-tunes Whisper on the splits; reports WER/CER with and without tones |
| `02-kaggle_full_finetune.ipynb` | Kaggle: whisper-small on all 8 JML archives (~134 h), resumable across sessions via a private HF repo, compared with finanfa-v6 on the same test set. Setup steps are in its first cell |

In `01-first_finetuning.ipynb`, set `RUN` in the configuration cell:

- `"smoke"`: whisper-tiny, 200 clips, 100 steps. Checks the pipeline on a laptop.
  Needs ~2 GB free disk; close other notebook kernels first (8 GB RAM is tight).
- `"full"`: whisper-small, all clips, 4,000 steps. Run on a GPU (e.g. Kaggle T4).

Checkpoints and the final model are saved to `kanbior-asr/runs/` (git-ignored).

## License and attribution

The JML data is CC-BY-SA-4.0: credit J'aime Ma Langue (DOI 10.57967/hf/11698) and keep
adapted data under the same license.
