"""Run Piper training (piper1-gpl) with kanbior's checkpoint callbacks.

Same command line as `python -m piper.train`, e.g.:
    python train_piper.py fit --data.csv_path metadata.csv ... --model.mos_metric none

Why not `python -m piper.train` directly: its defaults add a checkpoint callback that monitors
"val_mos". With --model.mos_metric none that key is never logged and Lightning stops with an error.
LightningCLI always appends its default callbacks to the ones given in a config file, so they cannot
be replaced from the command line. Here they are, plus last.ckpt (rewritten every epoch):
  - default (Mac runs): best checkpoint by val_mel;
  - PIPER_CKPT_EVERY=N (Kaggle runs): a checkpoint every N epochs, the last PIPER_CKPT_KEEP (default 3)
    kept. val_mel stops improving early in VITS while the audio keeps getting cleaner, so the final model
    is picked among these by ASR scores and listening, not by val_mel.
"""

import logging
import os

import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from piper.train.__main__ import VitsLightningCLI
from piper.train.vits.dataset import VitsDataModule
from piper.train.vits.lightning import VitsModel


def callbacks() -> list:
    every = int(os.environ.get("PIPER_CKPT_EVERY", "0"))
    if every > 0:
        # Every N epochs; "step" only orders them, so the newest PIPER_CKPT_KEEP are kept.
        main_ckpt = ModelCheckpoint(
            monitor="step",
            mode="max",
            every_n_epochs=every,
            save_top_k=int(os.environ.get("PIPER_CKPT_KEEP", "3")),
            filename="epoch={epoch:04d}",
            auto_insert_metric_name=False,
        )
    else:
        # Best by mel reconstruction loss. Only 1 kept: each checkpoint is ~0.85 GB with optimizer state.
        main_ckpt = ModelCheckpoint(
            monitor="val_mel",
            mode="min",
            save_top_k=1,
            filename="best-epoch={epoch}-val_mel={val_mel:.4f}",
            auto_insert_metric_name=False,
        )
    return [
        main_ckpt,
        # last.ckpt, rewritten every epoch (Piper's own recipe for this).
        ModelCheckpoint(
            monitor=None,
            save_top_k=1,
            every_n_epochs=1,
            filename="last",
            enable_version_counter=False,
        ),
    ]


def main(args=None, run=True):
    logging.basicConfig(level=logging.INFO)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    return VitsLightningCLI(
        VitsModel,
        VitsDataModule,
        trainer_defaults={"max_epochs": -1, "callbacks": callbacks()},
        args=args,
        run=run,
    )


if __name__ == "__main__":
    main()
