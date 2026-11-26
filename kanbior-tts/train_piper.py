"""Run Piper training (piper1-gpl) with kanbior's checkpoint callbacks.

Same command line as `python -m piper.train`, e.g.:
    python train_piper.py fit --data.csv_path metadata.csv ... --model.mos_metric none

Why not `python -m piper.train` directly: its defaults add a checkpoint callback that monitors
"val_mos". With --model.mos_metric none that key is never logged and Lightning stops with an error.
LightningCLI always appends its default callbacks to the ones given in a config file, so they cannot
be replaced from the command line. Here they are: best checkpoint by val_mel, plus last.ckpt.
"""

import logging

import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from piper.train.__main__ import VitsLightningCLI
from piper.train.vits.dataset import VitsDataModule
from piper.train.vits.lightning import VitsModel


def callbacks() -> list:
    return [
        # Best by mel reconstruction loss. Only 1 kept: each checkpoint is ~0.85 GB with optimizer state.
        ModelCheckpoint(
            monitor="val_mel",
            mode="min",
            save_top_k=1,
            filename="best-epoch={epoch}-val_mel={val_mel:.4f}",
            auto_insert_metric_name=False,
        ),
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
