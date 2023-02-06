from typing import List, Optional

import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from torch.utils.data import DataLoader

from beehive.dataset import BeehiveDataset, pad_collate_fn
from beehive.model import CenterNet


def train_model(
    # * dataset params
    data_dir: str,
    split_file: str,
    batch_size: Optional[int] = 8,
    # * model params
    n_classes: Optional[int] = 1,
    n_conv: Optional[int] = 2,
    head_convs: Optional[int] = 2,
    upsample_channels: Optional[List[int]] = [256, 128, 64],
    bilinear: Optional[bool] = True,
    backbone: Optional[str] = "r18",
    pretrained: Optional[bool] = True,
    # * training params
    lr: Optional[float] = 5e-4,
    lamda_kp: Optional[float] = 1.0,
    lamda_offset: Optional[float] = 1.0,
    alpha_kp: Optional[int] = 2,
    beta_kp: Optional[int] = 2,
    steps: Optional[int] = 10,
):
    """creates and trains a model.

    Args:
        data_dir (str): data directort containing the gt-dots and img folder.
        split_file (str): path to json file with train-val-test splits.
        batch_size (Optional[int], optional): batch size. Defaults to 8.
        n_classes (Optional[int], optional): number of types of objects. Defaults to 1.
        n_conv (Optional[int], optional): number of convs in each upsample block. Defaults to 2.
        head_convs (Optional[int], optional): number of convs in keypoint head. Defaults to 2.
        upsample_channels (Optional[List[int]], optional): upsample out channels. Defaults to [256, 128, 64].
        bilinear (Optional[bool], optional): if true, use bilinear interpolation. Defaults to True.
        backbone (Optional[str], optional): backbone to use. options: [r18, r34]. Defaults to "r18".
        pretrained (Optional[bool], optional): if true, use pretrained backbone. Defaults to True.
        lr (Optional[float], optional): lr for training. Defaults to 5e-4.
        lamda_kp (Optional[float], optional): keypoint loss weight. Defaults to 1.0.
        lamda_offset (Optional[float], optional): offset loss weight. Defaults to 1.0.
        alpha_kp (Optional[int], optional): focal loss alpha. Defaults to 2.
        beta_kp (Optional[int], optional): focal loss beta. Defaults to 2.
        steps (Optional[int], optional): max number of steps to train for. Defaults to 10.
    """
    pl.seed_everything(42)
    logger.info("Seed set...")
    logger.info("Creating model...")
    model = CenterNet(
        n_classes=n_classes,
        n_conv=n_conv,
        head_convs=head_convs,
        upsample_channels=upsample_channels,
        bilinear=bilinear,
        backbone=backbone,
        pretrained=pretrained,
        lr=lr,
        lamda_kp=lamda_kp,
        lamda_offset=lamda_offset,
        alpha_kp=alpha_kp,
        beta_kp=beta_kp,
    )

    logger.info("Creating training and validation datasets...")
    train_dataset = BeehiveDataset(
        data_dir=data_dir, split_json=split_file, split="train"
    )

    val_dataset = BeehiveDataset(
        data_dir=data_dir, split_json=split_file, split="val"
    )

    logger.info("Creating dataloaders for the datasets...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=pad_collate_fn,
        drop_last=False,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=pad_collate_fn,
        drop_last=False,
    )

    # early_stop_callback = EarlyStopping(
    #     monitor="val-loss",
    #     min_delta=1e-3,
    #     patience=5,
    #     verbose=False,
    #     mode="min",
    #     strict=True,
    # )

    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch")
    trainer = pl.Trainer(
        callbacks=[lr_monitor_callback],
        max_steps=steps,
        log_every_n_steps=5,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
    )

    logger.info("Starting model training...")
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    logger.info("Starting model training... Done.")
