from typing import List, Optional

import pytorch_lightning as pl
import torch

from beehive.modules import KeypointHead, UpScaleResNet


class CenterNet(pl.LightningModule):
    def __init__(
        self,
        n_classes: int,
        n_conv: Optional[int] = 2,
        head_convs: Optional[int] = 1,
        upsample_channels: Optional[List[int]] = [256, 128, 64],
        bilinear: Optional[bool] = True,
        backbone: Optional[str] = "r18",
        pretrained: Optional[bool] = True,
        lr: Optional[float] = 5e-4,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.backbone = UpScaleResNet(
            backbone=backbone,
            upsample_channels=upsample_channels,
            pretrained_backbone=pretrained,
            n_conv=n_conv,
            bilinear=bilinear,
        )

        self.keypoint_head = KeypointHead(
            n_classes=n_classes,
            in_channels=512,
            n_conv=head_convs,
        )

        self.lr = lr

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, imgs: torch.Tensor, as_probs: Optional[bool] = False):
        out = self.backbone(imgs)
        out = self.keypoint_head(out)
        return torch.sigmoid(out) if as_probs else out

    def training_step(self, batch, batch_idx):
        imgs, heat_maps = batch
        logits = self(imgs)

        keypoint_loss = self.focal_loss(heat_maps, logits)
        offshift_loss = self.loss(logits)

        loss = (self.lamda_kp * keypoint_loss) + (
            self.lamda_os * offshift_loss
        )

        self.log("train-loss-kp", keypoint_loss)
        self.log("train-loss-od", offshift_loss)
        self.log("train-loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, heat_maps = batch
        logits = self(imgs)

        keypoint_loss = self.focal_loss(heat_maps, logits)
        offshift_loss = self.loss(logits)

        loss = (self.lamda_kp * keypoint_loss) + (
            self.lamda_os * offshift_loss
        )

        self.log("val-loss-kp", keypoint_loss)
        self.log("val-loss-od", offshift_loss)
        self.log("val-loss", loss)
