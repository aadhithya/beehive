from typing import List, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchvision.utils import draw_segmentation_masks, make_grid

from beehive.loss import ModifiedFocalLoss, OffsetLoss
from beehive.modules import KeypointHead, UpScaleResNet
from beehive.postprocess import postprocess_preds


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
        lamda_kp: Optional[int] = 1,
        lamda_offset: Optional[int] = 1,
        alpha_kp: Optional[int] = 2,
        beta_kp: Optional[int] = 4,
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
            in_channels=upsample_channels[-1],
            n_conv=head_convs,
        )

        self.lr = lr
        self.lamda_kp = lamda_kp
        self.lamda_offset = lamda_offset

        self.focal_loss = ModifiedFocalLoss(alpha=alpha_kp, beta=beta_kp)
        self.offset_loss = OffsetLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=1000, gamma=0.25
        )
        return [optimizer], [scheduler]

    def forward(self, imgs: torch.Tensor, as_probs: Optional[bool] = True):
        out = self.backbone(imgs)
        out = self.keypoint_head(out)
        kp_outs = torch.sigmoid(out[:, :-2]) if as_probs else out[:, :-2]
        offsets = out[:, -2:]
        return kp_outs, offsets

    def training_step(self, batch, batch_idx):
        imgs, heatmaps, centers = batch
        kp_logits, offsets_preds = self(imgs)

        keypoint_loss = self.focal_loss(heatmaps, kp_logits)
        offset_loss = self.offset_loss(centers, offsets_preds)

        loss = (self.lamda_kp * keypoint_loss) + (
            self.lamda_offset * offset_loss
        )

        self.log("train-loss-kp", keypoint_loss)
        self.log("train-loss-od", offset_loss)
        self.log("train-loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, heatmaps, centers = batch
        kp_logits, offsets_preds = self(imgs)

        keypoint_loss = self.focal_loss(heatmaps, kp_logits)
        offset_loss = self.offset_loss(centers, offsets_preds)

        loss = (self.lamda_kp * keypoint_loss) + (
            self.lamda_offset * offset_loss
        )

        self.log("val-loss-kp", keypoint_loss)
        self.log("val-loss-od", offset_loss)
        self.log("val-loss", loss)

        return loss, imgs, heatmaps, kp_logits, offsets_preds

    def validation_epoch_end(self, outputs) -> None:
        self.visualize_images(outputs=outputs, mode="val")
        return super().validation_epoch_end(outputs)

    def visualize_images(self, outputs, mode="train"):
        _, imgs, gt_hm, kp_logits, offsets = outputs[-1]
        n_imgs = min(8, imgs.shape[0])

        gt_hm = F.interpolate(gt_hm, scale_factor=4, mode="bilinear").cpu()

        # preds = F.interpolate(kp_logits, scale_factor=4, mode="bilinear").cpu()

        preds, _ = postprocess_preds(kp_logits.cpu(), offsets.cpu())
        gt_masks = (gt_hm[:n_imgs] > 0).float()
        pred_masks = (preds[:n_imgs]).float()

        img_grid = make_grid(imgs[:n_imgs], n_imgs, normalize=True).cpu()
        img_grid = (img_grid * 255).to(torch.uint8)

        gt_mask_grid = make_grid(gt_masks, n_imgs, normalize=False).to(
            torch.bool
        )

        pred_mask_grid = make_grid(pred_masks, n_imgs, normalize=False).to(
            torch.bool
        )

        gt_overlay = draw_segmentation_masks(
            img_grid, gt_mask_grid, alpha=0.5, colors="red"
        )
        pred_overlay = draw_segmentation_masks(
            img_grid, pred_mask_grid, alpha=0.7, colors="red"
        )

        viz_imgs = make_grid(
            torch.stack((img_grid, gt_overlay, pred_overlay)), nrow=1
        )

        self.logger.experiment.add_image(
            f"{mode}-viz", viz_imgs, self.current_epoch
        )
