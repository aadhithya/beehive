import os
import pdb
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import List, Tuple

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from beehive.dataset import BeehiveDataset, pad_collate_fn
from beehive.inference import load_model
from beehive.postprocess import hmap_nms, postprocess_preds


def centers_to_bbox(
    centers: torch.Tensor, mask_size: Tuple[int, int], det_size: int = 15
):
    boxes = []
    sz = (det_size - 1) // 2
    for center in centers:
        nnz = len(center[:, 0] > 0)
        center = center[:nnz]
        # * xmin, ymin, xmax, ymax
        boxes += [
            torch.stack(
                [
                    center[:, 0] - sz,
                    center[:, 1] - sz,
                    center[:, 0] + sz,
                    center[:, 1] + sz,
                ],
                dim=-1,
            )
        ]
    boxes = torch.stack(boxes)

    for ix in range(4):
        boxes[..., ix] = boxes[..., ix].clamp(0, mask_size[(ix + 1) % 2])

    return boxes


def centers_to_mask(
    centers: torch.Tensor, mask_size: Tuple[int, int], det_size: int = 15
):
    # return BHW
    masks = []
    weights = torch.ones(1, 1, det_size, det_size)
    padding = [round((det_size - 1) / 2)] * 2
    for center in centers.long():
        mask = torch.zeros(*mask_size)
        mask[center[:, 1], center[:, 0]] = 1
        mask = F.conv2d(
            mask[None, None, ...],
            weight=weights,
            stride=1,
            padding=padding,
        ).squeeze()
        masks += [mask]
    return torch.stack(masks, 0)


@torch.no_grad()
def get_predictions(
    model: torch.nn.Module, data_dir: str, splits_path: str, thr: float = 0.0
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = BeehiveDataset(data_dir, splits_path, "test", False)
    test_dataloader = DataLoader(dataset, 1, False, collate_fn=pad_collate_fn)
    _ = model.eval().to(device)
    predictions = []
    targets = []
    imgs = []
    abs_err = []
    for img, _, gt_ct in test_dataloader:
        hm_pred, offsets = model(img.float().to(device))

        hm_pred = hm_pred.cpu()
        offsets = offsets.cpu()

        ct_pred = postprocess_preds(
            hm_pred, offsets, return_centers=True, threshold=thr, ks=3
        )

        pred = {"masks": [], "boxes": [], "scores": [], "labels": []}
        tgt = {"masks": [], "boxes": [], "labels": []}

        pred["masks"] += [
            centers_to_mask(ct_pred, img.shape[-2:], det_size=15).squeeze()
        ]
        tgt["masks"] += [
            centers_to_mask(gt_ct * 4, img.shape[-2:], det_size=15).squeeze()
        ]

        pred["boxes"] += [
            centers_to_bbox(ct_pred, img.shape[-2:], det_size=15).squeeze()
        ]
        tgt["boxes"] += [
            centers_to_bbox(gt_ct * 4, img.shape[-2:], det_size=15).squeeze()
        ]

        scores = hmap_nms(hmap=hm_pred, kernel_size=3)
        pred["scores"] += [scores[scores > thr].squeeze()]

        pred["labels"] += [torch.ones(len(ct_pred[0]))]
        tgt["labels"] += [torch.ones(len(gt_ct[0]))]

        pred = {k: torch.cat(v, 0) for k, v in pred.items()}
        tgt = {k: torch.cat(v, 0) for k, v in tgt.items()}

        predictions += [pred]
        targets += [tgt]
        imgs += [img]
        abs_err += [len(gt_ct[0]) - (scores > 0.5).sum()]

    return predictions, targets, imgs, torch.Tensor(abs_err)


def run_eval(
    ckpt_path: str, data_dir: str, splits_path: str, out_path: str = None
):
    model = load_model(ckpt_path)
    predictions, targets, _, abs_err = get_predictions(
        model, data_dir, splits_path
    )

    mae = abs_err.mean()

    map_metric = MeanAveragePrecision()

    map_metric.update(predictions, targets)
    result = dict(map_metric.compute())

    pprint(result)
    print(f"mae: {mae}")

    if out_path is None:
        filename = Path(ckpt_path).stem
        parent_dir = Path(ckpt_path).parent.absolute()
        out_path = os.path.join(parent_dir, f"{filename}_mAP_eval.yml")

    result = {k: v.item() for k, v in result.items()}
    result["MAE"] = mae
    with open(out_path, "w") as f:
        yaml.dump(result, f)
