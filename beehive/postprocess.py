import pdb
from typing import Optional

import torch
import torch.nn.functional as F


def hmap_nms(hmap, kernel_size: int = 3):
    """Finds and keeps only the peaks in the heatmap.

    Args:
        hmap (torch.Tensor): heatmap
        kernel_size (int, optional): kernelsize to use. Defaults to 3.

    Returns:
        torch.Tensor
    """
    padding = (kernel_size - 1) // 2
    hmax = F.max_pool2d(
        hmap, kernel_size=kernel_size, stride=1, padding=padding
    )

    keep = (hmax == hmap).float()

    return hmap * keep


def postprocess_preds(hmap, offsets, threshold=0.5, scale=4):
    b, c, h, w = hmap.shape
    scores = hmap_nms(hmap)
    pred_points = scores > threshold

    dilated_outs = []
    n_dets = []
    for ix in range(len(scores)):
        hmap_pred = torch.zeros(1, c, h * scale, w * scale)
        points = pred_points[ix, 0].nonzero()
        off_x = offsets[ix, 0][points[:, 0], points[:, 1]]
        off_y = offsets[ix, 1][points[:, 0], points[:, 1]]
        new_points = (points + torch.stack((off_x, off_y), -1)) * 4
        new_points = new_points.round().long()
        hmap_pred[..., new_points[:, 0], new_points[:, 1]] = 1

        dilated_pred = F.conv2d(
            hmap_pred, weight=torch.ones(1, 1, 7, 7), padding=(3, 3)
        )
        n_dets += [len(new_points)]
        dilated_outs += [dilated_pred]
    return torch.cat(dilated_outs, 0), torch.Tensor(n_dets)
