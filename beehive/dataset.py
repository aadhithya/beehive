import json
import os
import pdb

import albumentations as Alb
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from skimage import io
from torch.utils.data import Dataset

from beehive.gaussian_utils import draw_msra_gaussian, draw_umich_gaussian


class BeehiveDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split_json: str,
        split: str = "train",
        do_transform: bool = True,
    ) -> None:
        super().__init__()
        self.img_dir = os.path.join(data_dir, "img")
        self.lbl_dir = os.path.join(data_dir, "gt-dots")

        self.img_ids = self.__load_split_ids(split_json, split)

        self.lbl_prefix = "dots"
        self.img_prefix = "beeType1_"

        """
        * TRANSFORMS
        * 1. RandomFlip
        * 2. RandomScaling
        * 3. RandomCrop
        * 4. ColorJitter
        * 5. ToTensor
        """
        if do_transform:
            self.transform = Alb.Compose(
                [
                    # Alb.GaussianBlur(blur_limit=(1, 3)),
                    Alb.RandomScale(scale_limit=[-0.4, 0.4]),
                    Alb.RandomCrop(height=256, width=256),
                    Alb.Flip(),
                    Alb.Normalize(),
                    ToTensorV2(),
                ],
                keypoint_params=Alb.KeypointParams(format="xy"),
            )
        else:
            self.transform = Alb.Compose([Alb.Normalize(), ToTensorV2()])
        self.down_stride = 4

    def __load_split_ids(self, split_json: str, split: str):
        with open(split_json, "r") as f:
            img_ids = json.load(f)
        return img_ids[split]

    def __load_image(self, idx: int, is_lbl: bool = False):
        idx = self.img_ids[idx]
        if is_lbl:
            img_path = os.path.join(
                self.lbl_dir, f"{self.lbl_prefix}{idx}.png"
            )
        else:
            img_path = os.path.join(
                self.img_dir, f"{self.img_prefix}{idx}.jpg"
            )

        return io.imread(img_path)

    def __load_keypoints(self, idx: str):
        img = self.__load_image(idx, True)
        keypoints = list(zip(*img.nonzero()[::-1]))
        return np.array(keypoints)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img = self.__load_image(index)
        centers = self.__load_keypoints(index)

        tf = self.transform(image=img, keypoints=centers)

        img = tf["image"]
        centers = np.array(tf["keypoints"])

        lbl_h, lbl_w = (
            img.shape[-2] // self.down_stride,
            img.shape[-1] // self.down_stride,
        )

        heat_map = np.zeros([1, lbl_h, lbl_w], dtype=np.float32)
        # * centers_ds --> p/R
        centers_ds = centers / self.down_stride
        # * centers_ds_int --> floor(p/R) --> \tilde(p)
        centers_ds_int = centers_ds.astype(np.int32)

        if len(centers_ds_int) > 0:
            centers_ds_int[:, 0] = np.clip(centers_ds_int[:, 0], 0, lbl_h - 1)
            centers_ds_int[:, 1] = np.clip(centers_ds_int[:, 1], 0, lbl_w - 1)

            for ix in range(len(centers_ds_int)):
                heat_map[0] = draw_umich_gaussian(
                    heat_map[0], centers_ds_int[ix], radius=2
                )
                # heat_map = draw_msra_gaussian(heat_map, centers_ds[ix][::-1], 0.1)

        heat_map = torch.from_numpy(heat_map)
        return img, heat_map, centers_ds


def pad_collate_fn(data):
    img, heatmap, centers = zip(*data)
    img = torch.stack(img)
    heatmap = torch.stack(heatmap)

    cen_lens = [len(cen) for cen in centers]
    max_len = max(cen_lens)

    batch_size = len(heatmap)

    new_centers = torch.ones(batch_size, max_len, 2) * -1

    for ix, cen in enumerate(centers):
        if len(cen):
            new_centers[ix, : len(cen)] = torch.from_numpy(cen)

    return img, heatmap, new_centers
