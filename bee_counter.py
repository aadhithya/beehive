"""
minimal inference script independent
of the beehive project. This is done
to avoid installing dependencies not
needed for inference. Uses ONNXruntime
for ingerence.

To run  inference without installing beehive,
1. Install requirements: `pip install -r minimal_requirements.txt`
2. python infer.py --help
"""
import argparse
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from numpy.lib.stride_tricks import as_strided
from scipy import ndimage as ndi
from skimage import io


class BeeCounter:
    def __init__(self, ckpt_path: str = "./ckpt/export/v38.onnx") -> None:
        self.session = ort.InferenceSession(
            ckpt_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    # @staticmethod
    def pool2d(self, A, kernel_size, stride, padding=0, pool_mode="max"):
        """
        2D Pooling
        Source: https://stackoverflow.com/a/54966908

        Parameters:
            A: input 2D array
            kernel_size: int, the size of the window over which we take pool
            stride: int, the stride of the window
            padding: int, implicit zero paddings on both sides of the input
            pool_mode: string, 'max' or 'avg'
        """
        # Padding
        A = np.pad(A, padding, mode="constant")

        # Window view of A
        output_shape = (
            (A.shape[0] - kernel_size) // stride + 1,
            (A.shape[1] - kernel_size) // stride + 1,
        )

        shape_w = (output_shape[0], output_shape[1], kernel_size, kernel_size)
        strides_w = (
            stride * A.strides[0],
            stride * A.strides[1],
            A.strides[0],
            A.strides[1],
        )

        A_w = as_strided(A, shape_w, strides_w)

        # Return the result of pooling
        if pool_mode == "max":
            return A_w.max(axis=(2, 3))
        elif pool_mode == "avg":
            return A_w.mean(axis=(2, 3))

    def _prepare_inputs(self, img) -> Dict[str, np.array]:
        input_name = self.session.get_inputs()[0].name

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        img = img / 255
        normalized_img = (img - mean) / std

        # * [HWC] --> [NHWC] --> [NCHW]
        input_img = np.transpose(
            normalized_img[np.newaxis, ...], (0, 3, 1, 2)
        )

        return {input_name: input_img.astype(np.float32)}

    def _postprocess(self, hmap, offsets, threshold):
        _, _, h, w = hmap.shape
        hmax = self.pool2d(np.squeeze(hmap), 3, 1, 1)[
            None,
            None,
        ]
        keep = (hmax == hmap).astype(np.float32)
        scores = keep * hmap

        pred_points = scores > threshold

        dilated_outs = []
        n_dets = []
        for ix in range(len(scores)):
            hmap_pred = np.zeros((1, 1, h * 4, w * 4))
            points = pred_points[ix, 0].nonzero()
            points = np.array(list(zip(*points)))
            off_x = offsets[ix, 0][points[:, 0], points[:, 1]]
            off_y = offsets[ix, 1][points[:, 0], points[:, 1]]
            new_points = (points + np.stack((off_x, off_y), -1)) * 4
            new_points = new_points.round().astype(np.int32)
            hmap_pred[..., new_points[:, 0], new_points[:, 1]] = 1

            dilated_pred = ndi.binary_dilation(
                hmap_pred,
                structure=np.ones((1, 1, 7, 7)),
            )
            n_dets += [len(new_points)]
            dilated_outs += [dilated_pred]
        return np.concatenate(dilated_outs, 0), np.array(n_dets)

    def __call__(self, img_path: str):
        assert os.path.exists(
            img_path
        ), f"Image not found! Ensure {img_path} exists!"

        img = io.imread(img_path)
        model_inputs = self._prepare_inputs(img)

        hmap, offsets = self.session.run(["hmap", "offsets"], model_inputs)

        mask, n_dets = self._postprocess(hmap, offsets, threshold=0.5)
        mask = np.squeeze(mask)
        n_dets = n_dets[0]

        return mask, n_dets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", help="path to image")
    parser.add_argument(
        "--ckpt_path",
        default="./ckpt/export/v38.onnx",
        help="onnx model path",
    )
    parser.add_argument("--show", type=bool, help="if set, display output.")

    args = parser.parse_args()

    bc = BeeCounter(args.ckpt_path)
    mask, n_dets = bc(args.img_path)
    print(f"\033[94m\033[1mNumber of bees in the image: {int(n_dets)}\033[0m")

    if args.show:
        img = io.imread(args.img_path)
        det_mask = np.ma.masked_where(mask < 1, mask)
        plt.imshow(img)
        plt.imshow(det_mask, vmin=0, vmax=1, alpha=0.7)
        plt.title(f"nDetections: {n_dets}")
        plt.axis("off")
        plt.show()
