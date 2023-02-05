import os
from typing import Optional, Tuple

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from loguru import logger
from PIL import Image
from skimage import io
from skimage.transform import rescale
from torchvision.utils import draw_segmentation_masks

from beehive.model import CenterNet
from beehive.postprocess import postprocess_preds


def show_result(img, mask, scale):
    if scale != 1:
        mask = rescale(mask, scale)
        logger.info(f"Mask Scale Factor: {scale}")
        logger.info(f"Mask size after scaling: {mask.shape}")

    mask = mask > 0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1)
    mask_tensor = torch.from_numpy(mask)
    viz_img = (
        draw_segmentation_masks(
            img_tensor,
            mask_tensor,
            alpha=0.5,
            colors="red",
        )
        .permute(1, 2, 0)
        .numpy()
    )

    viz_img = Image.fromarray(viz_img)
    viz_img.show()


def get_inference_transforms(img_size: Tuple[int, int, int], scale: int):
    h, w, _ = img_size

    if h * scale < 256 or w * scale < 256:
        logger.warning("Minimum Image Size after scaling should be 256x256")
        logger.warning(
            f"For scale={scale}, image size will be {h*scale}x{w*scale}."
        )
        scale = min(256 / h, 256 / w)
        logger.warning(f"Setting scale={scale}.")

    new_h, new_w = int(h * scale), int(w * scale)
    logger.info(f"Image size after scaling: {new_h}x{new_w}")

    transforms = A.Compose(
        [
            A.Normalize(),
            A.Resize(new_h, new_w, always_apply=True),
            ToTensorV2(),
        ]
    )

    logger.info(
        f"Inference Transforms: Normalize -> Resize({new_h}, {new_w}) -> ToTensor"
    )

    return transforms, 1 / scale


def run_inference(
    img_path: str,
    ckpt_path: Optional[str] = "./ckpt/v38.ckpt",
    scale: Optional[float] = 1.0,
    show: Optional[bool] = True,
    v: Optional[bool] = True,
):
    if not v:
        logger.disable("beehive")
    assert os.path.exists(img_path), "Image does not exist!"
    img = io.imread(img_path)
    logger.info(f"Loaded Image: {img_path}")
    logger.info(f"image shape: {img.shape}")

    model = CenterNet.load_from_checkpoint(checkpoint_path=ckpt_path)
    _ = model.eval()
    logger.info(f"Loaded model from ckpt: {ckpt_path}")

    transforms, scale = get_inference_transforms(img.shape, scale)
    with torch.no_grad():
        input_img = transforms(image=img)["image"]
        hmap_pred, offsets = model(input_img.unsqueeze(0).to(model.device))

    mask, n_dets = postprocess_preds(hmap_pred.cpu(), offsets.cpu())

    print(
        f"\033[94m\033[1mNumber of bees in the image: {int(n_dets[0])}\033[0m"
    )

    if show:
        show_result(img, mask.squeeze().numpy(), scale)

    return


def export_onnx(ckpt_path: str, out_path: str):
    model = CenterNet.load_from_checkpoint(checkpoint_path=ckpt_path)
    _ = model.eval()
    logger.info(f"Loaded model from ckpt: {ckpt_path}")

    input_image = torch.randn(1, 3, 256, 256)
    dynamic_axes = {"img": [2, 3], "hmap": [2, 3], "offsets": [2, 3]}

    model.to_onnx(
        out_path,
        input_image,
        input_names=["img"],
        output_names=["hmap", "offsets"],
        dynamic_axes=dynamic_axes,
    )

    logger.success(f"Model Exported to Onnx: {out_path}")
