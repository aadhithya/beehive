import os
import urllib.request
from typing import Optional, Tuple

import albumentations as A
import matplotlib.pyplot as plt
import torch
from albumentations.pytorch import ToTensorV2
from loguru import logger
from skimage import io
from skimage.transform import rescale
from torchvision.utils import draw_segmentation_masks

from beehive.model import CenterNet
from beehive.postprocess import postprocess_preds

TRCH_MODEL_URL = "https://github.com/aadhithya/beehive/releases/download/weights/centernet-bees.ckpt"
MODEL_DL_PATH = "./ckpt/centernet-bees.ckpt"


def show_result(
    img: torch.Tensor, mask: torch.Tensor, scale: int, n_det: int
):
    """Displays detection result.

    Args:
        img (torch.Tensor): image
        mask (torch.Tensor): mask
        scale (int): scale factor
        n_det (int): number of detections
    """
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

    plt.imshow(viz_img)
    plt.title(f"Bee Detections: {int(n_det)} Bees")
    plt.axis("off")
    plt.show()


def get_inference_transforms(
    img_size: Tuple[int, int, int], scale: int
) -> A.Compose:
    """returns transforms for inference.

    Args:
        img_size (Tuple[int, int, int]): image size HxWxC format.
        scale (int): scale factor.

    Returns:
        A.Compose: transforms for inference.
    """
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


def load_model(ckpt_path: str, dl: bool = False) -> torch.nn.Module:
    """loads model from checkpoint. Downloads checkpoint if set.

    Args:
        ckpt_path (str): checkpoint path.
        dl (bool, optional): if true, downloads model if not available locally. Defaults to False.

    Raises:
        FileNotFoundError: if file is not found.

    Returns:
        torch.nn.Module: model for inference.
    """
    if not os.path.exists(ckpt_path):
        logger.warning(f"Checkpoint does not exist in {ckpt_path}.")
        if dl:
            if os.path.exists(MODEL_DL_PATH):
                logger.info(f"Model Cache found: {MODEL_DL_PATH}")
            else:
                logger.info(f"Downloading model from: {TRCH_MODEL_URL}")
                os.makedirs("./ckpt/", exist_ok=True)
                urllib.request.urlretrieve(
                    TRCH_MODEL_URL, filename=MODEL_DL_PATH
                )
                logger.info(f"Downloaded to: {MODEL_DL_PATH}")
            ckpt_path = MODEL_DL_PATH
        else:
            logger.info("You can download model by passing the -dl flag.")
            raise FileNotFoundError(ckpt_path)
    model = CenterNet.load_from_checkpoint(checkpoint_path=ckpt_path)
    _ = model.eval()
    logger.info(f"Loaded model from ckpt: {ckpt_path}")
    return model


def run_inference(
    img_path: str,
    model: torch.nn.Module,
    scale: Optional[float] = 1.0,
    show: Optional[bool] = False,
    v: Optional[bool] = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """runs inference on the image and returns mask and num detections.

    Args:
        img_path (str): image path.
        model (torch.nn.Module): model.
        scale (Optional[float], optional): scale factor. Defaults to 1.0.
        show (Optional[bool], optional): if true, displays detections. Defaults to False.
        v (Optional[bool], optional): if true, prints output. Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: mask and detections.
    """
    assert os.path.exists(img_path), "Image does not exist!"
    img = io.imread(img_path)
    logger.info(f"Loaded Image: {img_path}")
    logger.info(f"image shape: {img.shape}")

    transforms, scale = get_inference_transforms(img.shape, scale)
    with torch.no_grad():
        input_img = transforms(image=img)["image"]
        hmap_pred, offsets = model(input_img.unsqueeze(0).to(model.device))

    mask, n_dets = postprocess_preds(hmap_pred.cpu(), offsets.cpu())
    if v:
        print(
            f"\033[94m\033[1mNumber of bees in the image: {int(n_dets[0])}\033[0m"
        )
    if show:
        show_result(img, mask.squeeze().numpy(), scale, n_dets[0])
    return mask, n_dets


def export_onnx(ckpt_path: str, out_path: str):
    """exports model to onnx

    Args:
        ckpt_path (str): checkpoint path
        out_path (str): output path
    """
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
