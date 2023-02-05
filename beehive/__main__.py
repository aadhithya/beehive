from typing import List, Optional

from loguru import logger
from typer import Argument, Option, Typer

from beehive.train import train_model

app = Typer(name="beehive")


@app.command("train", help="Train a model.")
def train(
    # * dataset params
    data_dir: str = Argument(..., help="path ot data dir."),
    split_file: str = Argument(..., help="path to data split json."),
    batch_size: Optional[int] = Option(8, help="batch size to use."),
    epochs: Optional[int] = Option(
        10, help="Max number of epochs to train for."
    ),
    # * model params
    n_classes: Optional[int] = Option(1, help="number of object classes."),
    n_conv: Optional[int] = Option(
        2, help="number of convolution layers per upsample block."
    ),
    head_convs: Optional[int] = Option(
        2, help="number of convolution layers in head."
    ),
    upsample_channels: Optional[List[int]] = Option(
        [256, 128, 64], help="output channels in each upsample block."
    ),
    bilinear: Optional[bool] = Option(
        True, help="If true, uses bilinear interpolation for upsampling."
    ),
    backbone: Optional[str] = Option(
        "r18", help="Backbone model to use. Options: [r18, r34, r50]."
    ),
    pretrained: Optional[bool] = Option(
        True, help="if true, uses pretrained resnet backbone."
    ),
    # * training params
    lr: Optional[float] = Option(
        5e-4, help="Learning rate to use for training."
    ),
    lamda_kp: Optional[float] = Option(1.0, help="Weight for keypoint loss."),
    lamda_offset: Optional[float] = Option(
        1.0, help="Weight for offset loss."
    ),
    alpha_kp: Optional[int] = Option(
        2, help="alpha parameter for focal loss."
    ),
    beta_kp: Optional[int] = Option(4, help="beta param for focal loss"),
):
    train_model(
        data_dir,
        split_file,
        batch_size,
        n_classes,
        n_conv,
        head_convs,
        upsample_channels,
        bilinear,
        backbone,
        pretrained,
        lr,
        lamda_kp,
        lamda_offset,
        alpha_kp,
        beta_kp,
        epochs,
    )


@app.command("eval", help="Evaluate a trained model.")
def eval():
    pass


@app.command("infer", help="Run inference using a trained model.")
def infer():
    pass


if __name__ == "__main__":
    app()
