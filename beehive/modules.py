from typing import List, Optional, Union

import torch
import torch.nn as nn
from loguru import logger
from torchvision.models import resnet18, resnet34, resnet50


def conv_bn_relu(
    in_channels: int,
    out_channels: int,
    kernel_size: Optional[int] = 3,
    padding: Optional[int] = 1,
    stride: Optional[int] = 1,
    bias: Optional[bool] = True,
) -> nn.Sequential:
    """creates a block with ops: CONV -> BN -> ReLU.

    Args:
        in_channels (int): num. input channels.
        out_channels (int): num output channels.
        kernel_size (Optional[int], optional): conv kernel size. Defaults to 3.
        padding (Optional[int], optional): conv padding. Defaults to 1.
        stride (Optional[int], optional): conv stride. Defaults to 1.
        bias (Optional[bool], optional): if true, enable conv bias. Defaults to True.

    Returns:
        nn.Sequential: CONV -> BN -> ReLU block.
    """
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True),
    )


class UpConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_conv: Optional[int] = 3,
        bilinear: Optional[bool] = True,
    ) -> None:
        """Block that upsamples input by 2x.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            n_conv (Optional[int], optional): number of conv layers per block. Defaults to 3.
            bilinear (Optional[bool], optional): if True uses bilinear interpolation, else Transpose Conv. Defaults to True.
        """
        super().__init__()

        if n_conv <= 0:
            logger.warning(
                f"n_conv should be >= 1 and not {n_conv}. Setting n_conv=1"
            )
            n_conv = 1

        layers = []

        # * add initial conv layer.
        layers.append(
            conv_bn_relu(in_channels=in_channels, out_channels=out_channels)
        )

        # * add extra conv layers if needed.
        if n_conv > 1:
            layers += [
                conv_bn_relu(
                    in_channels=out_channels, out_channels=out_channels
                )
                for _ in range(n_conv - 1)
            ]

        # * add upsampling layer.
        layers.append(
            self.__build_upsample_layer(bilinear, out_channels, out_channels)
        )

        self.block = nn.Sequential(*layers)

    def __build_upsample_layer(
        self,
        bilinear: bool,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
    ) -> Union[nn.UpsamplingBilinear2d, nn.ConvTranspose2d]:
        if bilinear:
            return nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            # * if not bilinear, return 2x UpConv layer.
            return nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpScaleResNet(nn.Module):
    def __init__(
        self,
        backbone: Optional[str] = "r18",
        upsample_channels: Optional[List[int]] = [256, 128, 64],
        pretrained_backbone: Optional[bool] = True,
        n_conv: Optional[int] = 2,
        bilinear: Optional[bool] = True,
    ) -> None:
        """UpScaleResNet uses the resnet backbone and adds upsampling blocks to it.

        Args:
            backbone (Optional[str], optional): backbone to use [r18, r34, r50]. Defaults to "r18".
            upsample_channels (Optional[List[int]], optional): upsample layer channels. Defaults to [256, 128, 64].
            pretrained_backbone (Optional[bool], optional): if true, use pretrained resnet. Defaults to True.
            n_conv (Optional[int], optional): number of conv layers to use in upsample block. Defaults to 2.
            bilinear (Optional[bool], optional): if true uses bilinear interpolation. Defaults to True.
        """
        super().__init__()

        self.backbone = self.__load_resnet_backbone(
            backbone, pretrained_backbone
        )

        layers = []
        # * Last conv layer of resnet outputs 512 channels
        in_ch = 512

        for out_ch in upsample_channels:
            layers.append(
                UpConvBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    n_conv=n_conv,
                    bilinear=bilinear,
                )
            )
            in_ch = out_ch
        self.up_block = nn.Sequential(*layers)

    def __load_resnet_backbone(
        self, net: str, pretrained: bool
    ) -> nn.Sequential:
        if net == "r18":
            resnet = resnet18
        elif net == "r34":
            resnet = resnet34
        elif net == "r50":
            resnet = resnet50
        else:
            logger.error(
                "Only following resnet backbones supported: [r18, r34, r50]."
            )
            raise Exception(
                "Only following resnet backbones supported: [r18, r34, r50]."
            )

        backbone = resnet(weights="DEFAULT") if pretrained else resnet()

        # * remove the last avg pool and FC layer as we don't need them.
        backbone = nn.Sequential(*list(backbone.children())[:-2])

        return backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        return self.up_block(out)


class KeypointHead(nn.Module):
    def __init__(
        self, n_classes: int, in_channels: int, n_conv: Optional[int] = 1
    ) -> None:
        """KeypointHead produces the heat maps for the derections.

        Args:
            n_classes (int): number of types of objects to detect.
            in_channels (int): number of input channels.
            n_conv (Optional[int], optional): number of 1x1 convolutions tu use before final layer. Defaults to 1.
        """
        super().__init__()

        layers = []
        in_ch = in_channels
        out_ch = in_channels // 2
        if n_conv > 1:
            for _ in range(n_conv - 1):
                layers += [
                    conv_bn_relu(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=3,
                        padding=1,
                    )
                ]
                in_ch = out_ch
            in_ch = out_ch

        layers.append(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=n_classes + 2,
                kernel_size=1,
            )
        )

        self.head = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        as_probs: Optional[bool] = False,
    ) -> torch.Tensor:
        out = self.head(x)
        return torch.sigmoid(out) if as_probs else out
