import torch
import torch.nn as nn

import beehive.modules as mod


def get_upconv_block_layers(
    in_channels: int = 8,
    out_channels: int = 16,
    n_conv: int = 1,
    bilinear: bool = True,
):
    block = mod.UpConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        n_conv=n_conv,
        bilinear=bilinear,
    )
    return list(block.children())[0], block


def test_upconv_block():
    layers, _ = get_upconv_block_layers()
    assert len(layers) == 2

    layers, block = get_upconv_block_layers(n_conv=3, bilinear=False)
    assert len(layers) == 4
    assert isinstance(layers[-1], nn.ConvTranspose2d)

    inputs = torch.randn(2, 8, 16, 16)
    outs = block(inputs)

    assert outs.shape[-1] == 32


def test_res_block():
    res_block = mod.UpScaleResNet()

    inputs = torch.randn(2, 3, 256, 256)
    outs = res_block(inputs)
    assert outs.shape[-1] == 64


def test_keypoint_head():
    head = mod.KeypointHead(4, 16)
    layers = list(head.children())[0]
    assert len(layers) == 1

    head = mod.KeypointHead(4, 16, n_conv=2)
    layers = list(head.children())[0]
    assert len(layers) == 2
