import torch

from beehive.eval import centers_to_bbox, centers_to_mask


def test_centers_to_mask():
    inp_arr = torch.zeros(1, 5, 5)
    inp_arr[0, 2, 2] = 1
    centers = torch.tensor([[[2, 2]]])
    mask = centers_to_mask(centers=centers, mask_size=(5, 5), det_size=3)
    assert mask.sum() == 9
    assert torch.all(mask[0, 1:-1, 1:-1] == 1)
    assert mask.shape == inp_arr.shape


def test_centers_to_bbox():
    centers = torch.Tensor([[[2, 2], [3, 3], [4, 4]]])
    expected = torch.Tensor([[[1, 1, 3, 3], [2, 2, 4, 4], [3, 3, 5, 4]]])
    boxes = centers_to_bbox(centers=centers, mask_size=(4, 5), det_size=3)

    assert (expected - boxes).sum() == 0
