import pytest
import torch


@pytest.fixture
def offsets():
    return torch.eye(5)[None, None].repeat([1, 2, 1, 1])


@pytest.fixture
def hmap_pred():
    hmap_l = torch.zeros(1, 1, 5, 5)
    hmap_h = hmap_l.clone()
    diag = torch.arange(5)
    hmap_h[..., diag, diag] = torch.randint(700, 1000, (5,)) / 1000
    hmap_l[..., diag, diag] = torch.randint(100, 500, (5,)) / 1000
    return hmap_h, hmap_l


@pytest.fixture
def hmap():
    return torch.eye(5).unsqueeze(0).unsqueeze(0)


@pytest.fixture
def centers():
    return torch.Tensor([[[1.3, 1.5], [3.5, 3.3], [4.3, 4.5]]])
