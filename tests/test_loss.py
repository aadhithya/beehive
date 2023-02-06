from beehive.loss import ModifiedFocalLoss, OffsetLoss


def test_offset_loss(centers, offsets):
    loss_fn = OffsetLoss()
    loss = loss_fn(centers, offsets)
    assert loss.item() - 1.2 < 1e-14


def test_focal_loss(hmap, hmap_pred):
    hmap_h, hmap_l = hmap_pred
    loss_fn = ModifiedFocalLoss(1, 1)
    loss_l = loss_fn(hmap, hmap_l)
    loss_h = loss_fn(hmap, hmap_h)

    assert loss_h < loss_l
