import pdb

import torch
import torch.nn.functional as F


class ModifiedFocalLoss:
    def __init__(self, alpha=2, beta=4):
        """Modified Focal Loss from the paper Objects as Points.

        Args:
            alpha (int, optional): alpha as specified in the paper.. Defaults to 2.
            beta (int, optional): beta as specified in the paper. Defaults to 4.
        """
        self.alpha = alpha
        self.beta = beta

    def __call__(self, target, preds) -> torch.Tensor:
        pos_idxs = (target == 1).float()
        neg_idxs = (target < 1).float()

        preds = torch.clamp(preds, 1e-14)

        preds_compl = 1 - preds

        pos_loss = (
            torch.pow(preds_compl, self.alpha) * torch.log(preds) * pos_idxs
        ).sum()

        neg_loss = (
            torch.pow((1 - target), self.beta)
            * torch.pow(preds, self.alpha)
            * torch.log(preds_compl)
            * neg_idxs
        ).sum()

        if pos_idxs.sum():
            loss = -(pos_loss + neg_loss) / pos_idxs.sum()
        else:
            loss = -neg_loss
        return loss


class OffsetLoss:
    """
    Offset loss is applied between offsets predicted by the net
    and the offset computed in the ground truth.
    """

    def __call__(self, centers, preds) -> torch.Tensor:
        # * centers --> p/R
        # * get p_tilde = floor(p/R)
        # * shape(centers) = shape(p_tilde) = [N, X, 2]
        # * X is the padded length
        p_tilde = torch.floor(centers).long()

        mask = p_tilde >= 0

        offset = centers - p_tilde.float()

        o_hats = torch.stack(
            [
                pred[1:, p_tilde[ix, :, 1], p_tilde[ix, :, 0]]
                for ix, pred in enumerate(preds)
            ]
        )
        o_hats = o_hats.permute((0, 2, 1))

        loss = F.l1_loss(o_hats, offset, reduction="none")
        loss = (loss * mask).sum()

        if mask.sum():
            # * we divide mask sum by 2 because it is active for both x and
            # * y coords. we just need number of center points in batch.
            loss = loss / (mask.sum() / 2)

        return loss
