import numpy as np
import torch
from torch import Tensor

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class CutMix(torch.nn.Module):
    """ CutMix based on "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    adpoted from https://github.com/clovaai/CutMix-PyTorch/blob/2d8eb68faff7fe4962776ad51d175c3b01a25734/train.py#L279
    """
    def __init__(
        self,
        beta: int=1,
        cutmix_prob: float=0.3,
        device: str='cpu',

    ) -> None:
        super().__init__()
        self.beta = beta
        self.cutmix_prob = cutmix_prob
        self.device = device

    def forward(self, inputs: Tensor, labels: Tensor) -> tuple:
        """
            img (PIL Image or Tensor): Image to be transformed.
            label (PIL Image or Tensor): Label to be transformed.
        Returns:
            PIL Image or Tensor: Transformed image & label.
        """
        r = np.random.rand(1)
        if self.beta > 0 and r < self.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = torch.randperm(inputs.size()[0]).to(self.device)
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            labels[:, bbx1:bbx2, bby1:bby2] = labels[rand_index, bbx1:bbx2, bby1:bby2]
        return inputs, labels