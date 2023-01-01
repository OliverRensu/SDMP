# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
from timm.data import Mixup
import torch.nn.functional as F
from imageio import imwrite
def rand_bbox(img_shape, lam, count=None):
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    cy = np.random.randint(0 + cut_h // 2, img_h - cut_h // 2, size=count)
    cx = np.random.randint(0 + cut_w // 2, img_w - cut_w // 2, size=count)
    yl = cy - cut_h // 2
    yh = cy + cut_h // 2
    xl = cx - cut_w // 2
    xh = cx + cut_w // 2
    return yl, yh, xl, xh, cut_h*cut_w


def datamixing(x, lam, rank,bz):
    if lam is None:
        lam = np.random.beta(1., 1., size=[x.shape[0],1,1,1]).astype(np.float32)
        lam = torch.from_numpy(lam).cuda()
    new_lams = lam * 0.8 + 0.1
    x_flip = x.flip(0).clone()
    for i in range(x.shape[0]):
        yl, yh, xl, xh, bbox_area = rand_bbox(x.shape, new_lams[i].cpu())
        new_lams[i] = torch.tensor(1. - bbox_area / float(x.shape[-2] * x.shape[-1])).cuda()
        if (i >= rank*bz) and (i<rank*bz):
            x[i:i+1, :, yl:yh, xl:xh] = F.interpolate(x_flip[i:i+1], (yh - yl, xh - xl), mode="bilinear")
    return x, new_lams

class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T

        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k, redunction):
        # normalize
        bz = q.shape[0]
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        rank = torch.distributed.get_rank()
        labels = torch.arange(k.size()[0], dtype=torch.long)[rank*bz:bz*(rank+1)].cuda()
        return nn.CrossEntropyLoss(reduction=redunction)(logits, labels) * (2 * self.T)

    def contrastive_mix_loss(self, q, k, lam1, lam2):
        # normalize
        bz = q.shape[0]
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        rank = torch.distributed.get_rank()
        labels_flip = torch.arange(k.size()[0], dtype=torch.long).flip(0)[rank * bz:bz * (rank + 1)].cuda()  # B
        labels = torch.arange(k.size()[0], dtype=torch.long)[rank * bz:bz * (rank + 1)].cuda()
        logits = logits.softmax(-1)
        #logits /= logits.sum(-1)
        loss = -torch.log(-(nn.NLLLoss(reduction='none')(logits, labels).mul(lam1) +
                            nn.NLLLoss(reduction='none')(logits, labels_flip).mul(lam2)))
        return loss.mean() * (2 * self.T)

    def forward(self, x1, x2, m):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """

        # compute features
        bz, _,_,_ = x1.shape
        lam = np.random.beta(1., 1., size=[x1.shape[0], 1, 1, 1]).astype(np.float32)
        lam = torch.from_numpy(lam).cuda()
        rank = torch.distributed.get_rank()
        x1, x2, lam = concat_all_gather(x1), concat_all_gather(x2), concat_all_gather(lam)
        swap = torch.rand([1]).cuda()
        swap = concat_all_gather(swap)
        if swap[0]>0.5:
            x1, x2 = x2, x1
        x1_mix, new_lam = datamixing(x1.clone(), lam, rank,bz)
        x2_mix, new_lam = datamixing(x2.clone(), lam, rank,bz)
        q1 = self.predictor(self.base_encoder(x1[bz*rank:bz*(rank+1)]))
        q2_mix = self.predictor(self.base_encoder(x2_mix[bz*rank:bz*(rank+1)]))
        # part 1
        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder
            # compute momentum features as targets
            k1 = self.momentum_encoder(x1[bz*rank:bz*(rank+1)])
            k1_mix = self.momentum_encoder(x1_mix[bz*rank:bz*(rank+1)])
            k2 = self.momentum_encoder(x2[bz*rank:bz*(rank+1)])
        source_loss = self.contrastive_loss(q1, k2, "mean")
        if new_lam is not None:
            lam = new_lam
        lam = lam.squeeze()
        mixloss_source = self.contrastive_mix_loss(q2_mix, k1, lam[bz*rank:bz*(rank+1)], 1-lam[bz*rank:bz*(rank+1)])

        common = np.minimum(lam[bz*rank:bz*(rank+1)].cpu(), 1-lam.flip(0)[bz*rank:bz*(rank+1)].clone().cpu()).cuda()+np.minimum(1-lam[bz*rank:bz*(rank+1)].cpu(), lam.flip(0)[bz*rank:bz*(rank+1)].clone().cpu()).cuda()
        mixloss_mix = self.contrastive_mix_loss(q2_mix, k1_mix, 1/(1+common), common/(1+common))
        return source_loss, mixloss_source/2, mixloss_mix/2



class MoCo_ResNet(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc, self.momentum_encoder.fc # remove original fc layer

        # projectors
        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)


class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
