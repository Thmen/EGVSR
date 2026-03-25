
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .util import (
    normalize_tensor, l2, psnr, dssim, rgb2lab, tensor2np, np2tensor,
    tensor2tensorlab, tensorlab2tensor, tensor2im, im2tensor, tensor2vec,
    voc_ap,
)

import torch


class PerceptualLoss(torch.nn.Module):
    def __init__(self, model='net-lin', net='alex', colorspace='rgb',
                 spatial=False, use_gpu=True, gpu_ids=[0], version='0.1'):
        super(PerceptualLoss, self).__init__()
        from .dist_model import DistModel

        print('Setting up Perceptual loss...')
        self.use_gpu = use_gpu
        self.spatial = spatial
        self.gpu_ids = gpu_ids
        self.model = DistModel()
        self.model.initialize(
            model=model, net=net, use_gpu=use_gpu, colorspace=colorspace,
            spatial=self.spatial, gpu_ids=gpu_ids, version=version)
        print('...[%s] initialized' % self.model.name())
        print('...Done')

    def forward(self, pred, target, normalize=False):
        if normalize:
            target = 2 * target - 1
            pred = 2 * pred - 1
        return self.model.forward(target, pred)
