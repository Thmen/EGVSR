import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch as F

from .base_nets import BaseSequenceGenerator
from utils.data_utils import float32_to_uint8


class ESPNet(BaseSequenceGenerator):
    def __init__(self, scale=4, in_nc=1, out_nc=1, up_method='subconv'):
        super(ESPNet, self).__init__()

        self.scale = scale
        self.scale_channel = 1 * (scale ** 2)
        self.up_method = up_method  # 'subconv', 'reconv', 'deconv'
        self.conv1 = nn.Conv2d(in_nc, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1))

        self.pwconv = nn.Conv2d(32, self.scale_channel, (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(scale)

        self.resize = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True) # bicubic, bilinear
        self.reconv = nn.Conv2d(32, out_nc, (1, 1))

        self.deconv = nn.ConvTranspose2d(32, out_nc, kernel_size=5, stride=scale,    # k -> odd num
                                         padding=(5-1)//2, output_padding=scale-1)             # 2p = (k-s) + o_p

    def generate_dummy_input(self, lr_size):
        c, lr_h, lr_w = lr_size
        s = self.scale

        lr_curr = torch.rand(1, c, lr_h, lr_w, dtype=torch.float32)

        data_dict = {
            'lr_curr': lr_curr
        }

        return data_dict

    def forward(self, lr_curr):
        x = lr_curr
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.tanh(self.conv3(x))

        if self.up_method == 'deconv':
            # deconvolution / transposed convolution
            x = self.deconv(x)
        elif self.up_method == 'reconv':
            # resize convolution
            x = self.reconv(self.resize(x))
        else:       # self.up_method == 'subconv':
            # pixel-shuffle / sub-pixel convolution
            x = self.pixel_shuffle(self.pwconv(x))

        return F.sigmoid(x)

    def infer_sequence(self, lr_data, device):
        """
            Parameters:
                :param lr_data: torch.FloatTensor in shape tchw
                :param device: torch.device

                :return hr_seq: uint8 np.ndarray in shape tchw
        """

        # setup params
        tot_frm, c, h, w = lr_data.size()
        s = self.scale

        # forward
        hr_seq = []

        for i in range(tot_frm):
            with torch.no_grad():
                self.eval()

                lr_curr = lr_data[i: i + 1, ...].to(device)
                hr_curr = self.forward(lr_curr)
                hr_frm = hr_curr.squeeze(0).cpu().numpy()  # chw|rgb|uint8

            hr_seq.append(float32_to_uint8(hr_frm))

        return np.stack(hr_seq).transpose(0, 2, 3, 1)  # thwc

