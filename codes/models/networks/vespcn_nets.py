"""
VESPCN
Created Author: YP CAO
Created Date: 2021-3-1
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_nets import BaseSequenceGenerator
from utils.data_utils import float32_to_uint8
from utils.motion import STN, CoarseFineFlownet, pad_if_divide


_logger = logging.getLogger("VSR.VESPCN")
_logger.info("LICENSE: VESPCN is proposed at CVPR2017 by Twitter. ")


class ReluRB(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(ReluRB, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, outchannels, 3, 1, 1)

    def forward(self, inputs):
        x = F.relu(inputs)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x + inputs


class MotionCompensation(nn.Module):
    def __init__(self, channel, gain=32):
        super(MotionCompensation, self).__init__()
        self.gain = gain
        self.flownet = CoarseFineFlownet(channel)
        self.warp_f = STN(padding_mode='border')

    def forward(self, target, ref):
        flow = self.flownet(target, ref, self.gain)
        warping = self.warp_f(ref, flow[:, 0], flow[:, 1])
        return warping, flow


class SRNet(nn.Module):
    def __init__(self, scale, channel, depth):
        super(SRNet, self).__init__()
        self.entry = nn.Conv2d(channel * depth, 64, 3)
        self.exit = nn.Conv2d(64, channel, 3)
        self.body = nn.Sequential(
            ReluRB(64, 64),
            ReluRB(64, 64),
            ReluRB(64, 64),
            nn.ReLU(True))
        self.conv = nn.Conv2d(64, 64 * scale ** 2, 3)
        self.up = nn.PixelShuffle(scale)

    def forward(self, inputs):
        x = self.entry(inputs)
        y = self.body(x) + x
        y = self.conv(y)
        y = self.up(y)
        y = self.exit(y)
        return y


class VESPNet(BaseSequenceGenerator):
    def __init__(self, scale, channel, depth):
        super(VESPNet, self).__init__()
        self.sr = SRNet(scale, channel, depth)
        # self.mc = MotionCompensation(channel)
        self.mc = CoarseFineFlownet(channel)
        self.warp_f = STN(padding_mode='border')
        self.depth = depth
        self.scale = scale

    def forward(self, lr_seq):

        # print(lr_seq.size())

        inputs = pad_if_divide(lr_seq, self.scale)
        a = (inputs[0].size(1) - inputs[0].size(1)) * self.scale
        b = (inputs[0].size(2) - inputs[0].size(2)) * self.scale
        slice_h = slice(None) if a == 0 else slice(a // 2, -a // 2)
        slice_w = slice(None) if b == 0 else slice(b // 2, -b // 2)

        center = self.depth // 2
        target = inputs[center:center+1]
        refs = torch.cat((inputs[center+1:], inputs[center+1:]), 0)

        warps = []
        flows = []
        for i in range(self.depth-1):
            flow = self.mc(target, refs[i:i+1], 32)
            warp = self.warp_f(refs[i:i+1], flow[:, 0], flow[:, 1])
            # warp, flow = self.mc(target, refs[i:i+1])
            warps.append(warp)
            flows.append(flow)

        warps.append(target)
        x = torch.cat(warps, 1)
        sr = self.sr(x)
        return sr[..., slice_h, slice_w]

    def generate_dummy_input(self, lr_size):
        n = self.depth
        c, lr_h, lr_w = lr_size

        lr_seq = torch.rand(n, c, lr_h, lr_w, dtype=torch.float32)
        data_dict = {
            'lr_seq': lr_seq
        }

        return data_dict

    def infer_sequence(self, lr_data, device):
        """
            Parameters:
                :param lr_data: torch.FloatTensor in shape tchw
                :param device: torch.device

                :return hr_seq: uint8 np.ndarray in shape tchw
        """

        # setup params
        tot_frm, c, h, w = lr_data.size()
        p = self.depth // 2

        print(lr_data.size())

        # forward
        hr_seq = []

        for i in range(p, tot_frm-p):
            with torch.no_grad():
                self.eval()

                lr_seq = lr_data[i-p: i+p+1, ...].to(device)
                hr_curr = self.forward(lr_seq)
                hr_frm = hr_curr.squeeze(0).cpu().numpy()  # chw|rgb|uint8

            hr_seq.append(float32_to_uint8(hr_frm))

        return np.stack(hr_seq).transpose(0, 2, 3, 1)  # thwc