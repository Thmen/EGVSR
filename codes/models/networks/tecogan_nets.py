import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_nets import BaseSequenceGenerator, BaseSequenceDiscriminator
from utils.net_utils import space_to_depth, backward_warp, get_upsampling_func
from utils.net_utils import initialize_weights
from utils.data_utils import float32_to_uint8

import flow_vis

# -------------------- generator modules -------------------- #
class FNet(nn.Module):
    """ Optical flow estimation network
    """

    def __init__(self, in_nc):
        super(FNet, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(2*in_nc, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2))

        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2))

        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2))

        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.decoder3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.flow = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 2, 3, 1, 1, bias=True))

    def forward(self, x1, x2):
        """ Compute optical flow from x1 to x2
        """

        out = self.encoder1(torch.cat([x1, x2], dim=1))
        out = self.encoder2(out)
        out = self.encoder3(out)
        out = F.interpolate(
            self.decoder1(out), scale_factor=2, mode='bilinear', align_corners=False)
        out = F.interpolate(
            self.decoder2(out), scale_factor=2, mode='bilinear', align_corners=False)
        out = F.interpolate(
            self.decoder3(out), scale_factor=2, mode='bilinear', align_corners=False)
        out = torch.tanh(self.flow(out)) * 24  # 24 is the max velocity

        return out


class ResidualBlock(nn.Module):
    """ Residual block without batch normalization
    """

    def __init__(self, nf=64):
        super(ResidualBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True))

    def forward(self, x):
        out = self.conv(x) + x

        return out


class SRNet(nn.Module):
    """ Reconstruction & Upsampling network
    """

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upsample_func=None,
                 scale=4):
        super(SRNet, self).__init__()

        # input conv.
        self.conv_in = nn.Sequential(
            nn.Conv2d((scale**2 + 1) * in_nc, nf, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True))

        # residual blocks
        self.resblocks = nn.Sequential(*[ResidualBlock(nf) for _ in range(nb)])

        # upsampling
        self.conv_up = nn.Sequential(
            nn.ConvTranspose2d(nf, nf, 3, 2, 1, output_padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, 3, 2, 1, output_padding=1, bias=True),
            nn.ReLU(inplace=True))

        # output conv.
        self.conv_out = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # upsampling function
        self.upsample_func = upsample_func

    def forward(self, lr_curr, hr_prev_tran):
        """ lr_curr: the current lr data in shape nchw
            hr_prev_tran: the previous transformed hr_data in shape n(4*4*c)hw
        """

        out = self.conv_in(torch.cat([lr_curr, hr_prev_tran], dim=1))
        out = self.resblocks(out)
        out = self.conv_up(out)
        out = self.conv_out(out)
        out += self.upsample_func(lr_curr)

        return out


class FRNet(BaseSequenceGenerator):
    """ Frame-recurrent network proposed in https://arxiv.org/abs/1801.04590
    """

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, degradation='BI',
                 scale=4):
        super(FRNet, self).__init__()

        self.scale = scale

        # get upsampling function according to the degradation mode
        self.upsample_func = get_upsampling_func(self.scale, degradation)

        # define fnet & srnet
        self.fnet = FNet(in_nc)
        self.srnet = SRNet(in_nc, out_nc, nf, nb, self.upsample_func)

    def generate_dummy_input(self, lr_size):
        c, lr_h, lr_w = lr_size
        s = self.scale

        lr_curr = torch.rand(1, c, lr_h, lr_w, dtype=torch.float32)
        lr_prev = torch.rand(1, c, lr_h, lr_w, dtype=torch.float32)
        hr_prev = torch.rand(1, c, s * lr_h, s * lr_w, dtype=torch.float32)

        data_dict = {
            'lr_curr': lr_curr,
            'lr_prev': lr_prev,
            'hr_prev': hr_prev
        }

        return data_dict

    def forward(self, lr_curr, lr_prev, hr_prev):
        """
            Parameters:
                :param lr_curr: the current lr data in shape nchw
                :param lr_prev: the previous lr data in shape nchw
                :param hr_prev: the previous hr data in shape nc(4h)(4w)
        """

        # estimate lr flow (lr_curr -> lr_prev)
        lr_flow = self.fnet(lr_curr, lr_prev)

        # pad if size is not a multiple of 8
        pad_h = lr_curr.size(2) - lr_curr.size(2) // 8 * 8
        pad_w = lr_curr.size(3) - lr_curr.size(3) // 8 * 8
        lr_flow_pad = F.pad(lr_flow, (0, pad_w, 0, pad_h), 'reflect')

        # upsample lr flow
        hr_flow = self.scale * self.upsample_func(lr_flow_pad)

        # warp hr_prev
        hr_prev_warp = backward_warp(hr_prev, hr_flow)

        # compute hr_curr
        hr_curr = self.srnet(lr_curr, space_to_depth(hr_prev_warp, self.scale))

        return hr_curr, hr_flow, hr_prev_warp

    def forward_sequence(self, lr_data):
        """
            Parameters:
                :param lr_data: lr data in shape ntchw
        """

        n, t, c, lr_h, lr_w = lr_data.size()
        hr_h, hr_w = lr_h * self.scale, lr_w * self.scale

        # calculate optical flows
        lr_prev = lr_data[:, :-1, ...].reshape(n * (t - 1), c, lr_h, lr_w)
        lr_curr = lr_data[:, 1:, ...].reshape(n * (t - 1), c, lr_h, lr_w)
        lr_flow = self.fnet(lr_curr, lr_prev)  # n*(t-1),2,h,w

        # upsample lr flows
        hr_flow = self.scale * self.upsample_func(lr_flow)
        hr_flow = hr_flow.view(n, (t - 1), 2, hr_h, hr_w)

        # compute the first hr data
        hr_data = []
        hr_prev = self.srnet(
            lr_data[:, 0, ...],
            torch.zeros(n, (self.scale**2)*c, lr_h, lr_w, dtype=torch.float32,
                        device=lr_data.device))
        hr_data.append(hr_prev)

        # compute the remaining hr data
        for i in range(1, t):
            # warp hr_prev
            hr_prev_warp = backward_warp(hr_prev, hr_flow[:, i - 1, ...])

            # compute hr_curr
            hr_curr = self.srnet(
                lr_data[:, i, ...],
                space_to_depth(hr_prev_warp, self.scale))

            # save and update
            hr_data.append(hr_curr)
            hr_prev = hr_curr

        hr_data = torch.stack(hr_data, dim=1)  # n,t,c,hr_h,hr_w

        # construct output dict
        ret_dict = {
            'hr_data': hr_data,  # n,t,c,hr_h,hr_w
            'hr_flow': hr_flow,  # n,t,2,hr_h,hr_w
            'lr_prev': lr_prev,  # n(t-1),c,lr_h,lr_w
            'lr_curr': lr_curr,  # n(t-1),c,lr_h,lr_w
            'lr_flow': lr_flow,  # n(t-1),2,lr_h,lr_w
        }

        return ret_dict

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
        lr_prev = torch.zeros(1, c, h, w, dtype=torch.float32).to(device)
        hr_prev = torch.zeros(
            1, c, s * h, s * w, dtype=torch.float32).to(device)

        for i in range(tot_frm):
            with torch.no_grad():
                self.eval()

                lr_curr = lr_data[i: i + 1, ...].to(device)
                hr_curr, hr_flow, hr_warp = self.forward(lr_curr, lr_prev, hr_prev)
                lr_prev, hr_prev = lr_curr, hr_curr

                hr_warp = hr_warp.squeeze(0).cpu().numpy()  # chw|rgb|uint8
                hr_frm = hr_warp.transpose(1, 2, 0)  # hwc
                flow_frm = hr_flow.squeeze(0).cpu().numpy()  # chw|rgb|uint8
                flow_uv = flow_frm.transpose(1, 2, 0)  # hwc
                flow_color = flow_vis.flow_to_color(flow_uv, convert_to_bgr=False)

            hr_seq.append(float32_to_uint8(hr_frm))
            # hr_seq.append(float32_to_uint8(flow_color))

        return np.stack(hr_seq)



# ------------------ discriminator modules ------------------ #
class DiscriminatorBlocks(nn.Module):
    def __init__(self):
        super(DiscriminatorBlocks, self).__init__()

        self.block1 = nn.Sequential(  # /2
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.block2 = nn.Sequential(  # /4
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.block3 = nn.Sequential(  # /8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.block4 = nn.Sequential(  # /16
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        feature_list = [out1, out2, out3, out4]

        return out4, feature_list


class SpatioTemporalDiscriminator(BaseSequenceDiscriminator):
    """ Spatio-Temporal discriminator in proposed in TecoGAN
    """

    def __init__(self, in_nc=3, spatial_size=128, tempo_range=3, scale=4):
        super(SpatioTemporalDiscriminator, self).__init__()

        # basic settings
        mult = 3  # (conditional triplet, input triplet, warped triplet)
        self.spatial_size = spatial_size
        self.tempo_range = tempo_range
        assert self.tempo_range == 3, 'currently only support 3 as tempo_range'
        self.scale = scale

        # input conv.
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_nc*tempo_range*mult, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        # discriminator block
        self.discriminator_block = DiscriminatorBlocks()  # downsample 16x

        # classifier
        self.dense = nn.Linear(256 * spatial_size // 16 * spatial_size // 16, 1)

    def forward(self, x):
        out = self.conv_in(x)
        out, feature_list = self.discriminator_block(out)

        out = out.view(out.size(0), -1)
        out = self.dense(out)

        return out, feature_list

    def forward_sequence(self, data, args_dict):
        """
            :param data: should be either hr_data or gt_data
            :param args_dict: a dict including data/config needed here
        """

        # ------------ setup params ------------ #
        net_G = args_dict['net_G']
        lr_data = args_dict['lr_data']
        bi_data = args_dict['bi_data']
        hr_flow = args_dict['hr_flow']

        n, t, c, lr_h, lr_w = lr_data.size()
        _, _, _, hr_h, hr_w = data.size()

        s_size = self.spatial_size
        t = t // 3 * 3  # discard other frames
        n_clip = n * t // 3  # total number of 3-frame clips in all batches

        c_size = int(s_size * args_dict['crop_border_ratio'])
        n_pad = (s_size - c_size) // 2


        # ------------ compute forward & backward flow ------------ #
        if 'hr_flow_merge' not in args_dict:
            if args_dict['use_pp_crit']:
                hr_flow_bw = hr_flow[:, 0:t:3, ...]  # e.g., frame1 -> frame0
                hr_flow_idle = torch.zeros_like(hr_flow_bw)
                hr_flow_fw = hr_flow.flip(1)[:, 1:t:3, ...]
            else:
                lr_curr = lr_data[:, 1:t:3, ...]
                lr_curr = lr_curr.reshape(n_clip, c, lr_h, lr_w)

                lr_next = lr_data[:, 2:t:3, ...]
                lr_next = lr_next.reshape(n_clip, c, lr_h, lr_w)

                # compute forward flow
                lr_flow_fw = net_G.fnet(lr_curr, lr_next)
                hr_flow_fw = self.scale * net_G.upsample_func(lr_flow_fw)

                hr_flow_bw = hr_flow[:, 0:t:3, ...]  # e.g., frame1 -> frame0
                hr_flow_idle = torch.zeros_like(hr_flow_bw)  # frame1 -> frame1
                hr_flow_fw = hr_flow_fw.view(n, t // 3, 2, hr_h, hr_w)  # frame1 -> frame2

            # merge bw/idle/fw flows
            hr_flow_merge = torch.stack(
                [hr_flow_bw, hr_flow_idle, hr_flow_fw], dim=2)  # n,t//3,3,2,h,w

            # reshape and stop gradient propagation
            hr_flow_merge = hr_flow_merge.view(n_clip * 3, 2, hr_h, hr_w).detach()

        else:
            # reused data to reduce computations
            hr_flow_merge = args_dict['hr_flow_merge']


        # ------------ build up inputs for D (3 parts) ------------ #
        # part 1: bicubic upsampled data (conditional inputs)
        cond_data = bi_data[:, :t, ...].reshape(n_clip, 3, c, hr_h, hr_w)
        # note: permutation is not necessarily needed here, it's just to keep
        #       the same impl. as TecoGAN-Tensorflow (i.e., rrrgggbbb)
        cond_data = cond_data.permute(0, 2, 1, 3, 4)
        cond_data = cond_data.reshape(n_clip, c * 3, hr_h, hr_w)

        # part 2: original data
        orig_data = data[:, :t, ...].reshape(n_clip, 3, c, hr_h, hr_w)
        orig_data = orig_data.permute(0, 2, 1, 3, 4)
        orig_data = orig_data.reshape(n_clip, c * 3, hr_h, hr_w)

        # part 3: warped data
        warp_data = backward_warp(
            data[:, :t, ...].reshape(n * t, c, hr_h, hr_w), hr_flow_merge)
        warp_data = warp_data.view(n_clip, 3, c, hr_h, hr_w)
        warp_data = warp_data.permute(0, 2, 1, 3, 4)
        warp_data = warp_data.reshape(n_clip, c * 3, hr_h, hr_w)
        # remove border to increase training stability as proposed in TecoGAN
        warp_data = F.pad(
            warp_data[..., n_pad: n_pad + c_size, n_pad: n_pad + c_size],
            (n_pad,) * 4, mode='constant')

        # combine 3 parts together
        input_data = torch.cat([orig_data, warp_data, cond_data], dim=1)


        # ------------ classify ------------ #
        pred = self.forward(input_data)  # out, feature_list


        # construct output dict (return other data beside pred)
        ret_dict = {
            'hr_flow_merge': hr_flow_merge
        }

        return pred, ret_dict


class SpatialDiscriminator(BaseSequenceDiscriminator):
    """ Spatial discriminator
    """

    def __init__(self, in_nc=3, spatial_size=128, use_cond=False):
        super(SpatialDiscriminator, self).__init__()

        # basic settings
        self.use_cond = use_cond  # whether to use conditional input
        mult = 2 if self.use_cond else 1
        tempo_range = 1

        # input conv
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_nc*tempo_range*mult, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        # discriminator block
        self.discriminator_block = DiscriminatorBlocks()  # /16

        # classifier
        self.dense = nn.Linear(256 * spatial_size // 16 * spatial_size // 16, 1)

    def forward(self, x):
        out = self.conv_in(x)
        out, feature_list = self.discriminator_block(out)

        out = out.view(out.size(0), -1)
        out = self.dense(out)

        return out, feature_list

    def forward_sequence(self, data, args_dict):
        # ------------ setup params ------------ #
        n, t, c, hr_h, hr_w = data.size()
        data = data.view(n * t, c, hr_h, hr_w)


        # ------------ build up inputs for D ------------ #
        if self.use_cond:
            bi_data = args_dict['bi_data'].view(n * t, c, hr_h, hr_w)
            input_data = torch.cat([bi_data, data], dim=1)
        else:
            input_data = data


        # ------------ classify ------------ #
        pred = self.forward(input_data)


        # construct output dict (nothing to return)
        ret_dict = {}

        return pred, ret_dict