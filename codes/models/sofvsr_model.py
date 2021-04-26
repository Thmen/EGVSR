from .base_model import BaseModel
from .networks import define_generator
from utils import data_utils
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as tvs


class SOFVSRModel(BaseModel):
    def __init__(self, opt):
        super(SOFVSRModel, self).__init__(opt)

        if self.verbose:
            self.logger.info('{} Model Info {}'.format('=' * 20, '=' * 20))
            self.logger.info('Model: {}'.format(opt['model']['name']))

        # set network
        self.set_network()

    def set_network(self):
        # define net G
        self.net_G = define_generator(self.opt).to(self.device)
        if self.verbose:
            self.logger.info('Generator: {}\n'.format(
                self.opt['model']['generator']['name']) + self.net_G.__str__())

        # load network
        load_path_G = self.opt['model']['generator'].get('load_path')
        if load_path_G is not None:
            self.load_network(self.net_G, load_path_G)
            if self.verbose:
                self.logger.info('Load generator from: {}'.format(load_path_G))

    def infer(self, lr_data):

        lr_data = data_utils.canonicalize(lr_data)  # to torch.FloatTensor  thwc

        print(lr_data.size())
        _, h, w, _ = lr_data.size()

        lr_yuv = data_utils.rgb2yCbCr(lr_data)
        lr_yuv = lr_yuv.permute(0, 3, 1, 2)  # thwc

        lr_y = lr_yuv[:, 0:1, :, :]
        lr_u = lr_yuv[:, 1:2, :, :]
        lr_v = lr_yuv[:, 2:3, :, :]

        # dual direct temporal padding
        lr_y_seq, n_pad_front = self.pad_sequence(lr_y)

        # infer
        hr_y_seq = self.net_G.infer_sequence(lr_y_seq, self.device)
        hr_u_seq = tvs.resize(lr_u, [self.scale*h, self.scale*w], interpolation=3)    # bilinear:2(default) bicubic:3
        hr_v_seq = tvs.resize(lr_v, [self.scale*h, self.scale*w], interpolation=3)    # bilinear:2(default) bicubic:3

        hr_yuv = torch.cat((hr_y_seq, hr_u_seq, hr_v_seq), dim=1)
        hr_yuv = hr_yuv.permute(0, 2, 3, 1)  # tchw

        hr_rgb = data_utils.yCbCr2rgb(hr_yuv).numpy()
        hr_seq = data_utils.float32_to_uint8(hr_rgb) # thwc|rgb|uint8

        return hr_seq

    # def eval(self, inputs, labels=None, **kwargs):
    #     metrics = {}
    #     pre, cur, nxt = torch.split(inputs[0], 1, dim=1)
    #     low_res = torch.cat([pre, cur, nxt], dim=2)
    #     low_res = torch.squeeze(low_res, dim=1)
    #     sr, _, _ = self.sof(low_res)
    #     sr = sr.cpu().detach()
    #     if labels is not None:
    #         hr = labels[0][:, self.center]
    #         metrics['psnr'] = Metrics.psnr(sr, hr)
    #     return [sr.numpy()], metrics

