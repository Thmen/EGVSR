from collections import OrderedDict

import torch
import torch.optim as optim

from .base_model import BaseModel
from .networks import define_generator
from .optim import define_criterion, define_lr_schedule
from utils import net_utils, data_utils


class VSRModel(BaseModel):
    """ A model wraper for objective video super-resolution

        It contains a generator, as well as relative functions to train and test
        the generator
    """

    def __init__(self, opt):
        super(VSRModel, self).__init__(opt)

        if self.verbose:
            self.logger.info('{} Model Info {}'.format('=' * 20, '=' * 20))
            self.logger.info('Model: {}'.format(opt['model']['name']))

        # set network
        self.set_network()

        # configs for training
        if self.is_train:
            self.config_training()

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

    def config_training(self):
        # set criterion
        self.set_criterion()

        # set optimizer
        self.optim_G = optim.Adam(
            self.net_G.parameters(),
            lr=self.opt['train']['generator']['lr'],
            weight_decay=self.opt['train']['generator'].get('weight_decay', 0),
            betas=(
                self.opt['train']['generator'].get('beta1', 0.9),
                self.opt['train']['generator'].get('beta2', 0.999)))

        # set lr schedule
        self.sched_G = define_lr_schedule(
            self.opt['train']['generator'].get('lr_schedule'), self.optim_G)

    def set_criterion(self):
        # pixel criterion
        self.pix_crit = define_criterion(
            self.opt['train'].get('pixel_crit'))

        # warping criterion
        self.warp_crit = define_criterion(
            self.opt['train'].get('warping_crit'))

    def train(self, data):
        """ Function of mini-batch training

            Parameters:
                :param data: a batch of training data (lr & gt) in shape ntchw
        """

        # ------------ prepare data ------------ #
        lr_data, gt_data = data['lr'], data['gt']


        # ------------ clear optim ------------ #
        self.net_G.train()
        self.optim_G.zero_grad()


        # ------------ forward G ------------ #
        net_G_output_dict = self.net_G.forward_sequence(lr_data)
        hr_data = net_G_output_dict['hr_data']


        # ------------ optimize G ------------ #
        loss_G = 0
        self.log_dict = OrderedDict()

        # pixel loss
        pix_w = self.opt['train']['pixel_crit'].get('weight', 1)
        loss_pix_G = pix_w * self.pix_crit(hr_data, gt_data)
        loss_G += loss_pix_G
        self.log_dict['l_pix_G'] = loss_pix_G.item()

        # warping loss
        if self.warp_crit is not None:
            # warp lr_prev according to lr_flow
            lr_curr = net_G_output_dict['lr_curr']
            lr_prev = net_G_output_dict['lr_prev']
            lr_flow = net_G_output_dict['lr_flow']
            lr_warp = net_utils.backward_warp(lr_prev, lr_flow)

            warp_w = self.opt['train']['warping_crit'].get('weight', 1)
            loss_warp_G = warp_w * self.warp_crit(lr_warp, lr_curr)
            loss_G += loss_warp_G
            self.log_dict['l_warp_G'] = loss_warp_G.item()

        # optimize
        loss_G.backward()
        self.optim_G.step()

    def infer(self, lr_data):
        """ Function of inference

            Parameters:
                :param lr_data: a rgb video sequence with shape thwc
                :return: a rgb video sequence with type np.uint8 and shape thwc
        """
        print(lr_data.size())

        # canonicalize
        lr_data = data_utils.canonicalize(lr_data)  # to torch.FloatTensor
        lr_data = lr_data.permute(0, 3, 1, 2)  # tchw

        # temporal padding
        lr_data, n_pad_front = self.pad_sequence(lr_data)

        # infer
        hr_seq = self.net_G.infer_sequence(lr_data, self.device)
        hr_seq = hr_seq[n_pad_front:, ...]

        return hr_seq

    def save(self, current_iter):
        self.save_network(self.net_G, 'G', current_iter)
