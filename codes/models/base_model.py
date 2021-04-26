from collections import OrderedDict
import os.path as osp

import torch

from utils.base_utils import get_logger


class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.verbose = opt['verbose']
        self.scale = opt['scale']
        self.logger = get_logger('base')
        self.device = torch.device(opt['device'])
        self.is_train = opt['is_train']

        if self.is_train:
            self.ckpt_dir = opt['train']['ckpt_dir']
            self.log_decay = opt['logger'].get('decay', 0.99)
            self.log_dict = OrderedDict()
            self.running_log_dict = OrderedDict()

    def set_network(self):
        pass

    def config_training(self):
        pass

    def set_criterion(self):
        pass

    def train(self, data):
        pass

    def infer(self, data):
        pass

    def update_learning_rate(self):
        if hasattr(self, 'sched_G') and self.sched_G is not None:
            self.sched_G.step()

        if hasattr(self, 'sched_D') and self.sched_D is not None:
            self.sched_D.step()

    def get_current_learning_rate(self):
        lr_dict = OrderedDict()

        if hasattr(self, 'optim_G'):
            lr_dict['lr_G'] = self.optim_G.param_groups[0]['lr']

        if hasattr(self, 'optim_D'):
            lr_dict['lr_D'] = self.optim_D.param_groups[0]['lr']

        return lr_dict

    def update_running_log(self):
        d = self.log_decay
        for k in self.log_dict.keys():
            current_val = self.log_dict[k]
            running_val = self.running_log_dict.get(k)

            if running_val is None:
                running_val = current_val
            else:
                running_val = d * running_val + (1.0 - d) * current_val

            self.running_log_dict[k] = running_val

    def get_current_log(self):
        return self.log_dict

    def get_running_log(self):
        return self.running_log_dict

    def save(self, current_iter):
        pass

    def save_network(self, net, net_label, current_iter):
        save_filename = '{}_iter{}.pth'.format(net_label, current_iter)
        save_path = osp.join(self.ckpt_dir, save_filename)
        torch.save(net.state_dict(), save_path)

    def save_training_state(self, current_epoch, current_iter):
        # TODO
        pass

    def load_network(self, net, load_path):
        net.load_state_dict(torch.load(load_path))

    def pad_sequence(self, lr_data):
        """
        Parameters:
            :param lr_data: tensor in shape tchw
        """
        padding_mode = self.opt['test'].get('padding_mode', 'reflect')
        n_pad_front = self.opt['test'].get('num_pad_front', 0)

        if padding_mode == 'reflect':
            lr_data = torch.cat(
                [lr_data[1: 1 + n_pad_front, ...].flip(0), lr_data], dim=0)

        elif padding_mode == 'replicate':
            lr_data = torch.cat(
                [lr_data[:1, ...].expand(n_pad_front, -1, -1, -1), lr_data],
                dim=0)

        elif padding_mode == 'dual-reflect':
            lr_data = torch.cat(
                [lr_data[1: 1+n_pad_front, ...].flip(0), lr_data, lr_data[-1-n_pad_front: -1, ...].flip(0)],
                dim=0)

        else:
            raise ValueError('Unrecognized padding mode: {}'.format(
                padding_mode))

        return lr_data, n_pad_front
