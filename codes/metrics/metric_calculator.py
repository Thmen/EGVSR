import os
import os.path as osp
import json
from collections import OrderedDict

import numpy as np
import cv2
import torch

from utils import base_utils, data_utils, net_utils
from .LPIPS.models.dist_model import DistModel


class MetricCalculator():
    """ Metric calculator for model evaluation

        Currently supported metrics:
            * PSNR (RGB and Y)
            * LPIPS
            * tOF as described in TecoGAN paper

        TODO:
            * save/print metrics in a fixed order
    """

    def __init__(self, opt):
        # initialize
        self.metric_opt = opt['metric']
        self.device = torch.device(opt['device'])

        self.psnr_mult = 1
        self.psnr_colorspace = ''
        self.lpips_mult = 1
        self.dm = None
        self.tof_mult = 1

        self.reset()

        # update configs for each metric
        for metric_type, cfg in self.metric_opt.items():
            if metric_type.lower() == 'psnr':
                self.psnr_mult = cfg.get('mult', self.psnr_mult)
                self.psnr_colorspace = cfg['colorspace']

            if metric_type.lower() == 'lpips':
                self.lpips_mult = cfg.get('mult', self.lpips_mult)
                self.dm = DistModel()
                self.dm.initialize(
                    model=cfg['model'],
                    net=cfg['net'],
                    colorspace=cfg['colorspace'],
                    spatial=cfg['spatial'],
                    use_gpu=(opt['device'] == 'cuda'),
                    gpu_ids=[0],
                    version=cfg['version'])

            if metric_type.lower() == 'tof':
                self.tof_mult = cfg.get('mult', self.tof_mult)

    def reset(self):
        self.reset_per_sequence()
        self.metric_dict = OrderedDict()

    def reset_per_sequence(self):
        self.seq_idx_curr = ''
        self.true_img_cur = None
        self.pred_img_cur = None
        self.true_img_pre = None
        self.pred_img_pre = None

    def get_averaged_results(self):
        metric_avg_dict = {}

        for metric_type in self.metric_opt.keys():
            metric_avg_per_seq = []
            for seq, metric_dict_per_seq in self.metric_dict.items():
                metric_avg_per_seq.append(
                    np.mean(metric_dict_per_seq[metric_type]))

            metric_avg_dict[metric_type] = np.mean(metric_avg_per_seq)

        return metric_avg_dict

    def display_results(self):
        logger = base_utils.get_logger('base')

        # per sequence results
        for seq, metric_dict_per_seq in self.metric_dict.items():
            logger.info('Sequence: {}'.format(seq))
            for metric_type in self.metric_opt.keys():
                mult = getattr(self, '{}_mult'.format(metric_type.lower()))
                logger.info('\t{}: {:.6f} (x{})'.format(
                    metric_type,
                    mult*np.mean(metric_dict_per_seq[metric_type]), mult))

        # average results
        logger.info('Average')
        metric_avg_dict = self.get_averaged_results()
        for metric_type, avg_result in metric_avg_dict.items():
            mult = getattr(self, '{}_mult'.format(metric_type.lower()))
            logger.info('\t{}: {:.6f} (x{})'.format(
                metric_type, mult*avg_result, mult))

    def save_results(self, model_idx, save_path, override=False):
        # load previous results if existed
        if osp.exists(save_path):
            with open(save_path, 'r') as f:
                json_dict = json.load(f)
        else:
            json_dict = dict()

        # merge (averaged) results
        if model_idx not in json_dict:
            json_dict[model_idx] = dict()

        metric_avg_dict = self.get_averaged_results()
        for metric_type, avg_result in metric_avg_dict.items():
            # whether to override or skip when this metric already exists
            if metric_type in json_dict[model_idx] and not override:
                continue

            json_dict[model_idx][metric_type] = '{:.6f}'.format(avg_result)

        # sort
        json_dict = OrderedDict(sorted(
            json_dict.items(), key=lambda x: int(x[0].replace('G_iter', ''))))

        # save results
        with open(save_path, 'w') as f:
            json.dump(json_dict, f, sort_keys=False, indent=4)

    def compute_dataset_metrics(self, true_dir, pred_dir, sequence_list=None):
        """ compute metrics for a dataset, *_dir are the root of dataset, which
            contains several video clips/folders
        """

        # select sequences
        if sequence_list is None:
            sequence_list = sorted(list(
                set(os.listdir(true_dir)) & set(os.listdir(pred_dir))))

        # compute metrics for each sequence
        for seq in sequence_list:
            # setup paths
            true_seq_dir = osp.join(true_dir, seq)
            pred_seq_dir = osp.join(pred_dir, seq)

            self.compute_sequence_metrics(seq, true_seq_dir, pred_seq_dir)

    def compute_sequence_metrics(self, seq, true_seq_dir, pred_seq_dir,
                                 pred_seq=None):

        # clear
        self.reset_per_sequence()

        # initialize metric_dict for the current sequence
        self.seq_idx_curr = seq
        self.metric_dict[self.seq_idx_curr] = OrderedDict({
            metric: [] for metric in self.metric_opt.keys()})

        # retrieve files
        true_img_lst = base_utils.retrieve_files(true_seq_dir, 'png')
        pred_img_lst = base_utils.retrieve_files(pred_seq_dir, 'png')

        # compute metrics for each frame
        for i in range(len(true_img_lst)):
            self.true_img_cur = cv2.imread(true_img_lst[i])[..., ::-1]  # bgr2rgb
            # use a given pred_seq or load from disk
            if pred_seq is not None:
                self.pred_img_cur = pred_seq[i]  # hwc|rgb|uint8
            else:
                self.pred_img_cur = cv2.imread(pred_img_lst[i])[..., ::-1]

            # pred_img and true_img may have different sizes
            # crop the larger one to match the smaller one
            true_h, true_w = self.true_img_cur.shape[:-1]
            pred_h, pred_w = self.pred_img_cur.shape[:-1]
            min_h, min_w = min(true_h, pred_h), min(true_w, pred_w)
            self.true_img_cur = self.true_img_cur[:min_h, :min_w, :]
            self.pred_img_cur = self.pred_img_cur[:min_h, :min_w, :]

            # compute metrics for the current frame
            self.compute_frame_metrics()

            # update
            self.true_img_pre = self.true_img_cur
            self.pred_img_pre = self.pred_img_cur

    def compute_frame_metrics(self):
        metric_dict = self.metric_dict[self.seq_idx_curr]

        # compute evaluation
        for metric_type, opt in self.metric_opt.items():
            if metric_type == 'PSNR':
                PSNR = self.compute_PSNR()
                metric_dict['PSNR'].append(PSNR)

            elif metric_type == 'LPIPS':
                LPIPS = self.compute_LPIPS()[0, 0, 0, 0].cpu().numpy()
                metric_dict['LPIPS'].append(LPIPS)

            elif metric_type == 'tOF':
                # skip the first frame
                if self.pred_img_pre is not None:
                    tOF = self.compute_tOF()
                    metric_dict['tOF'].append(tOF)

    def compute_PSNR(self):
        if self.psnr_colorspace == 'rgb':
            true_img = self.true_img_cur
            pred_img = self.pred_img_cur
        else:
            # convert to ycbcr, and keep the y channel
            true_img = data_utils.rgb_to_ycbcr(self.true_img_cur)[..., 0]
            pred_img = data_utils.rgb_to_ycbcr(self.pred_img_cur)[..., 0]

        diff = true_img.astype(np.float64) - pred_img.astype(np.float64)
        RMSE = np.sqrt(np.mean(np.power(diff, 2)))

        if RMSE == 0:
            return np.inf

        PSNR = 20 * np.log10(255.0 / RMSE)
        return PSNR

    def compute_LPIPS(self):
        true_img = np.ascontiguousarray(self.true_img_cur)
        pred_img = np.ascontiguousarray(self.pred_img_cur)

        # to tensor
        true_img = torch.FloatTensor(true_img).unsqueeze(0).permute(0, 3, 1, 2)
        pred_img = torch.FloatTensor(pred_img).unsqueeze(0).permute(0, 3, 1, 2)

        # normalize to [-1, 1]
        true_img = true_img.to(self.device) * 2.0 / 255.0 - 1.0
        pred_img = pred_img.to(self.device) * 2.0 / 255.0 - 1.0

        with torch.no_grad():
            LPIPS = self.dm.forward(true_img, pred_img)

        return LPIPS

    def compute_tOF(self):
        true_img_cur = cv2.cvtColor(self.true_img_cur, cv2.COLOR_RGB2GRAY)
        pred_img_cur = cv2.cvtColor(self.pred_img_cur, cv2.COLOR_RGB2GRAY)
        true_img_pre = cv2.cvtColor(self.true_img_pre, cv2.COLOR_RGB2GRAY)
        pred_img_pre = cv2.cvtColor(self.pred_img_pre, cv2.COLOR_RGB2GRAY)

        # forward flow
        true_OF = cv2.calcOpticalFlowFarneback(
            true_img_pre, true_img_cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        pred_OF = cv2.calcOpticalFlowFarneback(
            pred_img_pre, pred_img_cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # EPE
        diff_OF = true_OF - pred_OF
        tOF = np.mean(np.sqrt(np.sum(diff_OF**2, axis=-1)))

        return tOF

