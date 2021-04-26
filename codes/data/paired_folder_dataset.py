import os
import os.path as osp

import cv2
import numpy as np
import torch

from .base_dataset import BaseDataset
from utils.base_utils import retrieve_files


class PairedFolderDataset(BaseDataset):
    def __init__(self, data_opt, **kwargs):
        """ Folder dataset with paired data
            support both BI & BD degradation
        """
        super(PairedFolderDataset, self).__init__(data_opt, **kwargs)

        # get keys
        gt_keys = sorted(os.listdir(self.gt_seq_dir))
        lr_keys = sorted(os.listdir(self.lr_seq_dir))
        self.keys = sorted(list(set(gt_keys) & set(lr_keys)))

        # filter keys
        if self.filter_file:
            with open(self.filter_file, 'r') as f:
                sel_keys = { line.strip() for line in f }
                self.keys = sorted(list(sel_keys & set(self.keys)))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        key = self.keys[item]

        # load gt frames
        gt_seq = []
        for frm_path in retrieve_files(osp.join(self.gt_seq_dir, key)):
            frm = cv2.imread(frm_path)[..., ::-1]
            gt_seq.append(frm)
        gt_seq = np.stack(gt_seq)  # thwc|rgb|uint8

        # load lr frames
        lr_seq = []
        for frm_path in retrieve_files(osp.join(self.lr_seq_dir, key)):
            frm = cv2.imread(frm_path)[..., ::-1].astype(np.float32) / 255.0
            lr_seq.append(frm)
        lr_seq = np.stack(lr_seq)  # thwc|rgb|float32

        # convert to tensor
        gt_tsr = torch.from_numpy(np.ascontiguousarray(gt_seq))  # uint8
        lr_tsr = torch.from_numpy(np.ascontiguousarray(lr_seq))  # float32

        # gt: thwc|rgb||uint8 | lr: thwc|rgb|float32
        return {
            'gt': gt_tsr,
            'lr': lr_tsr,
            'seq_idx': key,
            'frm_idx': sorted(os.listdir(osp.join(self.gt_seq_dir, key)))
        }
