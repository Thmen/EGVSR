import os.path as osp
import pickle
import random

import numpy as np
import torch

from .base_dataset import BaseDataset


class UnpairedLMDBDataset(BaseDataset):
    def __init__(self, data_opt, **kwargs):
        """ LMDB dataset with unpaired data, for BD degradation
        """
        super(UnpairedLMDBDataset, self).__init__(data_opt, **kwargs)

        # load meta info
        meta = pickle.load(
            open(osp.join(self.seq_dir, 'meta_info.pkl'), 'rb'))
        self.keys = sorted(meta['keys'])

        # use partial videos
        if self.filter_file is not None:
            with open(self.filter_file, 'r') as f:
                sel_seqs = { line.strip() for line in f }
            self.keys = list(filter(
                lambda x: self.parse_lmdb_key(x)[0] in sel_seqs, self.keys))

        # register parameters
        self.env = None

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        if self.env is None:
            self.env = self.init_lmdb(self.seq_dir)

        # parse info
        key = self.keys[item]
        idx, (tot_frm, h, w), cur_frm = self.parse_lmdb_key(key)
        c = 3 if self.data_type.lower() == 'rgb' else 1

        # get frames
        frms = []
        if self.moving_first_frame and (random.uniform(0, 1) > self.moving_factor):
            # load data
            frm = self.read_lmdb_frame(self.env, key, size=(h, w, c))
            frm = frm.transpose(2, 0, 1)  # chw|rgb|uint8

            # generate random moving parameters
            offsets = np.floor(
                np.random.uniform(-3.5, 4.5, size=(self.tempo_extent, 2)))
            offsets = offsets.astype(np.int32)
            pos = np.cumsum(offsets, axis=0)
            min_pos = np.min(pos, axis=0)
            topleft_pos = pos - min_pos
            range_pos = np.max(pos, axis=0) - min_pos
            c_h, c_w = h - range_pos[0], w - range_pos[1]

            # generate frames
            for i in range(self.tempo_extent):
                top, left = topleft_pos[i]
                frms.append(frm[:, top: top + c_h, left: left + c_w].copy())
        else:
            # read frames
            for i in range(cur_frm, cur_frm + self.tempo_extent):
                if i >= tot_frm:
                    # reflect temporal paddding, e.g., (0,1,2) -> (0,1,2,1,0)
                    key = '{}_{}x{}x{}_{:04d}'.format(
                        idx, tot_frm, h, w, 2 * tot_frm - i - 2)
                else:
                    key = '{}_{}x{}x{}_{:04d}'.format(
                        idx, tot_frm, h, w, i)

                frm = self.read_lmdb_frame(self.env, key, size=(h, w, c))
                frm = frm.transpose(2, 0, 1)  # chw|rgb|uint8
                frms.append(frm)

        frms = np.stack(frms)  # tchw|rgb|uint8

        # crop randomly
        pats = self.crop_sequence(frms)

        # augment patches
        pats = self.augment_sequence(pats)

        # convert to tensor and normalize to range [0, 1]
        tsr = torch.FloatTensor(np.ascontiguousarray(pats)) / 255.0

        # tchw|rgb|float32
        return {'gt': tsr}

    def crop_sequence(self, frms):
        csz = self.crop_size

        h, w = frms.shape[-2:]
        assert (csz <= h) and (csz <= w), \
            'the crop size is larger than the image size'

        # crop
        top = random.randint(0, h - csz)
        left = random.randint(0, w - csz)
        pats = frms[..., top: top + csz, left: left + csz]

        return pats

    @staticmethod
    def augment_sequence(pats):
        # flip spatially
        axis = random.randint(1, 3)
        if axis > 1:
            pats = np.flip(pats, axis)

        # flip temporally
        axis = random.randint(0, 1)
        if axis < 1:
            pats = np.flip(pats, axis)

        # rotate
        k = random.randint(0, 3)
        pats = np.rot90(pats, k, (2, 3))

        return pats
