import os
import os.path as osp
import argparse
import glob
import lmdb
import pickle
import random

import cv2
import numpy as np


def create_lmdb(dataset, raw_dir, lmdb_dir, filter_file=''):
    assert dataset in ('VimeoTecoGAN', 'VimeoTecoGAN-sub'), \
        'Unknown Dataset: {}'.format(dataset)
    print('Creating lmdb dataset: {}'.format(dataset))

    # retrieve sequences
    if filter_file:  # dump selective data into lmdb
        with open(filter_file, 'r') as f:
            seq_dir_lst = sorted([line.strip() for line in f])
    else:
        seq_dir_lst = sorted(os.listdir(raw_dir))
    print('Number of sequences: {}'.format(len(seq_dir_lst)))

    # compute space to allocate
    print('Calculating space needed for LMDB generation ... ', end='')
    nbytes = 0
    for seq_dir in seq_dir_lst:
        frm_path_lst = sorted(glob.glob(osp.join(raw_dir, seq_dir, '*.png')))
        nbytes_per_frm = cv2.imread(frm_path_lst[0], cv2.IMREAD_UNCHANGED).nbytes
        nbytes += len(frm_path_lst)*nbytes_per_frm
    alloc_size = round(1.5*nbytes)
    print('{:.2f} GB'.format(alloc_size / (1 << 30)))

    # create lmdb environment
    env = lmdb.open(lmdb_dir, map_size=alloc_size)

    # write data to lmdb
    commit_freq = 5
    keys = []
    txn = env.begin(write=True)
    for b, seq_dir in enumerate(seq_dir_lst):
        # log
        print('Processing {} ({}/{})\r'.format(
            seq_dir, b, len(seq_dir_lst)), end='')

        # setup
        frm_path_lst = sorted(glob.glob(osp.join(raw_dir, seq_dir, '*.png')))
        n_frm = len(frm_path_lst)

        # read frames
        for i in range(n_frm):

            frm = cv2.imread(frm_path_lst[i], cv2.IMREAD_UNCHANGED)
            frm = np.ascontiguousarray(frm[..., ::-1])  # hwc|rgb|uint8

            h, w, c = frm.shape
            key = '{}_{}x{}x{}_{:04d}'.format(seq_dir, n_frm, h, w, i)
            txn.put(key.encode('ascii'), frm)
            keys.append(key)

        # commit
        if b % commit_freq == 0:
            txn.commit()
            txn = env.begin(write=True)

    txn.commit()
    env.close()

    # create meta information
    meta_info = {
        'name': dataset,
        'keys': keys
    }
    pickle.dump(meta_info, open(osp.join(lmdb_dir, 'meta_info.pkl'), 'wb'))


def check_lmdb(dataset, lmdb_dir, display=True):

    def visualize(win, img):
        if display:
            cv2.namedWindow(win, 0)
            cv2.resizeWindow(win, img.shape[-2], img.shape[-3])
            cv2.imshow(win, img[..., ::-1])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            cv2.imwrite('_'.join(win.split('/')) + '.png', img[..., ::-1])

    assert dataset in ('VimeoTecoGAN', 'VimeoTecoGAN-sub'), \
        'Unknown Dataset: {}'.format(dataset)
    print('Checking lmdb dataset: {}'.format(dataset))

    # load keys
    meta_info = pickle.load(open(osp.join(lmdb_dir, 'meta_info.pkl'), 'rb'))
    keys = meta_info['keys']
    print('Number of keys: {}'.format(len(keys)))

    # randomly select frame to visualize
    with lmdb.open(lmdb_dir) as env:
        for i in range(3):  # replace to whatever you want
            idx = random.randint(0, len(keys) - 1)
            key = keys[idx]

            # parse key
            key_lst = key.split('_')
            vid, sz, frm = '_'.join(key_lst[:-2]), key_lst[-2], key_lst[-1]
            sz = tuple(map(int, sz.split('x')))
            sz = (*sz[1:], 3)
            print('video index: {} | size: {} | # of frame: {}'.format(
                vid, sz, frm))

            with env.begin() as txn:
                buf = txn.get(key.encode('ascii'))
                val = np.frombuffer(buf, dtype=np.uint8).reshape(*sz) # HWC

            visualize(key, val)


if __name__ == '__main__':
    # parse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='VimeoTecoGAN or VimeoTecoGAN-sub')
    parser.add_argument('--data_type', type=str, required=True,
                        help='GT or Bicubic4xLR')
    parser.add_argument('--ln', type=str, required=False,
                        help='ln dir')
    args = parser.parse_args()

    # setup
    raw_dir = 'data/{}/{}'.format(args.dataset, args.data_type)
    lmdb_dir = 'data/{}/{}.lmdb'.format(args.dataset, args.data_type)
    filter_file = ''

    # run
    if osp.exists(lmdb_dir):
        print('Dataset [{}] already exists'.format(args.dataset))
        print('Checking the LMDB dataset ...')
        check_lmdb(args.dataset, lmdb_dir)
    else:
        create_lmdb(args.dataset, raw_dir, lmdb_dir, filter_file)

        print('Checking the LMDB dataset ...')
        check_lmdb(args.dataset, lmdb_dir)

