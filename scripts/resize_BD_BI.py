import os
import os.path as osp
import glob
from multiprocessing import Pool

import numpy as np
import cv2


# setup params
# default settings
scale = 4  # downsampling scale
sigma = 1.5
ksize = 1 + 2 * int(sigma * 3.0)
# to be modified

dataset = 'Gvt72'
degrade_mode = 'BD' # 'BD' or 'BI'

n_process = 16  # the number of process to be used for downsampling
filepaths = glob.glob('../data/{}/GT/*/*.png'.format(dataset))
gt_dir_idx = 'GT'
lr_dir_idx = 'Gaussian4xLR' if degrade_mode=='BD' else 'Bicubic4xLR'


def downsample_worker(filepath, mode='BD'):
    # log
    print('Mode:{}, Processing {}'.format(mode, filepath))

    # setup dirs
    gt_folder, img_idx = osp.split(filepath)
    lr_folder = gt_folder.replace(gt_dir_idx, lr_dir_idx)

    # read image
    img = cv2.imread(filepath)  # hwc|bgr|uint8

    # dowmsample
    if mode == 'BI':
        img_lr = cv2.resize(img, dsize=(0,0), fx=1/scale, fy=1/scale, interpolation=cv2.INTER_CUBIC)

    else: # mode=='BD / default
        img_blur = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma)
        img_lr = img_blur[::scale, ::scale, ::]

    # save image
    cv2.imwrite(osp.join(lr_folder, img_idx), img_lr)


if __name__ == '__main__':
    # setup dirs
    print('# of images: {}'.format(len(filepaths)))
    for filepath in filepaths:
        gt_folder, _ = osp.split(filepath)
        lr_folder = gt_folder.replace(gt_dir_idx, lr_dir_idx)
        if not osp.exists(lr_folder):
            os.makedirs(lr_folder)

    # for filepath in sorted(filepaths):
    #     downsample_worker(filepath, degrade_mode)

    # for each image
    pool = Pool(n_process)
    for filepath in sorted(filepaths):
        pool.apply_async(downsample_worker, args=(filepath, degrade_mode))
    pool.close()
    pool.join()

