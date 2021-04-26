import os, glob
import imageio
import cv2
import numpy as np
from tqdm import trange

dataset = 'city'
LR_DIR = '../data/Vid4/Gaussian4xLR/'
HR_DIR = '../results/Vid4/G_iter420000/'
MIX_DIR = '../results'


def generate_mask(h, w, split_rate: float):
    """
    :param h: mask height
    :param w: mask width
    :param split_rate: 0.0 - 1.0
    :return:    mask zero_one matirx
                bottom-left corner is zero (split_rate*100% percent)
                top-right corner is one ((1-split_rate)*100% percent)
    """
    shape = (h, w)
    mask = np.zeros(shape, dtype=np.uint8)

    start = round(split_rate * (w + h) - h)
    for i in range(start, w):
        tmp = np.eye(*shape, k=i, dtype=np.uint8) * 255
        mask = np.bitwise_or(mask, tmp)

    return mask


def generate_image(lr, sr, mix_path):

    lr_imgs = sorted(glob.glob('{}/*'.format(lr)))
    sr_imgs = sorted(glob.glob('{}/*'.format(sr)))

    frame_num = min(len(lr_imgs), len(sr_imgs))

    for n in trange(frame_num):

        sr_img = cv2.imread(sr_imgs[n])
        sr_h, sr_w, sr_c = sr_img.shape

        lr_img = cv2.imread(lr_imgs[n])
        hr_img = cv2.resize(lr_img, (sr_w, sr_h), interpolation=cv2.INTER_CUBIC)

        split_rate = n / frame_num
        mask = generate_mask(sr_h, sr_w, split_rate)
        mask_not = np.bitwise_not(mask)

        hr_img_mask = cv2.bitwise_and(hr_img, hr_img, mask=mask)
        sr_img_mask = cv2.bitwise_and(sr_img, sr_img, mask=mask_not)

        mix_img = cv2.add(hr_img_mask, sr_img_mask)

        h_add_w = (sr_h + sr_w) * split_rate
        cv2.line(mix_img, (0, round(sr_h - h_add_w)), (round(h_add_w), sr_h), color= (255, 255, 255), thickness = 2)
        cv2.putText(mix_img, 'VSR x4', (10, sr_h - 10), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(mix_img, 'Upscale x4', (sr_w - 200, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 2)

        mix_img_name = "{:0>4d}.png".format(n)
        cv2.imwrite(os.path.join(mix_path, mix_img_name), mix_img)

    return

def create_gif(image_list, gif_name, dur):

    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    # Save them as frames into a gif
    imageio.mimsave(gif_name, frames, 'GIF', duration=dur)
    return


if __name__ == '__main__':

    lr_path = os.path.join(LR_DIR, dataset)
    sr_path = os.path.join(HR_DIR, dataset)

    print(lr_path)

    mix_path = os.path.join(MIX_DIR, dataset)

    if not os.path.exists(mix_path):
        os.mkdir(mix_path)

    generate_image(lr_path, sr_path, mix_path)

    image_path = os.path.join(MIX_DIR, dataset, '*')

    image_list = sorted(glob.glob(image_path))

    create_gif(image_list, '{}.gif'.format(dataset), dur=0.1)

    print("finish")
