# !/bin/python3
import os, glob, sys
import imageio
import cv2

print(os.getcwd())

TIME_GAP = 0.075  # 两帧之间的时间间隔，秒为单位
FILE_PATH = "../results/Vid4/TecoGAN_BD_iter500000/calendar/*"
BEGIN_INDEX = 1
END_INDEX = 7
FORMAT = ".jpg"

imgs = glob.glob(FILE_PATH)
imgs.sort()

# 返回gif
def create_gif(image_list, gif_name):
    print(image_list)

    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
        # Save them as frames into a gif
    imageio.mimsave(gif_name, frames, 'GIF', duration=TIME_GAP)
    return



if __name__ == '__main__':
    create_gif(imgs, 'created_gif.gif')

