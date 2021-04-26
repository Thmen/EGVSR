import sys, os, shutil, math
from PIL import Image, ImageDraw, ImageFont
import imageio
#####################################################
# parameter setting                                 #
#####################################################
bol_auto_place = False                     # auto place the image as a squared image， if 'True', ignore var 'row' and 'col' below
row            = 8                         # row number which means col number images per row
col            = 9                         # col number which means row number images per col
seq            = 7                         # image sequence frame number
nw             = 448 // 3                  # sub image size, nw x nh
nh             = 256 // 3
pad            = 2


father_path = '../data/Gvt72/GT'
file_ls = os.listdir(father_path)       # list all files in this folder
rst_path = '../results/show'
dest_im = Image.new('RGBA', (col * (nw+pad) -pad, row * (nh+pad) -pad), (255, 255, 255))    # the image size of splicing image, background color is white

for i in range(seq):
    for n, file in enumerate(file_ls):      # loop place the sub image
        img_path = os.path.join(father_path, file, 'im{}.png'.format(i+1))
        src_im = Image.open(img_path)  # open files in order
        resize_im = src_im.resize(size=(nw, nh))
        dest_im.paste(resize_im, (n%9 * (nw+pad), n//9 * (nh+pad)))  # paste to dest_im
    dest_im_path = os.path.join(rst_path, 'splicing_{}.png'.format(i+1))
    dest_im.save(dest_im_path, 'png')


for n, file in enumerate(file_ls):      # loop place the sub image
    img_path = os.path.join(father_path, file, 'im1.png')
    src_im = Image.open(img_path)  # open files in order
    resize_im = src_im.resize(size=(nw, nh))
    pos = (n%9 * (nw+pad), n//9 * (nh+pad))
    dest_im.paste(resize_im, pos)  # paste to dest_im
    # 添加文字
    draw = ImageDraw.Draw(dest_im)
    # font = ImageFont.truetype(size=10)
    # 参数：位置、文本、填充、字体
    draw.text(xy=pos, text=file, fill=(255, 255, 255))
dest_im_path = os.path.join(rst_path, 'splicing_name.png')
dest_im.save(dest_im_path, 'png')
dest_im.show()  # finish


# create gif
frames = []
for i in range(seq):
    frm_name = os.path.join(rst_path, 'splicing_{}.png'.format(i + 1))
    frm_im = imageio.imread(frm_name)  # open files in order
    frames.append(frm_im)
# Save them as frames into a gif
frames_path = os.path.join(rst_path, 'splicing.gif')
imageio.mimsave(frames_path, frames, 'GIF', duration=0.1)

