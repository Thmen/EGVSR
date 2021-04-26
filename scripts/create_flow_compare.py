import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

#####################################################
# parameter setting                                 #
#####################################################
row            = 4
col            = 5
pad            = 2

dataset = 'Vid4'

name = 'walk'
frm = 5
frm_num = [4, 5, 6, 7, 8, 9]
img_w, img_h = 704, 576

GT_root_path = '../data/'
rst_path = '../results/show'
Rst = ['vespcn_ep0500', 'SOFVSR_x4', 'FRVSR_BD_iter400000', 'TecoGAN_BD_iter500000', 'EGVSR_iter420000']
label_name = ['VESPCN', 'SOFVSR', 'FRVSR', 'TecoGAN', 'Ours', 'GT']

imgs_org = []
imgs_hr = []
imgs_flow = []
imgs_warp = []
imgs_without_mc = []
imgs_with_mc = []

for f in frm_num:
    img_org_path = os.path.join(GT_root_path, dataset, 'GT', name, '000{}.png'.format(f))
    img_org = Image.open(img_org_path)
    imgs_org.append(img_org)

imgs_hr = imgs_org[1:]

for f in frm_num[1:]:
    img_flow_path = os.path.join('../results/TestFlow/TecoGAN_BD_Flow', name, '000{}.png'.format(f))
    img_flow = Image.open(img_flow_path)
    imgs_flow.append(img_flow)

for f in frm_num[1:]:
    img_warp_path = os.path.join('../results/TestFlow/TecoGAN_BD_Warp', name, '000{}.png'.format(f))
    img_warp = Image.open(img_warp_path)
    imgs_warp.append(img_warp)

for i in range(frm):
    img_a = np.array(imgs_org[i])
    img_b = np.array(imgs_hr[i])
    img_c = np.array(imgs_warp[i])
    img_sub1 = abs(img_b - img_a)*0.8
    img1 = Image.fromarray(img_sub1.astype('uint8')).convert('L')
    img_sub2 = abs(img_c - img_b)
    img2 = Image.fromarray(img_sub2.astype('uint8')).convert('L')
    img2 = img2.filter(ImageFilter.SHARPEN)
    imgs_without_mc.append(img1)
    imgs_with_mc.append(img2)

imgs_sum = [imgs_hr, imgs_flow, imgs_without_mc, imgs_with_mc]

dest_im = Image.new('RGBA', (col * (img_w + pad) - pad, row * (img_h + pad) - pad), (255, 255, 255))

for r in range(row):
    for c in range(col):
        pos = (c * (img_w + pad), r * (img_h + pad))
        src_img = imgs_sum[r][c]
        img_resi = src_img.resize(size=(img_w, img_h))
        dest_im.paste(img_resi, pos)  # paste to dest_im


dest_im_path = os.path.join(rst_path, 'seq_flow_{}.png'.format(name))
dest_im.save(dest_im_path, 'png')
dest_im.show()  # finish