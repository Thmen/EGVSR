import os
from PIL import Image, ImageDraw, ImageFont

#####################################################
# parameter setting                                 #
#####################################################
row            = 2
col            = 6
pad            = 2

dataset = 'ToS3'
seq_num = 3
name = ['bridge', 'face', 'room']
frm_num = [13945, 9945, 4400]
ROI = [
# bridge
[[(467, 136), (555, 202)],
[(987, 374), (1163, 506)]],
# face
[[(643, 127), (1031, 418)],
[(422, 144), (589, 269)]],
# room
[[(622, 346), (820, 494)],
[(75, 364), (280, 518)]]
]

GT_root_path = '../data/'
Rst_root_path = '../results/'
rst_path = '../results/show'
Rst = ['vespcn_ep0500', 'SOFVSR_x4', 'FRVSR_BD_iter400000', 'TecoGAN_BD_iter500000', 'EGVSR_iter420000']
label_name = ['VESPCN', 'SOFVSR', 'FRVSR', 'TecoGAN', 'Ours', 'GT']
font = ImageFont.truetype(font='../resources/VCR_OSD_MONO_1.001.ttf', size=20)

img_h, img_w = 534//4, 1280//4

input_im = Image.new('RGBA', (seq_num * (img_w + pad) - pad, img_h), (255, 255, 255))
label_n = ImageDraw.ImageDraw(input_im)

for i in range(seq_num):
    lr_img_path = os.path.join(GT_root_path, dataset, 'Gaussian4xLR', name[i], '{}.png'.format(frm_num[i]))
    img_lr = Image.open(lr_img_path)
    pos = (i * (img_w + pad), 0)
    draw = ImageDraw.ImageDraw(img_lr)
    for rect in ROI[i]:
        rect = [(rect[0][0]//4, rect[0][1]//4), (rect[1][0]//4, rect[1][1]//4)]
        draw.rectangle(rect, fill=None, outline='red', width=2)
    input_im.paste(img_lr, pos)  # paste to input_im
    label_n.text(xy=pos, text=name[i], fill='white', font=font)

input_im_path = os.path.join(rst_path, 'input_img_{}.png'.format(dataset))
input_im.save(input_im_path, 'png')
input_im.show()  # finish

for i in range(seq_num):

    roi_h = roi_w = 0
    _ROI = ROI[i]
    for roi in _ROI:
        point1 = roi[0]
        point2 = roi[1]
        w = abs(point1[0] - point2[0])
        h = abs(point1[1] - point2[1])
        if h > roi_h:
            roi_h, roi_w = h, w

    dest_im = Image.new('RGBA', (col * (roi_w + pad) - pad, row * roi_h + pad), (255, 255, 255))
    label = ImageDraw.ImageDraw(dest_im)

    gt_img_path = os.path.join(GT_root_path, dataset, 'GT', name[i], '{}.png'.format(frm_num[i]))
    img_gt = Image.open(gt_img_path)

    print(roi_h, roi_w, roi_h/roi_w)

    for r in range(row):
        for c in range(col):
            pos = (c * (roi_w + pad), r * (roi_h + pad))
            box = (_ROI[r][0][0], _ROI[r][0][1], _ROI[r][1][0], _ROI[r][1][1])
            if c == col-1:
                src_im = img_gt
            else:
                src_path = os.path.join(Rst_root_path, dataset, Rst[c], name[i], '{}.png'.format(frm_num[i]))
                src_im = Image.open(src_path)
            img_crop = src_im.crop(box)
            img_resi = img_crop.resize(size=(roi_w, roi_h))
            dest_im.paste(img_resi, pos)  # paste to dest_im
            font_color = 'white' if c != 4 else 'red'
            label.text(xy=pos, text=label_name[c], fill=font_color, font=font)

    dest_im_path = os.path.join(rst_path, 'cmp_{}_{}_{}.png'.format(dataset, name[i], frm_num[i]))
    dest_im.save(dest_im_path, 'png')
    dest_im.show()  # finish