import os
from PIL import Image, ImageDraw, ImageFont

#####################################################
# parameter setting                                 #
#####################################################
row            = 9
col            = 6
pad            = 2

dataset = 'Gvt72'

name = ['0098', '0098', '0010', '0083', '07411', '0903', '0394', '0858', '0499']
seq_num = len(name)
frm_num = 6
ROI = [
[(59, 3), (119, 63)],
[(262, 14), (326, 78)],
[(216, 127), (337, 248)],
[(208, 10), (341, 143)],
[(117, 6), (315, 204)],
[(4, 35), (137, 168)],
[(188, 128), (265, 205)],
[(240, 43), (345, 148)],
[(9, 139), (116, 246)]
]

GT_root_path = '../data/'
Rst_root_path = '../results/'
rst_path = '../results/show'
Rst = ['vespcn_ep0500', 'SOFVSR_x4', 'FRVSR_BD_iter400000', 'TecoGAN_BD_iter500000', 'EGVSR_iter420000']
label_name = ['VESPCN', 'SOFVSR', 'FRVSR', 'TecoGAN', 'Ours', 'GT']
font = ImageFont.truetype(font='../resources/VCR_OSD_MONO_1.001.ttf', size=25)

roi_h = roi_w = 0
for roi in ROI:
    point1 = roi[0]
    point2 = roi[1]
    w = abs(point1[0] - point2[0])
    h = abs(point1[1] - point2[1])
    if h > roi_h:
        roi_h, roi_w = h, w
dest_im = Image.new('RGBA', (col * (roi_w + pad) - pad, row * (roi_h + pad) -pad), (255, 255, 255))
label = ImageDraw.ImageDraw(dest_im)

print(roi_h, roi_w)

for r in range(row):
    for c in range(col):
        pos = (c * (roi_w + pad), r * (roi_h + pad))
        box = (ROI[r][0][0], ROI[r][0][1], ROI[r][1][0], ROI[r][1][1])
        if c == col-1:
            gt_img_path = os.path.join(GT_root_path, dataset, 'GT', name[r], 'im{}.png'.format(frm_num))
            src_im = Image.open(gt_img_path)
        else:
            src_path = os.path.join(Rst_root_path, dataset, Rst[c], name[r], 'im{}.png'.format(frm_num))
            src_im = Image.open(src_path)
        img_crop = src_im.crop(box)
        img_resi = img_crop.resize(size=(roi_w, roi_h))
        dest_im.paste(img_resi, pos)  # paste to dest_im
        font_color = 'white' if c != 4 else 'red'
        label.text(xy=pos, text=label_name[c], fill=font_color, font=font)

dest_im_path = os.path.join(rst_path, 'cmp_{}.png'.format(dataset))
dest_im.save(dest_im_path, 'png')
dest_im.show()  # finish