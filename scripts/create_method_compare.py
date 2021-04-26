import os
from PIL import Image, ImageDraw, ImageFont

#####################################################
# parameter setting                                 #
#####################################################
row            = 2
col            = 3
pad            = 2

# Dataset = ['Vid4', 'Gvt72', 'Tos3']
roi_h = roi_w = 0

ROI = [
# walk 11
# [(210, 153), (375, 199)],
# [(446, 122), (611, 168)],
[(136, 196), (266, 233)],
[(414, 364), (649, 430)]
# foliage 20
# [(515, 269), (711, 324)],
# [(152, 137), (325, 186)]
# city 20
# [(457, 158), (609, 201)],
# [(234, 434), (458, 497)]
# calendar 20
# [(162, 213), (283, 247)],
# [(368, 451), (582, 511)]
]


group = len(ROI)
for roi in ROI:
    point1 = roi[0]
    point2 = roi[1]
    w = abs(point1[0] - point2[0])
    h = abs(point1[1] - point2[1])
    if h > roi_h:
        roi_h, roi_w = h, w


dataset = 'Vid4'
name = 'walk'
num = 11
GT_root_path = '../data/'
Rst_root_path = '../results/'
Rst = ['vespcn_ep0500', 'SOFVSR_x4', 'DUF-16L', 'TecoGAN_BD_iter500000']

gt_img_path = os.path.join(GT_root_path, dataset, 'GT', name, '00{}.png'.format(num))
lr_img_path = os.path.join(GT_root_path, dataset, 'Gaussian4xLR', name, '00{}.png'.format(num))

img_gt = Image.open(gt_img_path)
img_lr = Image.open(lr_img_path)


file_ls = os.listdir(os.path.join(Rst_root_path, dataset))       # list all files in this folder
print(file_ls)
print(roi_h, roi_w, roi_h/roi_w)

rst_path = '../results/show'

dest_im = Image.new('RGBA', (5 * roi_w + pad * 3, 4 * roi_h + pad * 3), (255, 255, 255))
label = ImageDraw.ImageDraw(dest_im)

src_im = Image.open(gt_img_path)
draw = ImageDraw.ImageDraw(src_im)
for rect in ROI:
    draw.rectangle(rect, fill=None, outline='red', width=4)
src_wh = src_im.size
img_h = roi_h * 4 + pad * 3
img_size = (round(img_h * (src_wh[0] / src_wh[1])), img_h)
img_resi = src_im.resize(size=img_size)
dest_im.paste(img_resi, (0, 0))  # paste to dest_im

label_name = ['LR', 'VESPCN', 'SOFVSR', 'DUF', 'Ours', 'GT']
font = ImageFont.truetype(font='../resources/VCR_OSD_MONO_1.001.ttf', size=15)


for i in range(row*col):
    box = (img_size[0] + pad + i % 3 * (roi_w + pad), i // 3 * (roi_h + pad))
    pos = (ROI[0][0][0], ROI[0][0][1], ROI[0][1][0], ROI[0][1][1])
    if i == 0:
        src_im = Image.open(lr_img_path)
        img_resi = src_im.resize(size=src_wh)
        img_crop = img_resi.crop(pos)
        img_resi = img_crop.resize(size=(roi_w, roi_h))
        Rst_num = 0
    elif i == 5:
        src_im = Image.open(gt_img_path)
        img_crop = src_im.crop(pos)
        img_resi = img_crop.resize(size=(roi_w, roi_h))
    else:
        try:
            src_path = os.path.join(Rst_root_path, dataset, Rst[Rst_num], name, '00{}.png'.format(num))
            src_im = Image.open(src_path)
        except:
            src_path = os.path.join(Rst_root_path, dataset, Rst[Rst_num], name, 'Frame0{}.png'.format(num))
            src_im = Image.open(src_path)
        img_crop = src_im.crop(pos)
        img_resi = img_crop.resize(size=(roi_w, roi_h))
        Rst_num += 1
    dest_im.paste(img_resi, box)  # paste to dest_im
    font_color = 'white' if i != 4 else 'red'
    label.text(xy=box, text=label_name[i], fill=font_color, font=font)

for i in range(row*col):
    box = (img_size[0] + pad + (i + 6) % 3 * (roi_w + pad), (i + 6) // 3 * (roi_h + pad))
    pos = (ROI[1][0][0], ROI[1][0][1], ROI[1][1][0], ROI[1][1][1])
    if i == 0:
        src_im = Image.open(lr_img_path)
        img_resi = src_im.resize(size=src_wh)
        img_crop = img_resi.crop(pos)
        img_resi = img_crop.resize(size=(roi_w, roi_h))
        Rst_num = 0
    elif i == 5:
        src_im = Image.open(gt_img_path)
        img_crop = src_im.crop(pos)
        img_resi = img_crop.resize(size=(roi_w, roi_h))
    else:
        try:
            src_path = os.path.join(Rst_root_path, dataset, Rst[Rst_num], name, '00{}.png'.format(num))
            src_im = Image.open(src_path)
        except:
            src_path = os.path.join(Rst_root_path, dataset, Rst[Rst_num], name, 'Frame0{}.png'.format(num))
            src_im = Image.open(src_path)
        img_crop = src_im.crop(pos)
        img_resi = img_crop.resize(size=(roi_w, roi_h))
        Rst_num += 1
    dest_im.paste(img_resi, box)  # paste to dest_im
    font_color = 'white' if i != 4 else 'red'
    label.text(xy=box, text=label_name[i], fill=font_color, font=font)


window = (0, 0, img_size[0] + col * (roi_w + pad), img_size[1])
rst_im = dest_im.crop(window)

dest_im_path = os.path.join(rst_path, 'cmp_{}_{}_{}.png'.format(dataset, name, num))
rst_im.save(dest_im_path, 'png')
rst_im.show()  # finish