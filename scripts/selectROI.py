import os
import cv2

#####################################################
# parameter setting                                 #
#####################################################
group          = 2
scale          = 9/16 / group

Dataset = ['Vid4', 'Gvt72', 'ToS3']

GT_root_path = '../data/'
Rst_root_path = '../results/'

dataset = 'Vid4'
name = 'walk'
frm_num = 18
# bridge face room
gt_img_path = os.path.join(GT_root_path, dataset, 'GT', name, '00{}.png'.format(frm_num))

img = cv2.imread(gt_img_path)

global point1, point2

def on_mouse(event, x, y, flags, img):
    global point1, point2
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        point1 = (x, y)
        cv2.circle(img2, point1, 10, (0, 255, 0), 5)
        cv2.imshow('setect ROI', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
        tmp = (x, round((x-point1[0]) * scale + point1[1]))
        cv2.rectangle(img2, point1, tmp, (255, 0, 0), 5)
        cv2.imshow('setect ROI', img2)
    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
        point2 = (x, round((x-point1[0]) * scale + point1[1]))
        cv2.rectangle(img2, point1, point2, (0, 0, 255), 5)
        cv2.imshow('setect ROI', img2)
        print([point1, point2])

cv2.namedWindow('setect ROI')
cv2.setMouseCallback('setect ROI', on_mouse, img)
cv2.imshow('setect ROI', img)
cv2.waitKey(0)
cv2.destroyAllWindows()