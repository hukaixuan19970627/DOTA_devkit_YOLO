# -*- coding: utf-8 -*-
import dota_utils as util
import os
import numpy as np
from PIL import Image
import cv2
import random
from YOLO_Transform import longsideformat2cvminAreaRect, cvminAreaRect2longsideformat
import math

import cv2
import math
import numpy as np
import os
import shutil


def rotateAugment(angle, scale, image, labels):
    """
    旋转目标增强  随机旋转
    @param angle: 旋转增强角度 int 单位为度
    @param scale: 设为1,尺度由train.py中定义
    @param image:  img信息  shape(heght, width, 3)
    @param labels:  (num, [classid x_c y_c longside shortside Θ]) Θ ∈ int[0,180)
    @return:
            rotated_img: augmented_img信息  shape(heght, width, 3)
            rotated_labels: augmented_label:  (num, [classid x_c y_c longside shortside Θ])
    """
    Pi_angle = -angle * math.pi / 180.0  # 弧度制，后面旋转坐标需要用到，注意负号！！！
    rows, cols = image.shape[:2]
    a, b = cols / 2, rows / 2
    M = cv2.getRotationMatrix2D(center=(a, b), angle=angle, scale=scale)
    rotated_img = cv2.warpAffine(image, M, (cols, rows))  # 旋转后的图像保持大小不变
    rotated_labels = []
    for label in labels:
        # rect=[(x_c,y_c),(w,h),Θ] Θ:flaot[0-179]  -> (-180,0)
        rect = longsideformat2cvminAreaRect(label[1], label[2], label[3], label[4], (label[5] - 179.9))
        # poly = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
        poly = cv2.boxPoints(rect)  # 返回rect对应的四个点的值 normalized

        # 四点坐标反归一化
        poly[:, 0] = poly[:, 0] * cols
        poly[:, 1] = poly[:, 1] * rows

        # 下面是计算旋转后目标相对旋转过后的图像的位置
        X0 = (poly[0][0] - a) * math.cos(Pi_angle) - (poly[0][1] - b) * math.sin(Pi_angle) + a
        Y0 = (poly[0][0] - a) * math.sin(Pi_angle) + (poly[0][1] - b) * math.cos(Pi_angle) + b

        X1 = (poly[1][0] - a) * math.cos(Pi_angle) - (poly[1][1] - b) * math.sin(Pi_angle) + a
        Y1 = (poly[1][0] - a) * math.sin(Pi_angle) + (poly[1][1] - b) * math.cos(Pi_angle) + b

        X2 = (poly[2][0] - a) * math.cos(Pi_angle) - (poly[2][1] - b) * math.sin(Pi_angle) + a
        Y2 = (poly[2][0] - a) * math.sin(Pi_angle) + (poly[2][1] - b) * math.cos(Pi_angle) + b

        X3 = (poly[3][0] - a) * math.cos(Pi_angle) - (poly[3][1] - b) * math.sin(Pi_angle) + a
        Y3 = (poly[3][0] - a) * math.sin(Pi_angle) + (poly[3][1] - b) * math.cos(Pi_angle) + b

        poly_rotated = np.array([(X0, Y0), (X1, Y1), (X2, Y2), (X3, Y3)])
        # 四点坐标归一化
        poly_rotated[:, 0] = poly_rotated[:, 0] / cols
        poly_rotated[:, 1] = poly_rotated[:, 1] / rows

        rect_rotated = cv2.minAreaRect(np.float32(poly_rotated))  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）

        c_x = rect_rotated[0][0]
        c_y = rect_rotated[0][1]
        w = rect_rotated[1][0]
        h = rect_rotated[1][1]
        theta = rect_rotated[-1]  # Range for angle is [-90，0)

        label[1:] = cvminAreaRect2longsideformat(c_x, c_y, w, h, theta)

        if (sum(label[1:-1] <= 0) + sum(label[1:3] >= 1)) >= 1:  # 0<xy<1, 0<side<=1
            # print('bbox[:2]中有>= 1的元素,bbox中有<= 0的元素,已将某个box排除,')
            # print(
            #     '出问题的longside形式数据:[%.16f, %.16f, %.16f, %.16f, %.1f]' % (label[1], label[2], label[3], label[4], label[5]))
            continue

        label[-1] = int(label[-1] + 180.5)  # range int[0,180] 四舍五入
        if label[-1] == 180:  # range int[0,179]
            label[-1] = 179
        rotated_labels.append(label)

    return rotated_img, np.array(rotated_labels)




def drawLongsideFormatImg(imgpath, txtpath, dstpath, extractclassname, thickness=2, augment=False):
    """
    根据labels绘制边框(label_format:classid, x_c_normalized, y_c_normalized, longside_normalized, shortside_normalized, Θ)
    :param imgpath: the path of images
    :param txtpath: the path of txt in longside format
    :param dstpath: the path of image_drawed
    :param extractclassname: the category you selected
    """
    if os.path.exists(dstpath):
        shutil.rmtree(dstpath)  # delete output folder
    os.makedirs(dstpath)  # make new output folder
    # 设置画框的颜色    colors = [[178, 63, 143], [25, 184, 176], [238, 152, 129],....,[235, 137, 120]]随机设置RGB颜色
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(extractclassname))]
    filelist = util.GetFileFromThisRootDir(txtpath)  # fileist=['/.../P0005.txt', ..., /.../P000?.txt]
    for fullname in filelist:  # fullname='/.../P000?.txt'
        objects = util.parse_longsideformat(fullname)
        '''
        objects[i] = [classid, x_c_normalized, y_c_normalized, longside_normalized, shortside_normalized, theta]
        '''
        name = os.path.splitext(os.path.basename(fullname))[0]  # name='P000?'
        img_fullname = os.path.join(imgpath, name + '.png')  # img_fullname='/.../P000?.png'
        img_savename = os.path.join(dstpath, name + '_.png')  # img_fullname='/.../_P000?.png'
        img = Image.open(img_fullname)  # 图像被打开但未被读取
        img_w, img_h = img.size
        img = cv2.imread(img_fullname)  # 读取图像像素
        objects = np.array(objects)

        if augment:
            # flip up augment
            if len(objects):
                img = cv2.flip(img, 0)  # 垂直翻转
                print(img.shape)
                objects[:, 2] = 1 - objects[:, 2]  # y变x不变
                objects[:, -1] = 180 - objects[:, -1]  # θ根据上下偏转也进行改变
                objects[objects[:, -1] == 180, -1] = 0  # 原θ=0时，情况特殊

            # flip lr augment
            if len(objects):
                img = cv2.flip(img, 1)  # 水平翻转
                objects[:, 1] = 1 - objects[:, 1]  # x变y不变
                objects[:, -1] = 180 - objects[:, -1]  # θ根据左右偏转也进行改变
                objects[objects[:, -1] == 180, -1] = 0  # 原θ=0时，情况特殊

            #  旋转augment
            degrees = 45
            rotate_angle = random.uniform(-degrees, degrees)
            img, objects = rotateAugment(rotate_angle, 1, img, objects)


        for i, obj in enumerate(objects):
            # obj = [classid, x_c_normalized, y_c_normalized, longside_normalized, shortside_normalized, float:0-179]
            class_index = obj[0]
            # rect=[(x_c,y_c),(w,h),Θ] Θ:flaot[0-179]  -> (-180,0)
            rect = longsideformat2cvminAreaRect(obj[1], obj[2], obj[3], obj[4], (obj[5]-179.9))
            # poly = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
            poly = np.float32(cv2.boxPoints(rect))  # 返回rect对应的四个点的值 normalized

            # 四点坐标反归一化 取整
            poly[:, 0] = poly[:, 0] * img_w
            poly[:, 1] = poly[:, 1] * img_h
            poly = np.int0(poly)

            # 画出来
            cv2.drawContours(image=img,
                             contours=[poly],
                             contourIdx=-1,
                             color=colors[int(class_index)],
                             thickness=thickness)
        cv2.imwrite(img_savename, img)



if __name__ == '__main__':
    drawLongsideFormatImg(imgpath='DOTA_demo/images',
                          txtpath='DOTA_demo/yolo_labels',
                          dstpath='DOTA_demo/draw_longside_img',
                          extractclassname=util.classnames_v1_5,
                          augment=True)