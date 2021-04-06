# -*- coding: utf-8 -*-
import dota_utils as util
import os
import numpy as np
from PIL import Image
import cv2
import random
import  shutil
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPoint  # 多边形
import time
import argparse

## trans dota format to format YOLO(darknet) required
def dota2Darknet(imgpath, txtpath, dstpath, extractclassname):
    """
    :param imgpath: the path of images
    :param txtpath: the path of txt in dota format
    :param dstpath: the path of txt in YOLO format
    :param extractclassname: the category you selected
    :return:
           txt format: id x y w h
    """
    if os.path.exists(dstpath):
        shutil.rmtree(dstpath)  # delete output folder
    os.makedirs(dstpath)  # make new output folder
    filelist = util.GetFileFromThisRootDir(txtpath)  # fileist=['/.../P0005.txt', ..., /.../P000?.txt]
    for fullname in filelist:  # fullname='/.../P000?.txt'
        objects = util.parse_dota_poly(fullname)
        '''
        objects =
        [{'name': 'ship', 
          'difficult': '1', 
          'poly': [(1054.0, 1028.0), (1063.0, 1011.0), (1111.0, 1040.0), (1112.0, 1062.0)], 
          'area': 1159.5
          },
          ...
        ]
        '''
        name = os.path.splitext(os.path.basename(fullname))[0]  # name='P000?'
        img_fullname = os.path.join(imgpath, name + '.png')  # img_fullname='/.../P000?.png'
        img = Image.open(img_fullname)
        img_w, img_h = img.size
        # print img_w,img_h
        with open(os.path.join(dstpath, name + '.txt'), 'w') as f_out:
            for obj in objects:
                poly = obj['poly']  # poly=[(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
                bbox = np.array(util.dots4ToRecC(poly, img_w, img_h))  # bbox=[x y w h]
                if (sum(bbox <= 0) + sum(bbox >= 1)) >= 1:  # 若bbox中有<=0或>= 1的元素则将该box排除
                    continue
                if (obj['name'] in extractclassname):
                    id = extractclassname.index(obj['name'])  # id=类名的索引 比如'plane'对应id=0
                else:
                    continue
                outline = str(id) + ' ' + ' '.join(list(map(str, bbox)))  # outline='id x y w h'
                f_out.write(outline + '\n')  # 写入txt文件中并加上换行符号 \n

## trans dota format to  (cls, c_x, c_y, Longest side, short side, angle:[0,179))
def dota2LongSideFormat(imgpath, txtpath, dstpath, extractclassname):
    """
    trans dota farmat to longside format
    :param imgpath: the path of images
    :param txtpath: the path of txt in dota format
    :param dstpath: the path of txt in YOLO format
    :param extractclassname: the category you selected
    """
    if os.path.exists(dstpath):
        shutil.rmtree(dstpath)  # delete output folder
    os.makedirs(dstpath)  # make new output folder
    filelist = util.GetFileFromThisRootDir(txtpath)  # fileist=['/.../P0005.txt', ..., /.../P000?.txt]
    for fullname in filelist:  # fullname='/.../P000?.txt'
        objects = util.parse_dota_poly(fullname)
        '''
        objects =
        [{'name': 'ship', 
          'difficult': '1', 
          'poly': [(1054.0, 1028.0), (1063.0, 1011.0), (1111.0, 1040.0), (1112.0, 1062.0)], 
          'area': 1159.5
          },
          ...
        ]
        '''
        name = os.path.splitext(os.path.basename(fullname))[0]  # name='P000?'
        img_fullname = os.path.join(imgpath, name + '.png')  # img_fullname='/.../P000?.png'
        img = Image.open(img_fullname)
        img_w, img_h = img.size
        # print img_w,img_h
        with open(os.path.join(dstpath, name + '.txt'), 'w') as f_out:
            num_gt = 0
            for i, obj in enumerate(objects):
                num_gt = num_gt + 1  # 为当前有效gt计数
                poly = obj['poly']  # poly=[(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
                poly = np.float32(np.array(poly))
                # 四点坐标归一化
                poly[:, 0] = poly[:, 0]/img_w
                poly[:, 1] = poly[:, 1]/img_h

                rect = cv2.minAreaRect(poly)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
               # box = np.float32(cv2.boxPoints(rect))  # 返回rect四个点的值

                c_x = rect[0][0]
                c_y = rect[0][1]
                w = rect[1][0]
                h = rect[1][1]
                theta = rect[-1]  # Range for angle is [-90，0)

                trans_data = cvminAreaRect2longsideformat(c_x, c_y, w, h, theta)
                if not trans_data:
                    if theta != 90:  # Θ=90说明wh中有为0的元素，即gt信息不完整，无需提示异常，直接删除
                        print('opencv表示法转长边表示法出现异常,已将第%d个box排除,问题出现在该图片中:%s' % (i, img_fullname))
                    num_gt = num_gt - 1
                    continue
                else:
                    # range:[-180，0)
                    c_x, c_y, longside, shortside, theta_longside = trans_data

                bbox = np.array((c_x, c_y, longside, shortside))

                if (sum(bbox <= 0) + sum(bbox[:2] >= 1) ) >= 1:  # 0<xy<1, 0<side<=1
                    print('bbox[:2]中有>= 1的元素,bbox中有<= 0的元素,已将第%d个box排除,问题出现在该图片中:%s' % (i, img_fullname))
                    print('出问题的longside形式数据:[%.16f, %.16f, %.16f, %.16f, %.1f]' % (c_x, c_y, longside, shortside, theta_longside))
                    num_gt = num_gt - 1
                    continue
                if (obj['name'] in extractclassname):
                    id = extractclassname.index(obj['name'])  # id=类名的索引 比如'plane'对应id=0
                else:
                    print('预定类别中没有类别:%s;已将该box排除,问题出现在该图片中:%s' % (obj['name'], fullname))
                    num_gt = num_gt - 1
                    continue
                theta_label = int(theta_longside + 180.5)  # range int[0,180] 四舍五入
                if theta_label == 180:  # range int[0,179]
                    theta_label = 179
                # outline='id x y longside shortside Θ'

                # final check
                if id > 15 or id < 0:
                    print('id problems,问题出现在该图片中:%s' % (i, img_fullname))
                    print('出问题的longside形式数据:[%.16f, %.16f, %.16f, %.16f, %.1f]' % (
                    c_x, c_y, longside, shortside, theta_longside))
                if theta_label < 0 or theta_label > 179:
                    print('id problems,问题出现在该图片中:%s' % (i, img_fullname))
                    print('出问题的longside形式数据:[%.16f, %.16f, %.16f, %.16f, %.1f]' % (
                        c_x, c_y, longside, shortside, theta_longside))
                outline = str(id) + ' ' + ' '.join(list(map(str, bbox))) + ' ' + str(theta_label)
                f_out.write(outline + '\n')  # 写入txt文件中并加上换行符号 \n

        if num_gt == 0:
            os.remove(os.path.join(dstpath, name + '.txt'))  #
            os.remove(img_fullname)
            os.remove(fullname)
            print('%s 图片对应的txt不存在有效目标,已删除对应图片与txt' % img_fullname)
    print('已完成文件夹内DOTA数据形式到长边表示法的转换')


def cvminAreaRect2longsideformat(x_c, y_c, width, height, theta):
    '''
    trans minAreaRect(x_c, y_c, width, height, θ) to longside format(x_c, y_c, longside, shortside, θ)
    两者区别为:
            当opencv表示法中width为最长边时（包括正方形的情况），则两种表示方法一致
            当opencv表示法中width不为最长边 ，则最长边表示法的角度要在opencv的Θ基础上-90度         
    @param x_c: center_x
    @param y_c: center_y
    @param width: x轴逆时针旋转碰到的第一条边
    @param height: 与width不同的边
    @param theta: x轴逆时针旋转与width的夹角，由于原点位于图像的左上角，逆时针旋转角度为负 [-90, 0)
    @return: 
            x_c: center_x
            y_c: center_y
            longside: 最长边
            shortside: 最短边
            theta_longside: 最长边和x轴逆时针旋转的夹角，逆时针方向角度为负 [-180, 0)
    '''
    '''
    意外情况:(此时要将它们恢复符合规则的opencv形式：wh交换，Θ置为-90)
    竖直box：box_width < box_height  θ=0
    水平box：box_width > box_height  θ=0
    '''
    if theta == 0:
        theta = -90
        buffer_width = width
        width = height
        height = buffer_width

    if theta > 0:
        if theta != 90:  # Θ=90说明wh中有为0的元素，即gt信息不完整，无需提示异常，直接删除
            print('θ计算出现异常，当前数据为：%.16f, %.16f, %.16f, %.16f, %.1f;超出opencv表示法的范围：[-90,0)' % (x_c, y_c, width, height, theta))
        return False

    if theta < -90:
        print('θ计算出现异常，当前数据为：%.16f, %.16f, %.16f, %.16f, %.1f;超出opencv表示法的范围：[-90,0)' % (x_c, y_c, width, height, theta))
        return False

    if width != max(width, height):  # 若width不是最长边
        longside = height
        shortside = width
        theta_longside = theta - 90
    else:  # 若width是最长边(包括正方形的情况)
        longside = width
        shortside = height
        theta_longside = theta

    if longside < shortside:
        print('旋转框转换表示形式后出现问题：最长边小于短边;[%.16f, %.16f, %.16f, %.16f, %.1f]' % (x_c, y_c, longside, shortside, theta_longside))
        return False
    if (theta_longside < -180 or theta_longside >= 0):
        print('旋转框转换表示形式时出现问题:θ超出长边表示法的范围：[-180,0);[%.16f, %.16f, %.16f, %.16f, %.1f]' % (x_c, y_c, longside, shortside, theta_longside))
        return False

    return x_c, y_c, longside, shortside, theta_longside

def drawLongsideFormatimg(imgpath, txtpath, dstpath, extractclassname, thickness=2):
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

    # time.sleep()

def longsideformat2cvminAreaRect(x_c, y_c, longside, shortside, theta_longside):
    '''
    trans longside format(x_c, y_c, longside, shortside, θ) to minAreaRect(x_c, y_c, width, height, θ)
    两者区别为:
            当opencv表示法中width为最长边时（包括正方形的情况），则两种表示方法一致
            当opencv表示法中width不为最长边 ，则最长边表示法的角度要在opencv的Θ基础上-90度         
    @param x_c: center_x
    @param y_c: center_y
    @param longside: 最长边
    @param shortside: 最短边
    @param theta_longside: 最长边和x轴逆时针旋转的夹角，逆时针方向角度为负 [-180, 0)
    @return: ((x_c, y_c),(width, height),Θ)
            x_c: center_x
            y_c: center_y
            width: x轴逆时针旋转碰到的第一条边最长边
            height: 与width不同的边
            theta: x轴逆时针旋转与width的夹角，由于原点位于图像的左上角，逆时针旋转角度为负 [-90, 0)
    '''
    if (theta_longside >= -180 and theta_longside < -90):  # width is not the longest side
        width = shortside
        height = longside
        theta = theta_longside + 90
    else:
        width = longside
        height =shortside
        theta = theta_longside

    if theta < -90 or theta >= 0:
        print('当前θ=%.1f，超出opencv的θ定义范围[-90, 0)' % theta)

    return ((x_c, y_c), (width, height), theta)

def delete(imgpath, txtpath):
    filelist = util.GetFileFromThisRootDir(txtpath)  # fileist=['/.../P0005.txt', ..., /.../P000?.txt]
    for fullname in filelist:  # fullname='/.../P000?.txt'
        name = os.path.splitext(os.path.basename(fullname))[0]  # name='P000?'
        img_fullname = os.path.join(imgpath, name + '.png')  # img_fullname='/.../P000?.png'
        if not os.path.exists(img_fullname):  # 如果文件bu存在
            os.remove(fullname)

if __name__ == '__main__':
    ## an example

    dota2LongSideFormat('./DOTA_demo/images',
                        './DOTA_demo/labelTxt',
                        './DOTA_demo/yolo_labels',
                        util.classnames_v1_5)

    drawLongsideFormatimg(imgpath='DOTA_demo/images',
                          txtpath='DOTA_demo/yolo_labels',
                          dstpath='DOTA_demo/draw_longside_img',
                          extractclassname=util.classnames_v1_5)
