# -*- coding: utf-8 -*-
"""
    To use the code, users should to config detpath, annopath and imagesetfile
    detpath is the path for 15 result files, for the format, you can refer to "http://captain.whu.edu.cn/DOTAweb/tasks.html"
    search for PATH_TO_BE_CONFIGURED to config the paths
    Note, the evaluation is on the large scale images
"""
import os
import numpy as np
import dota_utils as util
import re
import time
import polyiou

## the IoU thresh for nms when merge image
nms_thresh = 0.3

def py_cpu_nms_poly(dets, thresh):
    """
    任意四点poly nms.取出nms后的边框的索引
    @param dets: shape(detection_num, [poly, confidence1]) 原始图像中的检测出的目标数量
    @param thresh:
    @return:
            keep: 经nms后的目标边框的索引
    """
    scores = dets[:, 8]
    polys = []
    areas = []
    for i in range(len(dets)):
        tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                            dets[i][2], dets[i][3],
                                            dets[i][4], dets[i][5],
                                            dets[i][6], dets[i][7]])
        polys.append(tm_polygon)

    # argsort将元素小到大排列 返回索引值 [::-1]即从后向前取元素
    order = scores.argsort()[::-1]  # 取出元素的索引值 顺序为从大到小
    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]  # 取出当前剩余置信度最大的目标边框的索引
        keep.append(i)
        for j in range(order.size - 1):  # 求出置信度最大poly与其他所有poly的IoU
            iou = polyiou.iou_poly(polys[i], polys[order[j + 1]])
            ovr.append(iou)
        ovr = np.array(ovr)
        inds = np.where(ovr <= thresh)[0]  # 找出iou小于阈值的索引
        order = order[inds + 1]
    return keep

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    #print('dets:', dets)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    ## index for dets
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def nmsbynamedict(nameboxdict, nms, thresh):
    """
    对namedict中的目标信息进行nms.不改变输入的数据形式
    @param nameboxdict: eg:{
                           'P706':[[poly1, confidence1], ..., [poly9, confidence9]],
                           ...
                           'P700':[[poly1, confidence1], ..., [poly9, confidence9]]
                            }
    @param nms:
    @param thresh: nms阈值, IoU阈值
    @return:
            nameboxnmsdict: eg:{
                                'P706':[[poly1, confidence1], ..., [poly_nms, confidence9]],
                                 ...
                                'P700':[[poly1, confidence1], ..., [poly_nms, confidence9]]
                               }
    """
    # 初始化字典
    nameboxnmsdict = {x: [] for x in nameboxdict}  # eg: nameboxnmsdict={'P0770': [], 'P1888': []}
    for imgname in nameboxdict:  # 提取nameboxdict中的key eg:P0770   P1888
        keep = nms(np.array(nameboxdict[imgname]), thresh)  # rotated_nms索引值列表
        outdets = []
        #print('nameboxdict[imgname]: ', nameboxnmsdict[imgname])
        for index in keep:
            # print('index:', index)
            outdets.append(nameboxdict[imgname][index])
        nameboxnmsdict[imgname] = outdets
    return nameboxnmsdict

def poly2origpoly(poly, x, y, rate):
    origpoly = []
    for i in range(int(len(poly)/2)):
        tmp_x = float(poly[i * 2] + x) / float(rate)
        tmp_y = float(poly[i * 2 + 1] + y) / float(rate)
        origpoly.append(tmp_x)
        origpoly.append(tmp_y)
    return origpoly

def mergebase(srcpath, dstpath, nms):
    """
    将源路径中所有的txt目标信息,经nms后存入目标路径中的同名txt
    @param srcpath: 合并前信息保存的txt源路径
    @param dstpath: 合并后信息保存的txt目标路径
    @param nms: NMS函数
    """
    filelist = util.GetFileFromThisRootDir(srcpath)  # srcpath文件夹下的所有文件相对路径 eg:['example_split/../P0001.txt', ..., '?.txt']
    for fullname in filelist:  # 'example_split/../P0001.txt'
        name = util.custombasename(fullname)  # 只留下文件名 eg:P0001
        dstname = os.path.join(dstpath, name + '.txt')  # eg: example_merge/..P0001.txt
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
        with open(fullname, 'r') as f_in:
            nameboxdict = {}
            lines = f_in.readlines()  # 读取txt中所有行,每行作为一个元素存于list中
            splitlines = [x.strip().split(' ') for x in lines]  # 再次分割list中的每行元素 shape:n行 * m个元素
            for splitline in splitlines:  # splitline:每行中的m个元素
                # splitline = [待merge图片名(该目标所处图片名称), confidence, x1, y1, x2, y2, x3, y3, x4, y4]
                subname = splitline[0]  # 每行的第一个元素 是被分割的图片的图片名 eg:P0706__1__0___0
                splitname = subname.split('__')  # 分割待merge的图像的名称 eg:['P0706','1','0','_0']
                oriname = splitname[0]  # 获得待merge图像的原图像名称 eg:P706
                pattern1 = re.compile(r'__\d+___\d+')  # 预先编译好r'__\d+___\d+' 提高重复使用效率 \d表示数字

                x_y = re.findall(pattern1, subname)  # 匹配subname中的字符串 eg: x_y=['__0___0']
                x_y_2 = re.findall(r'\d+', x_y[0])  # 匹配subname中的字符串 eg: x_y_2= ['0','0']
                x, y = int(x_y_2[0]), int(x_y_2[1])  # 找到当前subname图片在原图中的分割位置

                pattern2 = re.compile(r'__([\d+\.]+)__\d+___')  # \.表示一切字符

                rate = re.findall(pattern2, subname)[0]  # 找到该subname分割图片时的分割rate (resize rate before cut)

                confidence = splitline[1]
                poly = list(map(float, splitline[2:]))  # 每个元素映射为浮点数 再放入列表中
                origpoly = poly2origpoly(poly, x, y, rate)  # 将目标位置信息resize 恢复成原图的poly坐标
                det = origpoly  # shape(8)
                det.append(confidence)  # [poly, confidence]
                det = list(map(float, det))
                if (oriname not in nameboxdict):
                    nameboxdict[oriname] = []   # 弄个元组,汇集原图目标信息 eg: 'P706':[[poly1, confidence1], ..., ]
                nameboxdict[oriname].append(det)
            nameboxnmsdict = nmsbynamedict(nameboxdict, nms, nms_thresh)  # 对nameboxdict元组进行nms
            with open(dstname, 'w') as f_out:
                for imgname in nameboxnmsdict:
                    for det in nameboxnmsdict[imgname]:  # 取出对应图片的nms后的目标信息
                        #print('det:', det)
                        confidence = det[-1]
                        bbox = det[0:-1]
                        outline = imgname + ' ' + str(confidence) + ' ' + ' '.join(map(str, bbox))
                        #print('outline:', outline)
                        f_out.write(outline + '\n')

def mergebyrec(srcpath, dstpath):
    """
    srcpath: result files before merge and nms
    dstpath: result files after merge and nms
    """
    # srcpath = r'E:\bod-dataset\results\bod-v3_rfcn_2000000'
    # dstpath = r'E:\bod-dataset\results\bod-v3_rfcn_2000000_nms'

    mergebase(srcpath,
              dstpath,
              py_cpu_nms)
def mergebypoly(srcpath, dstpath):
    """
    srcpath: result files before merge and nms.txt的信息格式为:[P0770__1__0___0 confidence poly]
    dstpath: result files after merge and nms.保存的txt信息格式为:[P0770 confidence poly]
    """
    # srcpath = r'/home/dingjian/evaluation_task1/result/faster-rcnn-59/comp4_test_results'
    # dstpath = r'/home/dingjian/evaluation_task1/result/faster-rcnn-59/testtime'

    mergebase(srcpath,
              dstpath,
              py_cpu_nms_poly)
if __name__ == '__main__':
    # see demo for example
    mergebypoly(r'ResultMerge_example', r'ResultMerge_example_result')
    # mergebyrec()