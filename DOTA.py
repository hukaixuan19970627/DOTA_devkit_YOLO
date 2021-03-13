# coding=gbk

#The code is used for visulization, inspired from cocoapi
#  Licensed under the Simplified BSD License [see bsd.txt]

import os
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Circle
import numpy as np
import dota_utils as util
from collections import defaultdict
import cv2

def _isArrayLike(obj):
    if type(obj) == str:
        return False
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

class DOTA:
    def __init__(self, basepath):
        self.basepath = basepath
        self.labelpath = os.path.join(basepath, 'labelTxt')
        self.imagepath = os.path.join(basepath, 'images')
        self.imgpaths = util.GetFileFromThisRootDir(self.labelpath)
        self.imglist = [util.custombasename(x) for x in self.imgpaths]
        self.catToImgs = defaultdict(list)
        self.ImgToAnns = defaultdict(list)
        self.createIndex()

    def createIndex(self):
        for filename in self.imgpaths:
            objects = util.parse_dota_poly(filename)
            imgid = util.custombasename(filename)
            self.ImgToAnns[imgid] = objects
            for obj in objects:
                cat = obj['name']
                self.catToImgs[cat].append(imgid)

    def getImgIds(self, catNms=[]):
        """
        :param catNms: category names 类名 eg:catNms=['ships']
        :return: all the image ids contain the categories 所有包含该类名的图片id eg:['P0706', 'P
        1234', 'P2709']
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        if len(catNms) == 0:
            return self.imglist
        else:
            imgids = []
            for i, cat in enumerate(catNms):
                if i == 0:
                    imgids = set(self.catToImgs[cat])
                else:
                    imgids &= set(self.catToImgs[cat])
        return list(imgids)

    def loadAnns(self, catNms=[], imgId = None, difficult=None):
        """
        :param catNms: category names
        :param imgId: the img to load anns
        :return: objects
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        objects = self.ImgToAnns[imgId]
        if len(catNms) == 0:
            return objects
        outobjects = [obj for obj in objects if (obj['name'] in catNms)]
        return outobjects
    def showAnns(self, objects, imgId, range):
        """
        :param catNms: category names 类名
        :param objects: objects to show  即labels信息
        :param imgId: img to show 待显示的图片id
        :param range: display range in the img 图片的显示范围
        :return:
        """
        img = self.loadImgs(imgId)[0]
        plt.imshow(img)
        plt.axis('off')

        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        circles = []
        r = 5
        for obj in objects:
            c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
            poly = obj['poly']
            polygons.append(Polygon(poly))
            color.append(c)
            point = poly[0]
            circle = Circle((point[0], point[1]), r)
            circles.append(circle)
        p = PatchCollection(polygons, facecolors=color, linewidths=0, alpha=0.4)
        ax.add_collection(p)
        p = PatchCollection(polygons, facecolors='none', edgecolors=color, linewidths=2)
        ax.add_collection(p)
        p = PatchCollection(circles, facecolors='red')
        ax.add_collection(p)
    def loadImgs(self, imgids=[]):
        """
        :param imgids: integer ids specifying img 待加载的图片名 eg:imgids=['P0706','P0770']
        :return: loaded img objects 加载的图片张量数组 imgs=[...,...]
        """
        print('isarralike:', _isArrayLike(imgids))
        imgids = imgids if _isArrayLike(imgids) else [imgids]
        print('imgids:', imgids)
        imgs = []
        for imgid in imgids:
            filename = os.path.join(self.imagepath, imgid + '.png')
            print('filename:', filename)
            img = cv2.imread(filename)
            imgs.append(img)
        return imgs

if __name__ == '__main__':
    examplesplit = DOTA(r'./DOTA_demo')  # (r'./example')
    imgids = examplesplit.getImgIds(catNms=['small-vehicle'])  # 获取包含该类名的所有图片id eg:['P1088']
    img = examplesplit.loadImgs(imgids)  # 获取对应id图片所对应的small-vehicle张量数组
    for imgid in imgids:
        imgid = 'P0003'  #图片名称
        anns = examplesplit.loadAnns(imgId=imgid)  # 加载对应id图片的labels相关信息
        '''
        anns =
        [{'name': 'ship', 
          'difficult': '1', 
          'poly': [(1054.0, 1028.0), (1063.0, 1011.0), (1111.0, 1040.0), (1112.0, 1062.0)], 
          'area': 1159.5
          },
          ...
        ]
        '''
        examplesplit.showAnns(anns, imgid, 2)  # 将labels信息显示在对应id的图片上
        plt.show()