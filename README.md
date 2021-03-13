
## Brief Introduction
Based on [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit).  
Add some modules to trans DOTA annotation format to YOLO annotation format.  
Add some files for every demo.


## Fuction
* `DOTA.py`     Load and image, and show the bounding box on it.
* `ImgSplit.py` Split the image and the label.
* `ResultMerge.py` Merge the detection result annotation txt.
* `dota_×_evaluation_task×.py` Evaluate the detection result annotation txt.
* `YOLO_Transformer.py` Trans DOTA format to YOLO format.
* `Draw_DOTA_YOLO.py`Picture the YOLO_OBB labels(after augmented) 

## Installation
Same as [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit).

## More detailed explanation
想要了解这几个函数实现的细节和原理可以看我的知乎文章;    
[DOTA遥感数据集以及相关工具DOTA_devkit的整理(踩坑记录)](https://zhuanlan.zhihu.com/p/355862906);    
[DOTA数据格式转YOLO数据格式工具(cv2.minAreaRect踩坑记录)](https://zhuanlan.zhihu.com/p/356416158);


## How to use it
* `DOTA.py`     
```javascript
$  python DOTA.py
```
* `ImgSplit.py` 
```javascript
$  python ImgSplit_multi_process.py
```
* `ResultMerge.py` 
```javascript
$  python ResultMerge.py
```
* `dota_v1.5_evaluation_task1.py` 

change the path with yours.
```javascript
detpath = r'/home/test/Persons/hukaixuan/DOTA_devkit-master/evaluation_example/result_classname/Task1_{:s}.txt'
annopath = r'/home/test/Persons/hukaixuan/DOTA_devkit-master/evaluation_example/row_DOTA_labels/{:s}.txt'
imagesetfile = r'/home/test/Persons/hukaixuan/DOTA_devkit-master/evaluation_example/imgnamefile.txt'
```
```javascript
$  python dota_v1.5_evaluation_task1.py
```

* `YOLO_Transformer.py` 
```javascript
$  python YOLO_Transformer.py
```
* `Draw_DOTA_YOLO.py`

1.Run YOLO_Transformer.py to get the YOLO_OBB_labels first.

2.then :
```javascript
$  Draw_DOTA_YOLO.py
```


## 有问题反馈
在使用中有任何问题，欢迎反馈给我，可以用以下联系方式跟我交流

* 知乎（@略略略）
* 代码问题提issues,其他问题请知乎上联系


## 感激
感谢以下的项目,排名不分先后

* [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit)

## 关于作者

```javascript
  Name  : "胡凯旋"
  describe myself："咸鱼一枚"
  
```
