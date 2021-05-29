
## Brief Introduction
Based on [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit).  
Add some modules to trans DOTA annotation format to YOLO annotation format.  
Add some files for every demo.


## Fuction
* `DOTA.py`  Load image, and show the bounding oriented box.

* `ImgSplit.py` Split image and the label.

* `ResultMerge.py` Merge the detection result annotation txt.

* `dota_×_evaluation_task×.py` Evaluate the detection result annotation txt.

* `YOLO_Transformer.py`     Trans DOTA format to YOLO(OBB or HBB) format.

* `Draw_DOTA_YOLO.py` Picture the YOLO_OBB labels(after augmented).

## Installation
Same as [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit).  Then:

```
$  pip install -r requirements.txt
```

## More detailed explanation
想要了解这几个函数实现的细节和原理可以看我的知乎文章;    
[DOTA遥感数据集以及相关工具DOTA_devkit的整理(踩坑记录)](https://zhuanlan.zhihu.com/p/355862906);    
[DOTA数据格式转YOLO数据格式工具(cv2.minAreaRect踩坑记录)](https://zhuanlan.zhihu.com/p/356416158);


## Usage Example
* `DOTA.py`     
```javascript
$  python DOTA.py
```
![DOTA_HBB_label](./P0003_HBB.png)
![DOTA_OBB_label](./P0003_OBB.png)
* `ImgSplit.py` 
```javascript
$  python ImgSplit_multi_process.py
```
![Img_before_split](./P0130.png)
![Img_after_split](./P0130__1__0___0.png)
* `ResultMerge.py` 
```javascript
$  python ResultMerge.py
```
![visualize_detection_result1](./P0004__1__0___0.png)
![visualize_detection_result2](./P0004__1__0___440.png)
![visualize_merged_result](./P0004_.png)




* `dota_v1.5_evaluation_task1.py` 

change the path with yours.
```javascript
detpath = r'/.../evaluation_example/result_classname/Task1_{:s}.txt'
annopath = r'/.../evaluation_example/row_DOTA_labels/{:s}.txt'
imagesetfile = r'/.../evaluation_example/imgnamefile.txt'
```
```javascript
$  python dota_v1.5_evaluation_task1.py
```

* `YOLO_Transform.py` 
```javascript
$  python YOLO_Transform.py
```
```javascript
DOTA format:    poly classname diffcult
    To
YOLO HBB format: classid x_c y_c width height   ——   def dota2Darknet()
longside format： classid x_c y_c longside shortside Θ  Θ∈[0, 180)  ——  def dota2LongSideFormat()
```


* `Draw_DOTA_YOLO.py`

1.Run YOLO_Transformer.py to get the YOLO_OBB_labels first.

2.then augment YOLO_OBB_labels and visualize it:
```javascript
$  Draw_DOTA_YOLO.py
```
![visualize_augmented_labels](./P0003_augment_.png)


## 有问题反馈
在使用中有任何问题，欢迎反馈给我，可以用以下联系方式跟我交流

* 知乎（@[略略略](https://www.zhihu.com/people/lue-lue-lue-3-92-86)）
* 代码问题提issues,其他问题请知乎上联系


## 感激
感谢以下的项目,排名不分先后

* [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit)

## 关于作者

```javascript
  Name  : "胡凯旋"
  describe myself："咸鱼一枚"
  
```
