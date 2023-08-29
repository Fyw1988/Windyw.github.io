---
title: Grad CAM
abbrlink: 1
cover: /img/violet.png
---
之前想画一下YOLOv5的类激活图，但是尝试了半天都没成功，偶然发现YOLOv5的issue中有人已经实现了支持画网络某一层的gram-cam，只是暂时好像还没有并入YOLOv5项目中：[Add GradCAM integration](https://github.com/ultralytics/yolov5/pull/10649)

#### 1. 使用方法
- **打开上面的github链接**：
![[Pasted image 20230505162819.png]]
- **点击`Files changed`**:
![[Pasted image 20230505162929.png]]
- **复制此处explainer文件夹中的explainer.py到yolov5的文件目录下**：
![[Pasted image 20230505163136.png]]
- **指定参数**：
```python
def parseopt():  
    parser = argparse.ArgumentParser()  
    # 指定你训练好的网络权重路径  
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/yolov5l/weights/best.pt', help='model path or triton URL')  
    # 指定要画类激活图的图像路径(目前只支持一张一张地画)  
    parser.add_argument('--source', type=str, default='/home/ding/fyw/YOLOv5/data/images/0.jpg', help='file/dir/URL/glob/screen/0(webcam)')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')  
    # 指定gpu  
    parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')  
    # 选择类激活图类型  
    parser.add_argument('--method',  
                        type=str,  
                        default='EigenCAM',  
                        help='the method to use for interpreting the feature maps')  
    parser.add_argument('--verbose', action='store_true', help='verbose log')
```
- **运行即可**：
`python explainer/explainer.py`

#### 2.怎么画特定层的累激活图
这里以YOLOv5l为例子。首先我们用`print(model)`打印出模型(太长了这里仅展现一部分)：
```python
YOLOV5TorchObjectDetector(
  (model): DetectionModel(
    (model): Sequential(
      (0): Conv(
        (conv): Conv2d(3, 64, kernel_size=(6, 6), stride=(2, 2), padding=(2, 2), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (1): Conv(
        (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU()   # grad_cam 1
      )
      (2): C3(
        (cv1): Conv(
          (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): SiLU()
        )
        (cv3): Conv(
          (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): SiLU()
        )
        (m): Sequential(
          (0): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU()   # grad_cam 2
            )
```
举个例子，如果你想画出`grad_cam 1`(见上面模型文件的注释)，那么修改代码中的`target_layer`：
![[Pasted image 20230505165547.png]]
修改成：
`target_layers = [model.model.model.model[1]._modules['act']]`
如果想画出`grad_cam 2`，则修改成：
`target_layers = [model.model.model.model[2]._modules['m']._modules['0']._modules['cv1']._modules['act']]`
对于模型中其他的任意层，都按照上面的方式类推就行。

另外，在跑这个代码的时候可能会出现因版本问题导致的错误：
```
TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
```
修改这一行代码，加上`.cpu()`就可以解决：
![[Pasted image 20230505171054.png]]
