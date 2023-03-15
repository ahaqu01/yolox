# yolox

## 一、算法介绍

本仓库为基于yolox的行人检测器，yolox论文参考见：https://arxiv.org/pdf/2107.08430.pdf

本仓库的yolox目前只支持行人检测

## 二、算法依赖

依赖见 requirements.txt

## 三、算法使用样例及参数说明

**you can use it as submodule**

在自己的项目目录下，git submodule add  https://github.com/ahaqu01/yolox.git

便会在项目目录下下载到yolox相关代码

下载完成后，便可在自己项目中使用yolox API，**使用样例和输入输出说明**如下：

**创建yolox.src.detecter.Detecter类时，相关参数如下：**

input_size： 输入图像尺寸，默认为（800，1440），输入尺寸越大，速度越慢

model_weighs：模型路径

model_config：模型配置文件路径

device：运行时的device

half：是否用半精度推理

fuse：是否把conv层和bn层融合，融合后推理速度更快

**使用yolox.src.detecter.Detecter.inference进行推理时，输入输出如下：**

输入：numpy.ndarray格式的图像, (H, W, 3), BGR通道顺序

输出：假设输出为pred，pred[:, :4]为每个行人的bbox框，xyxy，pred[4]为行人bbox框的置信度

```python
import torch
from yolox.src.detecter import Detecter
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
yx = Detecter(input_size=(800, 1440),
              model_weighs="./yolox/src/weights/bytetrack_m_mot17.pth.tar",
              model_config="./yolox/src/configs/yolox_m_mix_det.yaml",
              device=device,
              half=False,
              fuse=False)
yolovx_pred = yx.inference(frame_bgr)
```
