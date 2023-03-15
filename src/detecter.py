import os
import yaml
import os.path as osp
import time
import cv2
import torch
import torch.nn as nn
from .utils.data_augment import preproc
from .utils.model_utils import fuse_model, get_model_info
from .utils.boxes import postprocess
from .models.yolox import YOLOX
from .models.yolo_head import YOLOXHead
from .models.yolo_pafpn import YOLOPAFPN


class Detecter(nn.Module):
    def __init__(self,
                 input_size=(800, 1440),
                 model_weighs="",
                 model_config="",
                 device=None,
                 half=False,
                 fuse=False):
        super().__init__()
        # load model configs
        with open(model_config, 'r', encoding='utf-8') as m_f:
            cont = m_f.read()
            m_cfg = yaml.load(cont)
            self.m_cfg = m_cfg
        self.confthre = self.m_cfg["confthre"]
        self.nmsthre = self.m_cfg["nmsthre"]
        self.num_classes = self.m_cfg["num_classes"]
        self.depth = self.m_cfg["depth"]
        self.width = self.m_cfg["width"]

        self.input_size = input_size
        self.model_weighs = model_weighs
        self.device = device
        self.half = half
        self.fuse = fuse

        # create model-yolox
        in_channels = [256, 512, 1024]
        backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels)
        head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels)
        self.model = YOLOX(backbone, head)

        self.model.apply(self.init_yolo)
        self.model.head.initialize_biases(1e-2)

        self.model.to(self.device)
        self.model.eval()

        # load model weights
        ckpt = torch.load(self.model_weighs, map_location="cpu")
        # load the model state dict
        self.model.load_state_dict(ckpt["model"])
        print("loaded checkpoint done.")

        # half and fuse
        if self.fuse:
            self.model = fuse_model(self.model)
        if self.half:
            self.model = self.model.half()

        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def init_yolo(self, M):
        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    def inference(self, img):
        # img, ndarray, (H, W, 3), BGR
        img_info = {}
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        # img pre-process
        img, ratio = preproc(img, self.input_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.half:
            img = img.half()  # to FP16

        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
        if outputs[0] is not None:
            outputs[0][:, :4] /= img_info["ratio"]
            outputs[0][:, 4] = outputs[0][:, 4] * outputs[0][:, 5]
        return outputs
