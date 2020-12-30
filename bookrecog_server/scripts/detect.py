import os
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import base64
import numpy as np
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from scripts.tools import custom_dict
from skimage.io import imsave
from deepcv.base import to_rgb
'''
基于yolov5的目标检测
'''

yolo_conf = custom_dict(
    {
    "device" : "0",
    "weights" : "./weights/yolov5_best.pt",
    "img_size" : 640,
    "augment" : True,
    "conf_thres" : 0.25,
    "iou_thres" : 0.45,
    "classes" : [0,1,2,3],
    "agnostic_nms" : True,
    "model_f":"./models/yolov5.py",
    })



def load_parm():
    imgsz = yolo_conf.img_size
    device = select_device(yolo_conf.device)
    half = device.type != 'cpu'
    model = attempt_load(yolo_conf.weights, map_location=device)
    print("laoded model yolov5")
    print("load weight")

    imgsz = check_img_size(imgsz, s=model.stride.max())
    if half:
        model.half()

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    print(f"start infering [{yolo_conf.model_f}, {yolo_conf.weights}]")

    return device, model, half, names, colors, imgsz

def get_detect_one(img_bytes):
    img_string = np.array(img_bytes).tostring()
    img_string = np.asarray(bytearray(img_string), dtype="uint8")
    image = cv2.imdecode(img_string, cv2.IMREAD_COLOR)
    
    img0 = to_rgb(image)
    img = letterbox(img0, new_shape=yolo_conf.img_size)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    pred = model(img, augment=yolo_conf.augment)[0]
    pred = non_max_suppression(pred, yolo_conf.conf_thres, yolo_conf.iou_thres, classes=yolo_conf.classes, agnostic=yolo_conf.agnostic_nms)

    for i, det in enumerate(pred):  # detections per image
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                label = '%s %.2f' % (names[int(cls)], conf)
                plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)
    
    #imsave("./tmp/test.jpg",img0)
    return img0

def get_detect(imgf, save_dir = './'):
    os.makedirs(save_dir, exist_ok=True)
    dataset = LoadImages(imgf, img_size=imgsz)

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=yolo_conf.augment)[0]

        pred = non_max_suppression(pred, yolo_conf.conf_thres, yolo_conf.iou_thres, classes=yolo_conf.classes, agnostic=yolo_conf.agnostic_nms)

        
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = Path(path), '', im0s
            
            save_path = os.path.join(save_dir, p.name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()


                for *xyxy, conf, cls in reversed(det):
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            cv2.imwrite(save_path, im0)

    return save_dir

# 全局                    
device, model, half, names, colors, imgsz = load_parm()

# if __name__ == "__main__":
#     img=cv2.imread("./data/book/book1_0.jpg")
#     img0 = get_detect_one(img)
#     res = base64.b64encode(img0)
#     print(res)
