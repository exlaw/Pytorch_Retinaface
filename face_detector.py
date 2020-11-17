from __future__ import print_function
from typing import List
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50, cfg_re18, cfg_re34
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import time


parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet18', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

class BoundingBox:
    left: int
    right: int
    top: int
    bottom: int
    
    def __init__(self, left, top, right, bottom):
        self.left = int(left)
        self.top = int(top)
        self.right = int(right)
        self.bottom = int(bottom)


class FaceDetection:
    bbox: BoundingBox
    category: int
    confidence: float

    def __init__(self, left, top, right, bottom, category, confidence):
        self.bbox = BoundingBox(left, top, right, bottom)
        self.category = int(category)
        self.confidence = confidence

    def __str__(self):
        return f'#FaceDetection# bbox=[({self.bbox.left},{self.bbox.top})->({self.bbox.right},{self.bbox.bottom})], confidence={self.confidence}'
    

# criterion = MultiBoxLoss(2, 0.35, True, 0, True, 7, 0.35, False)


class FaceDetector():

    def __init__(self):
        # TODO: add initialization logic
        torch.set_grad_enabled(False)
        self.cfg = None
        if args.network == "mobile0.25":
            self.cfg = cfg_mnet
        elif args.network == "resnet50":
            self.cfg = cfg_re50
        elif args.network == "resnet18":
            self.cfg = cfg_re18
        elif args.network == "resnet34":
            self.cfg = cfg_re34
        # net and model
        self.net = RetinaFace(cfg=self.cfg, phase='test')
        # self.net = load_model(self.net, args.trained_model, args.cpu)
        self.net.eval()
        print('Finished loading model!')
        print(self.net)
        cudnn.benchmark = True
        self.device = torch.device("cpu" if args.cpu else "cuda")
        self.net = self.net.to(self.device)

        self.resize = 1

    def detect_image(self, img) -> List[FaceDetection]:
        # TODO: add detect logic for single image
        print(np.shape(img))
        tic = time.time()
        img = np.float32(img)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        loc, conf, landms = self.net(img)  # forward pass
        
        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / self.resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)

        keep = py_cpu_nms(dets, args.nms_threshold)

        dets = dets[keep, :]

        dets = dets[:args.keep_top_k, :]

        # show image
        box_list = []
        for b in dets:
            if b[4] < args.vis_thres:
                continue
            score = b[4]
            b = list(map(int, b))
            box_list.append(FaceDetection(b[0], b[1], b[2], b[3], 0, score))

        print('net forward time: {:.4f}'.format(time.time() - tic))

        return box_list

    def detect_images(self, imgs) -> List[List[FaceDetection]]:
        boxes_list = []
        for img in imgs:
            boxes = self.detect_image(img)
            boxes_list.append(boxes)
        return boxes_list
    
    def visualize(self, image, detection_list: List[FaceDetection], color=(0,0,255), thickness=5):
        img = image.copy()
        for detection in detection_list:
            bbox = detection.bbox
            p1 = bbox.left, bbox.top
            p2 = bbox.right, bbox.bottom
            cv2.rectangle(img, p1, p2, color, thickness=thickness, lineType=cv2.LINE_AA)
        return img
