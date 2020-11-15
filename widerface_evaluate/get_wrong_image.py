"""
WiderFace evaluation code
author: wondervictor
mail: tianhengcheng@gmail.com
copyright@wondervictor
"""

import os
import tqdm
import pickle
import argparse
import numpy as np
from scipy.io import loadmat
from bbox import bbox_overlaps
import cv2

event_dir = {}


def get_gt_boxes(gt_dir):
    """ gt dir: (wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat, wider_hard_val.mat)"""

    gt_mat = loadmat(os.path.join(gt_dir, 'wider_face_val.mat'))
    hard_mat = loadmat(os.path.join(gt_dir, 'wider_hard_val.mat'))
    medium_mat = loadmat(os.path.join(gt_dir, 'wider_medium_val.mat'))
    easy_mat = loadmat(os.path.join(gt_dir, 'wider_easy_val.mat'))

    facebox_list = gt_mat['face_bbx_list']
    event_list = gt_mat['event_list']
    # print(event_list)
    for event in event_list:
        items = event[0][0].split("--")
        event_dir[items[0]] = event[0][0]
    file_list = gt_mat['file_list']

    hard_gt_list = hard_mat['gt_list']
    medium_gt_list = medium_mat['gt_list']
    easy_gt_list = easy_mat['gt_list']

    return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list


def read_pred_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        img_file = lines[0].rstrip('\n\r')
        lines = lines[2:]

    # b = lines[0].rstrip('\r\n').split(' ')[:-1]
    # c = float(b)
    # a = map(lambda x: [[float(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4])] for a in x.rstrip('\r\n').split(' ')], lines)
    boxes = []
    for line in lines:
        line = line.rstrip('\r\n').split(' ')
        if line[0] is '':
            continue
        # a = float(line[4])
        boxes.append([float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])])
    boxes = np.array(boxes)
    # boxes = np.array(list(map(lambda x: [float(a) for a in x.rstrip('\r\n').split(' ')], lines))).astype('float')
    return img_file.split('/')[-1], boxes


def get_preds(pred_dir):
    events = os.listdir(pred_dir)
    boxes = dict()
    pbar = tqdm.tqdm(events)

    for event in pbar:
        pbar.set_description('Reading Predictions ')
        event_dir = os.path.join(pred_dir, event)
        event_images = os.listdir(event_dir)
        current_event = dict()
        for imgtxt in event_images:
            imgname, _boxes = read_pred_file(os.path.join(event_dir, imgtxt))
            current_event[imgname.rstrip('.jpg')] = _boxes
        boxes[event] = current_event
    return boxes


def image_eval(pred, gt, iou_thresh, name, output, _match, _least):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """
    _pred = pred.copy()
    _gt = gt.copy()

    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    overlaps = bbox_overlaps(_pred[:, :4], _gt)
    match = 0

    for h in range(_pred.shape[0]):

        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        if max_overlap >= iou_thresh:
            match += 1

    if match == _match and gt.shape[0] > _least:
        items = name[0][0].split("_")
        folder = event_dir[items[0]]
        image_path = '../data/widerface/val/images/' + folder + "/" + name[0][0] + ".jpg"
        print(image_path)
        image = cv2.imread(image_path)

        img1 = image.copy()
        for i in range(np.shape(_gt)[0]):
            p1 = int(_gt[i][0]), int(_gt[i][1])
            p2 = int(_gt[i][2]), int(_gt[i][3])
            cv2.rectangle(img1, p1, p2, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
        cv2.imwrite(output + name[0][0] + "gt.jpg", img1)
        img2 = image.copy()
        for i in range(np.shape(_pred)[0]):
            p1 = int(_pred[i][0]), int(_pred[i][1])
            p2 = int(_pred[i][2]), int(_pred[i][3])
            cv2.rectangle(img2, p1, p2, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
        cv2.imwrite(output + name[0][0] + "pred.jpg", img2)


def evaluation(pred, gt_path, output, match, least, iou_thresh=0.5):
    pred = get_preds(pred)
    facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(gt_path)
    event_num = len(event_list)
    pbar = range(event_num)
    for i in pbar:
        event_name = str(event_list[i][0][0])
        img_list = file_list[i][0]
        pred_list = pred[event_name]
        gt_bbx_list = facebox_list[i][0]
        for j in range(len(img_list)):
            pred_info = pred_list[str(img_list[j][0][0])]
            gt_boxes = gt_bbx_list[j][0].astype('float')
            if len(gt_boxes) == 0 or len(pred_info) == 0:
                continue
            image_eval(pred_info, gt_boxes, iou_thresh, img_list[j], output, match, least)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred', default="./widerface_resnet_pretrain/")
    parser.add_argument('-g', '--gt', default='./ground_truth/')
    parser.add_argument('-o', '--output', default='./output')
    parser.add_argument('-m', '--match', default=0)
    parser.add_argument('-l', '--least', default=0)

    args = parser.parse_args()

    output = args.output + "_" + str(args.match) + "_" + str(args.least) + "/"
    os.makedirs(output, exist_ok=True)
    evaluation(args.pred, args.gt,  output, args.match, args.least)

