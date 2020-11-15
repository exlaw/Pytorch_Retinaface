import os
import tqdm
import pickle
import argparse
import numpy as np
import cv2
import pickle
import shutil


def get_gt_boxes(gt_dir):
    """ gt dir: (wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat, wider_hard_val.mat)"""

    gt_dict = {}
    for i in range(1, 11):
        filename = os.path.join(gt_dir, 'FDDB-fold-{}-ellipseList.txt'.format('%02d' % i))
        assert os.path.exists(filename)
        gt_sub_dict = {}
        annotationfile = open(filename)
        while True:
            filename = annotationfile.readline()[:-1].replace('/', '_')
            if not filename:
                break
            line = annotationfile.readline()
            if not line:
                break
            facenum = int(line)
            face_loc = []
            for j in range(facenum):
                line = annotationfile.readline().strip().split()
                major_axis_radius = float(line[0])
                minor_axis_radius = float(line[1])
                angle = float(line[2])
                center_x = float(line[3])
                center_y = float(line[4])
                score = float(line[5])
                angle = angle / 3.1415926 * 180
                mask = np.zeros((1000, 1000), dtype=np.uint8)
                cv2.ellipse(mask, ((int)(center_x), (int)(center_y)),
                            ((int)(major_axis_radius), (int)(minor_axis_radius)), angle, 0., 360., (255, 255, 255))
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2:]
                r = cv2.boundingRect(contours[0])
                x_min = r[0]
                y_min = r[1]
                x_max = r[0] + r[2]
                y_max = r[1] + r[3]
                face_loc.append([x_min, y_min, x_max, y_max])
            face_loc = np.array(face_loc)

            gt_sub_dict[filename] = face_loc
        gt_dict[i] = gt_sub_dict

    with open(cache_file, 'wb') as f:
        pickle.dump(gt_dict, f, pickle.HIGHEST_PROTOCOL)

    return gt_dict


def process_fddb_data(gt_dir):
    label_file = open("data/FDDB/train/label.txt", "w")
    for i in range(1, 11):
        filename = os.path.join(gt_dir, 'FDDB-fold-{}-ellipseList.txt'.format('%02d' % i))
        annotationfile = open(filename)
        image_dir = "data/FDDB/train/images/" + str(i) + "/"
        os.makedirs(image_dir, exist_ok=True)
        while True:
            filepath = annotationfile.readline()[:-1]
            if not filepath:
                break
            filename = filepath.replace('/', '_') + ".jpg"
            old_path = "data/FDDB/images/" + filepath + ".jpg"
            new_path = image_dir + filename
            shutil.copyfile(old_path, new_path)
            label_file.write("# " + str(i) + "/" + filename + "\n")
            line = annotationfile.readline()
            if not line:
                break
            facenum = int(line)
            for j in range(facenum):
                line = annotationfile.readline().strip().split()
                major_axis_radius = float(line[0])
                minor_axis_radius = float(line[1])
                angle = float(line[2])
                center_x = float(line[3])
                center_y = float(line[4])
                score = float(line[5])
                angle = angle / 3.1415926 * 180
                mask = np.zeros((1000, 1000), dtype=np.uint8)
                cv2.ellipse(mask, ((int)(center_x), (int)(center_y)),
                            ((int)(major_axis_radius), (int)(minor_axis_radius)), angle, 0., 360., (255, 255, 255))
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2:]
                r = cv2.boundingRect(contours[0])
                label_file.write(str(r[0]) + " " + str(r[1]) + " " + str(r[2]) + " " + str(r[3]) + " ")
                for k in range(0, 15):
                    label_file.write("-1.0" + " ")
                label_file.write("1\n")


"""
# filename
x1, x2, y1, y2 ...四个点？
"""


def read_wider_face_label():
    '''
    虽然有20个值，在训练的时候只用到了前面的14个值
    Returns:

    '''
    ann = open("data/widerface/train/label.txt", "r")
    for i in ann.readlines():
        if not i.startswith("#"):
            print(len(i.strip().split(" ")))


if __name__ == "__main__":
    # read_wider_face_label()
    process_fddb_data("FDDB_Evaluation/ground_truth")
