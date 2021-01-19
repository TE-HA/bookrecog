# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import shutil
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))

from tqdm import tqdm

sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

import tools.infer.utility as utility
from ppocr.utils.utility import initial_logger

logger = initial_logger()
import cv2
from skimage import io
import tools.infer.predict_det as predict_det
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_cls as predict_cls
import copy
import numpy as np
import math
import time
from ppocr.utils.utility import get_image_file_list, check_and_read_gif, get_ori_img_det_box
from PIL import Image
from tools.infer.utility import draw_ocr
from tools.infer.utility import draw_ocr_box_txt


class TextSystem(object):
    def __init__(self, args):
        self.args = args
        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        # if self.use_angle_cls:
        self.text_classifier = predict_cls.TextClassifier(args)

    def get_rotate_crop_image(self, ori_img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))

        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            ori_img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def print_draw_crop_rec_res(self, img_crop_list, rec_res):
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite("./output/img_crop_%d.jpg" % bno, img_crop_list[bno])
            # print(bno, rec_res[bno])

    def __call__(self, img_name, img):
        det_img = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        # print("dt_boxes num : {}, elapse : {}".format(len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None
        dt_boxes = sorted_boxes(dt_boxes)
        # img_crop_list = []
        # for bno in range(len(dt_boxes)):
        #     tmp_box = copy.deepcopy(dt_boxes[bno])
        #     img_crop = self.get_rotate_crop_image(det_img, tmp_box)
        #     img_crop_list.append(img_crop)
        # img_crop_list = get_image_crop_images(det_img, dt_boxes)
        # save_img("det_img", img_crop_list)
        # cv2.imwrite("tools/infer/det_img/a.jpg", det_img)

        image_name = utility.img_path_to_filename(img_name)
        ori_img = cv2.imread(self.args.origin_image_dir + image_name)
        dt_boxes = get_ori_img_det_box(ori_img, det_img, dt_boxes)
        img_crop_list = get_image_crop_images(ori_img, dt_boxes)
        # img_crop_list = []
        # for bno in range(len(dt_boxes)):
        #     tmp_box = copy.deepcopy(dt_boxes[bno])
        #     img_crop = self.get_rotate_crop_image(det_img, tmp_box)
        #     img_crop_list.append(img_crop)
        # save_img("ori_img", img_crop_list)
        # cv2.imwrite("tools/infer/ori_img/a.jpg", ori_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        img_crop_list, angle_list, elapse = self.text_classifier(
            img_crop_list)
        rec_res, elapse = self.text_recognizer(img_crop_list)
        pre_end, score = judge_direction(dt_boxes, angle_list, rec_res)
        # print("pre_end: "+str(pre_end))
        return pre_end, score


def get_image_crop_images(image, dt_boxes):
    img_crop_list = []
    for bno in range(len(dt_boxes)):
        tmp_box = copy.deepcopy(dt_boxes[bno])
        xmin, ymin = np.min(tmp_box, axis=0).astype(np.int)
        xmax, ymax = np.max(tmp_box, axis=0).astype(np.int)
        img = image[ymin:ymax, xmin:xmax]
        # print(tmp_box)
        # xmax, ymax = int(tmp_box[3][0]), int(tmp_box[3][1])
        # xmin, ymin = int(tmp_box[1][0]), int(tmp_box[1][1])
        # # print(xmax, ymax, xmin, ymin)
        # img_crop = image.crop([ymin, xmax, ymax, xmin])
        # img = cv2.cvtColor(np.array(img_crop), cv2.COLOR_RGB2BGR)
        # dst_img_height, dst_img_width = img.shape[0:2]
        if img.shape[0] * 1.0 / img.shape[1] >= 1.5:
            img = np.rot90(img)
        img_crop_list.append(img)
    return img_crop_list


def save_img(path, img_crop_list):
    for box_index in range(len(img_crop_list)):
        if not os.path.exists("tools/infer/"+path):
            os.makedirs("tools/infer/"+path)
        # print("saving...")
        cv2.imwrite("tools/infer/"+path+"/"+str(box_index)+".jpg", img_crop_list[box_index])
        # img_crop_list[box_index].save("tools/infer/ori_img/"+str(box_index)+".jpg")
    print("down...")


def cac_area(eve_box):
    # print(eve_box)
    xmin, ymin = np.min(eve_box, axis=0).astype(np.int)
    xmax, ymax = np.max(eve_box, axis=0).astype(np.int)
    area = (xmax - xmin) * (ymax - ymin)
    # print(area)
    return area


def judge_direction(dt_boxes, angle_list, rec_res):
    scores = [rec_res[i][1] for i in range(len(rec_res))]
    # print(scores)
    pre_label = [0 if label[0] == "0" else 1 for label in angle_list]
    pre_end = 0
    area_list_up, area_list_down, score_up, score_down = [], [], [], []
    num_dt_boxes = len(dt_boxes)
    # print("angle_list")
    # print(angle_list)
    for drop in [0.8, 0]:
        for eve_box_index in range(num_dt_boxes):
            # print(angle_list[eve_box_index][1])
            multi = 1
            if angle_list[eve_box_index][1] == 1.0:
                multi = 10
            if scores[eve_box_index] <= drop or angle_list[eve_box_index][1] < drop:
                continue
            if angle_list[eve_box_index][0] == "0":
                area_list_up.append(
                    multi * cac_area(dt_boxes[eve_box_index]) * scores[eve_box_index] * angle_list[eve_box_index][1] ** 2)
                score_up.append(scores[eve_box_index])
            else:
                area_list_down.append(
                    multi * cac_area(dt_boxes[eve_box_index]) * scores[eve_box_index] * angle_list[eve_box_index][1] ** 2)
                score_down.append(scores[eve_box_index])
        if len(area_list_up) == 0 and len(area_list_down) == 0:
            continue
        else:
            break
    # print(area_list_up)
    # print(area_list_down)

    if np.sum(np.array(area_list_up)) >= np.sum(np.array(area_list_down)):
        return 0, np.sum(np.array(score_up)/len(score_up))
    else:
        return 1, np.sum(np.array(score_down)/len(score_down))


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


def main(args):
    image_file_list = []
    for img_dir in args.image_dirs:
        image_file_list += get_image_file_list(img_dir)
    text_sys = TextSystem(args)
    starttime = time.time()
    right = 0
    with tqdm(total=len(image_file_list)) as pbar:
        for image_file_name in image_file_list:
            img, flag = check_and_read_gif(image_file_name)
            if not flag:
                img = cv2.imread(image_file_name)
            if img is None:
                continue

            # predict
            pre_end, score = text_sys(image_file_name, img)

            # judge
            if 'inverse' in image_file_name and pre_end == 1:
                right += 1
            elif 'inverse' not in image_file_name and pre_end == 0:
                right += 1
            else:
                if not os.path.exists("./inference_results/cls_wrong"):
                    os.mkdir("./inference_results/cls_wrong")
                shutil.copy(image_file_name, "./inference_results/cls_wrong/" + utility.img_path_to_filename(image_file))
                print(image_file_name)

            pbar.update(1)
    # pre_time
    elapse = time.time() - starttime
    # acc
    total = len(image_file_list)
    acc = float(right / total)
    print("acc: {:.8f}".format(acc))
    print("time: {:.1f}s".format(elapse))


if __name__ == "__main__":
    main(utility.parse_args())
