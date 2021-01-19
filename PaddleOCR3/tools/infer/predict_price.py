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
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))

import shutil
from tqdm import tqdm
import threading
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import copy
import numpy as np
import time
from PIL import Image
import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
# from ppocr.utils.logging import get_logger
from tools.infer.utility import draw_ocr_box_txt


# logger = get_logger()


class TextSystem(object):
    def __init__(self, args):
        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

    def get_rotate_crop_image(self, img, points):
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
            img,
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
            # logger.info(bno, rec_res[bno])

    def __call__(self, img):
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        # logger.info("dt_boxes num : {}, elapse : {}".format(
        #    len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            # logger.info("cls num  : {}, elapse : {}".format(
            # len(img_crop_list), elapse))

        rec_res, elapse = self.text_recognizer(img_crop_list)
        # logger.info("rec_res num  : {}, elapse : {}".format(
        #    len(rec_res), elapse))
        # self.print_draw_crop_rec_res(img_crop_list, rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(dt_boxes, rec_res):
            text, score = rec_reuslt
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_reuslt)
        return filter_boxes, filter_rec_res


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


def img_path_to_filename(path):
    return path.split("/")[-1:][0]


def save_crop_price(img_name, imgs_4, dt_boxes, rec_res):
    # print(len(boxes)==len(txts))
    for ii in range(len(imgs_4)):
        boxes = dt_boxes[ii]
        image = imgs_4[ii]
        txts = [rec_res[ii][i][0] for i in range(len(rec_res[ii]))]
        scores = [rec_res[ii][i][1] for i in range(len(rec_res[ii]))]
        num_rec_end = len(boxes)
        price_box = None
        price_score = None
        save_to_root = None
        for index in range(num_rec_end):
            if "定价" not in txts[index] and (("元" or "圆") in txts[index] and '0' in txts[index] and "优" not in txts[index]):
                price_box = boxes[index]
                price_score = scores[index]
                save_to_root = "./inference_results/origin_price_2_server/"
            elif "定价" in txts[index]:
                price_box = boxes[index]
                price_score = scores[index]
                save_to_root = "./inference_results/origin_price_server/"
        if price_box is None:
            continue
        xmin, ymin = np.min(price_box, axis=0).astype(np.int)
        xmax, ymax = np.max(price_box, axis=0).astype(np.int)

        xmin, ymin, xmax, ymax = xmin - 30, ymin - 10, xmax + 30, ymax + 10
        # print(xmin, ymin, xmax, ymax, min(image.shape[0], ymax), min(image.shape[1], xmax))
        crop_pic = image[max(0, ymin):min(image.shape[0], ymax), max(0, xmin):min(image.shape[1], xmax)]
        if not os.path.isdir(save_to_root):
            os.makedirs(save_to_root)
        save_to = save_to_root + img_path_to_filename(img_name)
        # try:
        # print(crop_pic.shape[0], crop_pic.shape[1])
        cv2.imwrite(save_to, crop_pic)
        # except:
        #     print(img_name)
        return

    save_to_root = "./inference_results/origin_price_server_wrong/"
    if not os.path.isdir(save_to_root):
        os.mkdir(save_to_root)
    shutil.copy(img_name, save_to_root+utility.img_path_to_filename(img_name))


def split_four(img):
    height = img.shape[0]
    width = img.shape[1]
    crop_pic_1 = img[0:int(height/2), 0:int(width/2)]
    crop_pic_2 = img[0:int(height/2):, int(width/2):width]
    crop_pic_3 = img[int(height/2):height, 0:int(width/2)]
    crop_pic_4 = img[int(height/2):height, int(width/2):width]
    return [crop_pic_1, crop_pic_2, crop_pic_3, crop_pic_4]


def main(args):
    text_sys = TextSystem(args)
    image_file_list = get_image_file_list(args.image_dir)

    with tqdm(total=len(image_file_list)) as pbar:
        for image_file in image_file_list:
            img, flag = check_and_read_gif(image_file)
            if not flag:
                img = cv2.imread(image_file)
            if img is None:
                # logger.info("error in loading image:{}".format(image_file))
                continue
            starttime = time.time()

            imgs_4 = split_four(img)
            dt_boxes, rec_res = [], []
            for img_index in range(len(imgs_4)):
                dt_boxe, rec_re = text_sys(imgs_4[img_index])
                dt_boxes.append(dt_boxe)
                rec_res.append(rec_re)

            elapse = time.time() - starttime

            save_crop_price(image_file, imgs_4, dt_boxes, rec_res)
            pbar.update(1)


if __name__ == "__main__":
    main(utility.parse_args())
