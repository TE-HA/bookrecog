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

import logging
import os
import imghdr
import cv2
import paddle
import copy
from paddle import fluid
import numpy as np
import importlib


def initial_logger():
    FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    logger = logging.getLogger(__name__)
    return logger


def img_path_to_filename(img_path):
    l_img = img_path.split('/')
    return l_img[l_img.__len__() - 1]


def get_ori_img(origin_image_dir, image_name):
    return cv2.imread(origin_image_dir + image_name)


def get_image_crop_images(image, dt_boxes):
    img_crop_list = []
    for bno in range(len(dt_boxes)):
        tmp_box = copy.deepcopy(dt_boxes[bno])
        xmin, ymin = np.min(tmp_box, axis=0).astype(np.int)
        xmax, ymax = np.max(tmp_box, axis=0).astype(np.int)
        img = image[ymin:ymax, xmin:xmax]
        if img.shape[0] * 1.0 / img.shape[1] >= 1.5:
            img = np.rot90(img)
        img_crop_list.append(img)
    return img_crop_list


def get_ori_img_det_box(ori_img, det_img, dt_boxes):
    height_scale = float(ori_img.shape[0] / det_img.shape[0])
    width_scale = float(ori_img.shape[1] / det_img.shape[1])
    # dt_end = multi_box(dt_boxes, height_scale, width_scale)
    end_mat = []
    for box in dt_boxes:
        end_box = np.zeros([4, 2], dtype=np.float32)
        end_box[:, 0] = box[:, 0] * height_scale
        end_box[:, 1] = box[:, 1] * width_scale
        end_box = end_box.astype(int)
        end_box = end_box.astype(np.float32)
        end_mat.append(end_box)
    return end_mat


def create_module(module_str):
    tmpss = module_str.split(",")
    assert len(tmpss) == 2, "Error formate\
        of the module path: {}".format(module_str)
    module_name, function_name = tmpss[0], tmpss[1]
    somemodule = importlib.import_module(module_name, __package__)
    function = getattr(somemodule, function_name)
    return function


def get_check_global_params(mode):
    check_params = ['use_gpu', 'max_text_length', 'image_shape',
                    'image_shape', 'character_type', 'loss_type']
    if mode == "train_eval":
        check_params = check_params + [
            'train_batch_size_per_card', 'test_batch_size_per_card']
    elif mode == "test":
        check_params = check_params + ['test_batch_size_per_card']
    return check_params


def img_path_to_filename(img_path):
    l_img = img_path.split('/')
    return l_img[l_img.__len__() - 1]


def get_check_reader_params(mode):
    check_params = []
    if mode == "train_eval":
        check_params = ['TrainReader', 'EvalReader']
    elif mode == "test":
        check_params = ['TestReader']
    return check_params


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


def get_image_file_list(img_file):
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'GIF'}
    if os.path.isfile(img_file) and imghdr.what(img_file) in img_end:
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            file_path = os.path.join(img_file, single_file)
            if imghdr.what(file_path) in img_end:
                imgs_lists.append(file_path)
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    return imgs_lists


def check_and_read_gif(img_path):
    if os.path.basename(img_path)[-3:] in ['gif', 'GIF']:
        gif = cv2.VideoCapture(img_path)
        ret, frame = gif.read()
        if not ret:
            logging.info("Cannot read {}. This gif image maybe corrupted.")
            return None, False
        if len(frame.shape) == 2 or frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        imgvalue = frame[:, :, ::-1]
        return imgvalue, True
    return None, False


def create_multi_devices_program(program, loss_var_name):
    build_strategy = fluid.BuildStrategy()
    build_strategy.memory_optimize = False
    build_strategy.enable_inplace = True
    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_iteration_per_drop_scope = 1
    compile_program = fluid.CompiledProgram(program).with_data_parallel(
        loss_name=loss_var_name,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)
    return compile_program


def enable_static_mode():
    try:
        paddle.enable_static()
    except:
        pass
