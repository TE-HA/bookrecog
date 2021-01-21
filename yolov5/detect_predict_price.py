import argparse
import sys
import time
from pathlib import Path
import copy
import cv2
import os
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

sys.path.append("..")
import PaddleOCR3.ppocr.utils.utility as pppocr_utility


# import tools.infer.predict_det as predict_det
# import tools.infer.predict_rec as predict_rec
# import tools.infer.predict_cls as predict_cls
# import ppocr.utils.utility as pppocr_utility
#
#
# class TextSystem(object):
#     def __init__(self, args):
#         self.text_detector = predict_det.TextDetector(args)
#         self.text_recognizer = predict_rec.TextRecognizer(args)
#         self.text_classifier = predict_cls.TextClassifier(args)
#         self.args = args
#
#     def get_rotate_crop_image(self, img, points):
#         '''
#         img_height, img_width = img.shape[0:2]
#         left = int(np.min(points[:, 0]))
#         right = int(np.max(points[:, 0]))
#         top = int(np.min(points[:, 1]))
#         bottom = int(np.max(points[:, 1]))
#         img_crop = img[top:bottom, left:right, :].copy()
#         points[:, 0] = points[:, 0] - left
#         points[:, 1] = points[:, 1] - top
#         '''
#         img_crop_width = int(
#             max(
#                 np.linalg.norm(points[0] - points[1]),
#                 np.linalg.norm(points[2] - points[3])))
#         img_crop_height = int(
#             max(
#                 np.linalg.norm(points[0] - points[3]),
#                 np.linalg.norm(points[1] - points[2])))
#         pts_std = np.float32([[0, 0], [img_crop_width, 0],
#                               [img_crop_width, img_crop_height],
#                               [0, img_crop_height]])
#         M = cv2.getPerspectiveTransform(points, pts_std)
#         dst_img = cv2.warpPerspective(
#             img,
#             M, (img_crop_width, img_crop_height),
#             borderMode=cv2.BORDER_REPLICATE,
#             flags=cv2.INTER_CUBIC)
#         dst_img_height, dst_img_width = dst_img.shape[0:2]
#         if dst_img_height * 1.0 / dst_img_width >= 1.5:
#             dst_img = np.rot90(dst_img)
#         return dst_img
#
#     def __call__(self, img_name, img):
#         det_img = img.copy()
#         dt_boxes, elapse = self.text_detector(det_img)
#         # print("dt_boxes num : {}, elapse : {}".format(len(dt_boxes), elapse))
#         if dt_boxes is None:
#             return None, None
#         dt_boxes = pppocr_utility.sorted_boxes(dt_boxes)
#
#         # get origin image and crop
#         image_name = pppocr_utility.img_path_to_filename(img_name)
#         ori_img = pppocr_utility.get_ori_img(self.args.origin_image_dir, image_name)
#         dt_boxes = pppocr_utility.get_ori_img_det_box(ori_img, det_img, dt_boxes)
#         img_crop_list = pppocr_utility.get_image_crop_images(ori_img, dt_boxes)
#
#         img_crop_list, angle_list, elapse = self.text_classifier(
#             img_crop_list)
#         rec_res, elapse = self.text_recognizer(img_crop_list)
#         print(rec_res)
#         exit()


# def predict_price(args, text_sys_args):
#     image_file = cv2.imread(args.source)
#     text_sys = TextSystem(text_sys_args)
#     price = text_sys(args.source, image_file)
#     return price
#
#
# def parse_price_args():
#     def str2bool(v):
#         return v.lower() in ("true", "t", "1")
#
#     parser = argparse.ArgumentParser()
#     # params for prediction engine
#     parser.add_argument("--origin_image_dir", type=str, default="../data19/barcode_images/back/")
#     parser.add_argument("--use_gpu", type=str2bool, default=True)
#     parser.add_argument("--ir_optim", type=str2bool, default=True)
#     parser.add_argument("--use_tensorrt", type=str2bool, default=False)
#     parser.add_argument("--gpu_mem", type=int, default=8000)
#
#     # params for text detector
#     parser.add_argument("--image_dir", type=str)
#     parser.add_argument("--image_dirs", nargs='+', type=str)
#     parser.add_argument("--det_algorithm", type=str, default='DB')
#     parser.add_argument("--det_model_dir", type=str)
#     parser.add_argument("--det_max_side_len", type=float, default=960)
#
#     # DB parmas
#     parser.add_argument("--det_db_thresh", type=float, default=0.3)
#     parser.add_argument("--det_db_box_thresh", type=float, default=0.5)
#     parser.add_argument("--det_db_unclip_ratio", type=float, default=1.6)
#
#     # EAST parmas
#     parser.add_argument("--det_east_score_thresh", type=float, default=0.8)
#     parser.add_argument("--det_east_cover_thresh", type=float, default=0.1)
#     parser.add_argument("--det_east_nms_thresh", type=float, default=0.2)
#
#     # SAST parmas
#     parser.add_argument("--det_sast_score_thresh", type=float, default=0.5)
#     parser.add_argument("--det_sast_nms_thresh", type=float, default=0.2)
#     parser.add_argument("--det_sast_polygon", type=bool, default=False)
#
#     # params for text recognizer
#     parser.add_argument("--rec_algorithm", type=str, default='CRNN')
#     parser.add_argument("--rec_model_dir", type=str)
#     parser.add_argument("--rec_image_shape", type=str, default="3, 32, 320")
#     parser.add_argument("--rec_char_type", type=str, default='ch')
#     parser.add_argument("--rec_batch_num", type=int, default=6)
#     parser.add_argument("--max_text_length", type=int, default=25)
#     parser.add_argument(
#         "--rec_char_dict_path",
#         type=str,
#         default="./ppocr/utils/ppocr_keys_v1.txt")
#     parser.add_argument("--use_space_char", type=str2bool, default=True)
#     parser.add_argument(
#         "--vis_font_path", type=str, default="./doc/simfang.ttf")
#
#     # params for text classifier
#     parser.add_argument("--use_angle_cls", type=str2bool, default=False)
#     parser.add_argument("--cls_model_dir", type=str)
#     parser.add_argument("--cls_image_shape", type=str, default="3, 48, 192")
#     parser.add_argument("--label_list", type=list, default=['0', '180'])
#     parser.add_argument("--cls_batch_num", type=int, default=30)
#     parser.add_argument("--cls_thresh", type=float, default=0.9)
#
#     parser.add_argument("--drop_score", type=float, default=0.1)
#
#     parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
#     parser.add_argument("--use_zero_copy_run", type=str2bool, default=False)
#
#     parser.add_argument("--use_pdserving", type=str2bool, default=False)
#
#     return parser.parse_args()
#
#
#

def fit_ocr_frmart(dt_boxes):
    results = np.zeros(8, dtype=np.float32)
    dt_boxes[0], dt_boxes[1], dt_boxes[2], dt_boxes[3] = \
        dt_boxes[0] - 20, dt_boxes[1] - 20, dt_boxes[2] + 20, dt_boxes[3] + 20
    results[0], results[1] = dt_boxes[0], dt_boxes[1]
    results[2], results[3] = dt_boxes[0], dt_boxes[3]
    results[4], results[5] = dt_boxes[2], dt_boxes[3]
    results[6], results[7] = dt_boxes[2], dt_boxes[1]
    return [results.reshape(4, 2)]


def detect(opt, save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # makedirs
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    with tqdm(total=dataset.__len__()) as pbar:
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f'{n} {names[int(c)]}s, '  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        # TODO:rec_price
                        dt_boxes = []
                        if int(cls) == 1:
                            for value in xyxy:
                                dt_boxes.append(int(value.item()))
                            if len(dt_boxes) == 0:
                                continue

                            dt_boxes = fit_ocr_frmart(dt_boxes)
                            img_name = pppocr_utility.img_path_to_filename(path)
                            img_crop_list = pppocr_utility.get_image_crop_images(cv2.imread(path), dt_boxes)
                            for img_index in range(len(img_crop_list)):
                                cv2.imwrite(opt.save_dir + str(img_index) + "_" + img_name, img_crop_list[img_index])
            pbar.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--save_dir', type=str, default=None, help='save_path')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    args = parser.parse_args()
    check_requirements()

    with torch.no_grad():
        if args.update:  # update all models (to fix SourceChangeWarning)
            for args.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect(args)
                strip_optimizer(args.weights)
        else:
            detect(args)
