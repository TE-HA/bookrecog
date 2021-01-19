import cv2
from ppocr.utils.utility import parse_args, check_and_read_gif, get_highest_img

cv2.imshow("win", get_highest_img(parse_args, "low.jpg"))












