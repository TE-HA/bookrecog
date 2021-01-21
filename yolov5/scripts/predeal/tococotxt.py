import json
import argparse
import json
import os

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", type=str, default=None)
    parser.add_argument("save_path", type=str, default=None)
    return parser.parse_args()


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def main(args):
    json_file = args.json_file
    data = json.load(open(json_file, 'r'))
    ana_txt_save_path = args.save_path  # 保存的路径
    if not os.path.exists(ana_txt_save_path):
        os.makedirs(ana_txt_save_path)
    with tqdm(total=data['images'].__len__()) as pbar:
        for img in data['images']:
            filename = img["file_name"]
            img_width = img["width"]
            img_height = img["height"]
            img_id = img["id"]
            ana_txt_name = filename.split(".")[0] + ".txt"  # 对应的txt名字，与jpg一致
            if not os.path.exists(os.path.join(ana_txt_save_path, ana_txt_name)):
                os.mknod(os.path.join(ana_txt_save_path, ana_txt_name))
            with open(os.path.join(ana_txt_save_path, ana_txt_name), 'w') as f_txt:
                for ann in data['annotations']:
                    if ann['image_id'] == img_id:
                        box = convert((img_width, img_height), ann["bbox"])
                        f_txt.write("%s %s %s %s %s\n" % (ann["category_id"]-1, box[0], box[1], box[2], box[3]))
            pbar.update()


if __name__ == '__main__':
    main(parse_args())
