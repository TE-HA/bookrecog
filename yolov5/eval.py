import os
from tqdm import tqdm
import argparse
import multiprocessing as mp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir_before", type=str, default=None)
    parser.add_argument("--img_dir_after", type=str, default=None)
    parser.add_argument("--save_fault_txt_path", type=str, default=None)
    return parser.parse_args()


def get_img_file_list(img_dir):
    return os.listdir(img_dir)


def main(args):
    imgs_list_before = get_img_file_list(args.img_dir_before)
    imgs_list_after = get_img_file_list(args.img_dir_after)
    print(imgs_list_before[:2])
    print(imgs_list_after[:2])
    num_imgs_list = len(imgs_list_before)
    right = 0
    fault_det_imgs = []
    with tqdm(total=num_imgs_list) as pbar:
        for index in range(num_imgs_list):
            if "0_"+imgs_list_before[index] in imgs_list_after:
                right += 1
            else:
                fault_det_imgs.append(imgs_list_before[index])
            pbar.update()

    # print(fault_imgs)
    print("acc: {:.8f}".format(right/num_imgs_list))
    if args.save_fault_txt_path:
        if not os.path.exists(args.save_fault_txt_path):
            os.mknod(args.save_fault_txt_path)
        with open(args.save_fault_txt_path, 'w') as f:
            for fault_img in fault_det_imgs:
                f.write(fault_img+"\n")


if __name__ == '__main__':
    main(parse_args())
