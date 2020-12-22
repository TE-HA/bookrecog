import cv2
import os
import numpy as np

f = open('/home/teha/code/CV-OCR/data/front/all.txt', 'w')

pth0 = "/home/teha/code/CV-OCR/data/front/front_up"
pth1 = "/home/teha/code/CV-OCR/data/front/front_down"

# 翻转操作
# for root, dirs, files in os.walk(pth0, True):
#     for file in files:
#         img = cv2.imread(root+'/'+file)
#         # img2 =np.rot90(img)
#         img2 = cv2.flip(img, 0)
#         cv2.imwrite(pth1+'/'+file, img2)

# 写入all文件
for root, dirs, files in os.walk(pth0, True):
    for file in files:
        f.writelines(pth0 +'/'+ file + '\x20' + '0')
        f.write('\n')

for root, dirs, files in os.walk(pth1, True):
    for file in files:
        f.writelines(pth1 +'/'+ file + '\x20' + '1')
        f.write('\n')


pth4 = '/home/teha/code/CV-OCR/data/front/all.txt'

import pandas as pd

df = pd.read_csv(pth4, sep=' ', names=['location', 'label'])
df = df[:-1]
df['label'] = df['label'].astype(int)
df = df.sample(frac=1).reset_index(drop=True)

df_train = df.iloc[df.index < 1750]
df_train.to_csv("/home/teha/code/CV-OCR/data/front/train.txt", header=False,index=False)

df_val = df.iloc[1750:2250]
df_val.to_csv("/home/teha/code/CV-OCR/data/front/val.txt", header=False,index=False)

df_test = df.iloc[df.index >= 2250]
df_test.to_csv("/home/teha/code/CV-OCR/data/front/test.txt", index=False,header=False)
