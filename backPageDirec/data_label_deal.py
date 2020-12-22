# import cv2
import os
import numpy as np
import os
all_path = os.path.abspath('..')
# print(all_path)
f = open(all_path+'/data/back/all.txt', 'w')
#
pth0 = all_path+'/data/back/back_up'
pth1 = all_path+'/data/back/back_down'

# 旋转图片
# for root, dirs, files in os.walk(pth0, True):
#     for file in files:
#         img = cv2.imread(root+'/'+file)
#         # img2 = cv2.flip(img, 0)
#         img2 =np.rot90(img)
#         img2 =np.rot90(img2)
#         cv2.imwrite(pth1+'/down_'+file, img2)

for root, dirs, files in os.walk(pth0, True):
    for file in files:
        f.writelines(pth0 +'/'+ file + ',' + '0')
        f.write('\n')

for root, dirs, files in os.walk(pth1, True):
    for file in files:
        f.writelines(pth1 +'/'+ file + ',' + '1')
        f.write('\n')

pth4 = all_path+'/data/back/all.txt'

import pandas as pd

df = pd.read_csv(pth4, sep=',', names=['location', 'label'])
df = df[:-1]
df['label'] = df['label'].astype(int)
df = df.sample(frac=1).reset_index(drop=True)

df_train = df.iloc[df.index < 2100]
df_train.to_csv(all_path+'/data/back/train.txt', header=False,index=False)

df_val = df.iloc[2100:2700]
df_val.to_csv(all_path+'/data/back/val.txt', header=False,index=False)

df_test = df.iloc[df.index >= 2700]
df_test.to_csv(all_path+'/data/back/test.txt', index=False,header=False)
