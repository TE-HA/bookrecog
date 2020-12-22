# import cv2
import os
import numpy as np
import os
all_path = os.path.abspath('..')
# print(all_path)
f = open('/raid/.other_users/trainee/users/xys/bookrecog/data/front/all.txt', 'w')
#
pth0 = "/raid/.other_users/trainee/users/xys/bookrecog/data/front/front_up"
pth1 = "/raid/.other_users/trainee/users/xys/bookrecog/data/front/front_down"

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

pth4 = '/raid/.other_users/trainee/users/xys/bookrecog/data/front/all.txt'

import pandas as pd

df = pd.read_csv(pth4, sep=' ', names=['location', 'label'])
df = df[:-1]
df['label'] = df['label'].astype(int)
df = df.sample(frac=1).reset_index(drop=True)

df_train = df.iloc[df.index < 2100]
df_train.to_csv("/raid/.other_users/trainee/users/xys/bookrecog/data/front/train.txt", header=False,index=False)

df_val = df.iloc[2100:2700]
df_val.to_csv("/raid/.other_users/trainee/users/xys/bookrecog/data/front/val.txt", header=False,index=False)

df_test = df.iloc[df.index >= 2700]
df_test.to_csv("/raid/.other_users/trainee/users/xys/bookrecog/data/front/test.txt", index=False,header=False)
