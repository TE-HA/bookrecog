import numpy as np
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from angleDataset import AngleDataset
import torchvision
# from tensorboardX import SummaryWriter
# import matplotlib.pyplot as plt
from torch import nn, optim
from torch.autograd import Variable
# import albumentations as A
import os
import pandas as pd
from PIL import Image

all_path = os.path.abspath('..')

data_tf = transforms.Compose(
    [
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

f_front = all_path+'/bag/photo_front.txt'
f_back = all_path+'/bag/photo_back.txt'

model_path_back = '../model/angle_back_resnet18.pth'

model_path_front = '../model/angle_front_resnet18.pth'


test_front_data = AngleDataset(txt_path=f_front, transform=data_tf)
test_front_loader = DataLoader(dataset=test_front_data, batch_size=32,shuffle=True)

test_back_data = AngleDataset(txt_path=f_back, transform=data_tf)
test_back_loader = DataLoader(dataset=test_back_data, batch_size=32,shuffle=True)



# 加载最优模型评估
model_front = torch.load(model_path_front)
model_back = torch.load(model_path_back)
model_front.eval()
model_back.eval()

# 单张
# df_front = pd.read_csv(f_front, sep=',', names=['location', 'label'])
# df_front['label'] = df_front['label'].astype(int)
# # df = df_front.sample(frac=1).reset_index(drop=True)
#
# df_back = pd.read_csv(f_back, sep=',', names=['location', 'label'])
# df_back['label'] = df_back['label'].astype(int)
# # df = df_front.sample(frac=1).reset_index(drop=True)
# print(df_front.loc[0]['label'])
#
# def test(df_1, df_2):
#     """"""
#     both_correct = 0
#     back = 0
#     front = 0
#     wrong = 0
#     for index in range(df_1.shape[0]):
#         img_f = data_tf(Image.open(df_1.loc[index]['location']).convert('RGB')).unsqueeze(0)
#         label_f = df_1.loc[index]['label']
#         img_f, label_f = img_f.cuda(), label_f
#
#         img_b = data_tf(Image.open(df_2.loc[index]['location']).convert('RGB')).unsqueeze(0)
#         label_b = df_2.loc[index]['label']
#         img_b, label_b = img_b.cuda(), label_b
#         with torch.no_grad():
#             out_f = model_front(img_f)
#             out_f = nn.Softmax(out_f)
#             out_b = model_back(img_b)
#             out_b = nn.Softmax(out_b)
#         if out_f.argmax(dim=1) == label_f and out_b.argmax(dim=1) == label_b:
#             both_correct += 1
#         elif out_f.argmax(dim=1) != label_f and out_b.argmax(dim=1) == label_b:
#             if out_f.max(dim=0) <= out_b.max(dim=0):
#                 back += 1
#         elif out_f.argmax(dim=1) == label_f and out_b.argmax(dim=1) != label_b:
#             if out_f.max(dim=0) >= out_b.max(dim=0):
#                 front += 1
#         else:
#             wrong += 1
#
#     return both_correct, front, back, wrong, df_1.shape[0]
#
#
# both_correct, front, back, wrong, total = test(df_front, df_back)
# print(both_correct)
# print(front)
# print(back)
# print(wrong)
# print(total)
# print('{:.4f}'.format((both_correct+front+back)/total))

# 查找图片
total = test_front_loader.dataset.__len__()

out_all_front = []
out_all_front_label = []
for img, label in test_front_loader:
    img, label = img.cuda(), label.cuda()
    with torch.no_grad():
        out = model_front(img)
        out_all_front.append(out)
        out_all_front_label.append(label)

out_all_back = []
out_all_back_label = []
for img, label in test_back_loader:
    img, label = img.cuda(), label.cuda()
    with torch.no_grad():
        out = model_back(img)
        out_all_back.append(out)
        out_all_back_label.append(label)


correct = 0
back = 0
front = 0
wrong = 0
correct_dim = 0
for index in range(len(out_all_front)):
    front_out = out_all_front[index]
    back_out = out_all_back[index]
    front_label = out_all_front_label[index]
    back_label = out_all_back_label[index]

    front_pre = front_out.argmax(dim=1)
    back_pre = back_out.argmax(dim=1)

    for count in range(front_pre.shape[0]):
        if front_pre[count] == front_label[count] and back_pre[count] == back_label[count]:
            correct += 1
        elif front_pre[count] != front_label[count] and back_pre[count] == back_label[count]:
            if back_out[count].max(dim=0) >= front_out[count].max(dim=0):
                back += 1
                correct_dim += 1
        elif front_pre[count] == front_label[count] and back_pre[count] != back_label[count]:
            if back_out[count].max(dim=0) <= front_out[count].max(dim=0):
                front += 1
                correct_dim += 1
        else:
            wrong += 1

print(correct)
print(front)
print(back)
print(wrong)
print(correct_dim)
print('{:.4f}'.format((correct+correct_dim)/total))
