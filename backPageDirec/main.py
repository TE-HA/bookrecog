import numpy as np
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from angleDataset import AngleDataset
import torchvision
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.autograd import Variable

import os
all_path = os.path.abspath('..')


# 定义一些超参数
batch_size = 128
learning_rate = 0.001
num_epoches = 50
writer = SummaryWriter(comment='resnet')

data_tf_train = transforms.Compose(
    [
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), hue=(-0.5, 0.5)),
        transforms.RandomRotation(5, resample=False, expand=False),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

data_tf = transforms.Compose(
    [
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

f_train = '../data/back/train.txt'
f_test = '../data/back/test.txt'
f_val = '../data/back/val.txt'
model_path = '../model/angle_back_resnet18.pth'
# 训练集测试集txt

# 训练集和测试集用data_label_deal.py和MYdataset.py生成里面生成
train_data = AngleDataset(txt_path=f_train, transform=data_tf_train)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

val_data = AngleDataset(txt_path=f_val, transform=data_tf)
val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)

test_data = AngleDataset(txt_path=f_test, transform=data_tf)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size,shuffle=True)

is_train = False

# # 显示图片
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
#
# # get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
# print(labels)

def evaluate(_model, _loader):
    model.eval()
    correct = 0
    total = _loader.dataset.__len__()
    for img, label in _loader:
        img, label = img.cuda(), label.cuda()
        with torch.no_grad():
            out = model(img)
            pred = out.argmax(dim=1)

        correct += torch.eq(pred, label).sum().float().item()
    return correct / total

def make_target(label):
    target = torch.zeros((label.shape[0], 2))
    for _index in range(0, label.shape[0]):
        if label[_index] == 0:
            target[_index, 0] = 1
        else:
            target[_index, 1] = 1
    return target

# 训练模型
if is_train:
    epoch = 0
    num_batch=[]
    loss_data=[]
    ac_data=[]
    num_b=[]
    loss_best = 10
    # 定义模型
    model = models.resnet34(False)
    num_fc_ftr = model.fc.in_features
    model.fc = torch.nn.Linear(num_fc_ftr, 2)
    if torch.cuda.is_available():
        model = model.cuda()
    print(model)

    # model = torch.load(model_path)

    # 定义损失函数和优化器
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    for i in range(num_epoches):
        for data in train_loader:
            img, label = data

            model.train()

            if torch.cuda.is_available():
                img = img.cuda()
                # label = label.cuda()
            else:
                img = Variable(img)
                # label = Variable(label)

            out = model(img)
            out = nn.Sigmoid()(out)
            target = make_target(label)

            loss = criterion(out, target.cuda())
            # loss = criterion(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch += 1
            if loss.data.item() < loss_best:
                loss_best = loss.data.item()
                torch.save(model, model_path)
                print(str(loss_best)+' : save')
            if epoch % 5 == 0:
                print('epoch:{},num_batch: {}, loss: {:.4}'.format(i+1, epoch, loss.data.item()))
                writer.add_scalar('loss', loss, epoch)

        val_acc = evaluate(model, val_loader)
        print('acc: {:.4}'.format(val_acc))
        writer.add_scalar('acc', val_acc, epoch)


# 加载最优模型评估
model = torch.load(model_path)
print(evaluate(model, test_loader))

# 查找图片
model.eval()
correct = 0
total = test_loader.dataset.__len__()
for img, label in test_loader:
    img, label = img.cuda(), label.cuda()
    with torch.no_grad():
        out = model(img)
        pred = out.argmax(dim=1)

        for index in range(pred.shape[0]):
            if pred[index] != label[index]:
                imshow(img[index].cpu())

