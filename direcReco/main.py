import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from angleDataset import AngleDataset
import net
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.autograd import Variable

# 定义一些超参数
batch_size = 32
learning_rate = 0.0001
num_epoches = 50

data_tf = transforms.Compose(
    [
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

f_train = '/home/teha/code/CV-OCR/data/front/train.txt'
f_test = '/home/teha/code/CV-OCR/data/front/test.txt'
f_val = '/home/teha/code/CV-OCR/data/front/val.txt'
model_path = '/home/teha/code/CV-OCR/model/angle_right.pth'
# 训练集测试集txt

# 训练集和测试集用data_label_deal.py和MYdataset.py生成里面生成
train_data = AngleDataset(txt_path=f_train, transform=data_tf)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

val_data = AngleDataset(txt_path=f_val, transform=data_tf)
val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)

test_data = AngleDataset(txt_path=f_test, transform=data_tf)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

is_train = True
# model = net.ResNet18()
# if torch.cuda.is_available():
#     model = model.cuda()
# print(model)
model = torch.load(model_path)
model.eval()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 训练模型
if is_train:
    epoch = 0
    num_batch=[]
    loss_data=[]
    ac_data=[]
    num_b=[]
    loss_best = 1
    for i in range(num_epoches):
        for data in train_loader:
            img, label = data

            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()
            else:
                img = Variable(img)
                label = Variable(label)

            out = model(img)
            loss = criterion(out, label)
            print_loss = loss.data.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch += 1
            if loss.data.item() < loss_best:
                loss_best = loss.data.item()
                torch.save(model, model_path)
                print(str(loss_best)+' : save')
            if epoch % 5 == 0:
                num_batch.append(epoch)
                loss_data.append(loss.data.item())
                print('epoch:{},num_batch: {}, loss: {:.4}'.format(i+1, epoch, loss.data.item()))

        eval_loss = 0
        eval_acc = 0
        for data in val_loader:
            img, label = data
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()

            out = model(img)
            loss = criterion(out, label)
            eval_loss += loss.data.item() * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()
            eval_acc += num_correct.item()

        # print('epoch:{},lr_rate:{},batch_size:{}'.format(num_epoches, learning_rate, batch_size))
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(val_data)), eval_acc / (len(val_data))))
        num_b.append(epoch)
        ac_data.append(float(eval_acc / (len(val_data))))

# 加载最优模型评估
model = torch.load(model_path)
model.eval()
eval_loss = 0
eval_acc = 0
for data in test_loader:
    img, label = data
    if torch.cuda.is_available():
        img = img.cuda()
        label = label.cuda()

    out = model(img)
    loss = criterion(out, label)
    eval_loss += loss.data.item() * label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    eval_acc += num_correct.item()

print('epoch:{},lr_rate:{},batch_size:{}'.format(num_epoches, learning_rate, batch_size))
print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_data)), eval_acc / (len(test_data))))


# loss曲线生成
if is_train:
    plt.plot(num_batch, loss_data, c='r')
    plt.xlabel('num_batch')
    plt.ylabel('loss_data')
    plt.show()

    plt.plot(num_b, ac_data, c='r')
    plt.xlabel('num_b')
    plt.ylabel('ac_data')
    plt.show()
