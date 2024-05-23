import json
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import time
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

from CSRNet_RGBT.csrnet_rgbt import CSRNet_RGBT
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Res50.model.Res50 import Res50
from ECAN.model import CANNet
from train import ImgDataset

# test_path = "./dataset/train/rgb/"
# test_tir_path = "./dataset/train/tir/"
# gt_path = "./dataset/train/hdf5s/"
# img_paths = [f"{test_path}{i}.jpg" for i in range(1, 1001)]
# tir_img_paths = [f"{test_tir_path}{i}R.jpg" for i in range(1, 1001)]
# gt_paths = [f"{gt_path}{i}.h5" for i in range(1, 1001)]

img_dir = "./dataset/train/rgb/"
tir_img_dir = "./dataset/train/tir/"
gt_dir = "./dataset/train/hdf5s/"

model = CSRNet_RGBT()
# model = CANNet()
model = model.cuda()
# ./best/model_best.pth7.5.tar
checkpoint = torch.load("./model/model_best.pth.tar")
model.load_state_dict(checkpoint["state_dict"])

# for i in range(len(img_paths)):
#     img = 255.0 * F.to_tensor(Image.open(img_paths[i]).convert('RGB'))

#     img[0, :, :] = img[0, :, :]-92.8207477031
#     img[1, :, :] = img[1, :, :]-95.2757037428
#     img[2, :, :] = img[2, :, :]-104.877445883
#     img = img.cuda()
#     output = model(img.unsqueeze(0))
#     ans = output.detach().cpu().sum()
#     ans = "{:.2f}".format(ans.item())
#     print(f"{i+1},{ans}")

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.397], std=[0.229, 0.224, 0.225, 0.181]),
    ]
)

dataset = ImgDataset(
        img_dir, tir_img_dir, gt_dir, shuffle=True, transform=transform, train=True
    )
val_loader = DataLoader(
    dataset, batch_size=1, shuffle=False, num_workers=4
)

def validate(model, val_loader):
    """
    在验证集上评估模型的性能
    """

    print("begin test")

    # 将模型设置为评估模式
    model.eval()
    # 初始化 MAE（Mean Absolute Error）
    mae = 0

    # 迭代验证数据加载器中的每个批次
    for i, (img, target) in enumerate(val_loader):
        # 将输入图像移动到 GPU 上
        img = img.cuda()
        # 将输入图像包装成 Variable 对象
        img = Variable(img)
        # 将输入图像传递给模型，获取模型的输出
        output = model(img)

        # 计算预测值和目标值的绝对值误差，并累加到 MAE 中
        mae += abs(output.data.sum() - target.sum().type(torch.FloatTensor).cuda())

    # 计算平均 MAE
    mae = mae / (len(val_loader))
    # 打印平均 MAE
    print(" * MAE {mae:.3f} ".format(mae=mae))

    return mae

if __name__ == '__main__':
    validate(model, val_loader)