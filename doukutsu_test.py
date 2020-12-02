import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

#自作Unetクラスのインポート
from doukutsu_unet import Doukutsu_Unet
from doukutsu_dataset import Doukutsu_Dataset

"""
研究用
保存したモデルを用いて結果を求める。
"""

def imsave2(path_name, img):
    img = torchvision.utils.make_grid(img)
    img = img / 2 + 0.5
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(path_name)

if __name__ == '__main__':
    model_path = "./doukutsu_coloring/model_bat-20_epo-50_BN_.pt"

    #gpuを使うように設定
    device = torch.device('cuda:0' if torch.cuda.is_available() else'cpu')
    print('using device:',device)

    model = Doukutsu_Unet()
    #GPUで学習したらこう読み込む
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    #訓練データのtransform　グレースケールと正規化変換
    data_t = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,),(0.5,))
        ])
        
    #正解データのtransform　正規化変換
    label_t = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ])

    #データセットの読み込み
    dataset = Doukutsu_Dataset(data_trans=data_t, label_trans=label_t)
    trainset, testset = torch.utils.data.random_split(dataset, [len(dataset)-100, 100])
    print(len(dataset))
    print(len(trainset))
    print(len(testset))

    #データローダ
    batch_size = 20
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)

    #推論モード
    model.eval()

    save_folder = './doukutsu_coloring/'
    for counter,(img, label) in enumerate(testloader, 1):
        output = model(img)


        imsave2(save_folder + f'result_output_{counter}.png',output)
        imsave2(save_folder + f'result_label_{counter}.png',label)


