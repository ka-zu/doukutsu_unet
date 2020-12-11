import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
#洞窟画像を読み込むデータセットクラス
class Doukutsu_Dataset(torch.utils.data.Dataset):
    #コンストラクタ
    def __init__(self, train=True, data_trans=None, label_trans=None):        
        
        self.data_trans = data_trans
        self.label_trans = label_trans
        
        if train:
            self.img_dir = './data/doukutsu_img'
        else:
            self.img_dir = './data/doukutsu_test'
        
        self.img_paths = [str(p) for p in Path(self.img_dir).glob('*.jpg')]
        #print(self.img_paths)
        #__len__の返り値
        self.data_num = len(self.img_paths)

    #データ数を返す
    def __len__(self):
        return self.data_num

    #学習時の読み込みメソッド
    def __getitem__(self, idx):
        p = self.img_paths[idx]
        img = Image.open(p)

        out_data = img
        out_label = img

        if self.data_trans:
            out_data = self.data_trans(out_data)

        if self.label_trans:
            out_label = self.label_trans(out_label)

        return out_data, out_label

    def get_paths(self):
        return self.img_paths


def imshow(img):
    img = torchvision.utils.make_grid(img)
    img = img / 2 + 0.5
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#試しのメイン関数
if __name__ == "__main__":

    data_t = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,),(0.5,))
        ])
        
    label_t = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ])

    dataset = Doukutsu_Dataset(data_trans=data_t, label_trans=label_t)
    print(len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)

    print(len(dataloader))

    it = iter(dataloader)
    img, label = next(it)
    
    imshow(img)
    imshow(label)

    



