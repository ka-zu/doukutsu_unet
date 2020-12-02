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
学習プログラム
"""

def imsave2(path_name, img):
    img = torchvision.utils.make_grid(img)
    img = img / 2 + 0.5
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(path_name)

def train(net, criterion, optimizer, epochs, trainloader):
    losses = []
    output_and_label = []

    #gpuを使うように設定
    device = torch.device('cuda:0' if torch.cuda.is_available() else'cpu')
    print('using device:',device)

    #ネットワークをGPU上にする
    net = net.to(device)

    for epoch in range(1, epochs+1):
        print(f'epoch: {epoch},',end='')
        running_loss = 0.0
        for counter,(img, label) in enumerate(trainloader, 1):
            optimizer.zero_grad()
            #入力のために1次元データにする
            #cpuの場合 畳み込みなのでreshapeしない
            #img = img.cpu()
            #gpuの場合
            img = img.to(device)
            label = label.to(device)
            #入力して出力
            output = net(img)
            #損失関数から計算
            loss = criterion(output, label)
            #誤差を逆伝搬
            loss.backward()
            #最適化を進める
            optimizer.step()
            #ロスを計算
            running_loss += loss.item()
        avg_loss = running_loss / counter
        losses.append(avg_loss)
        print("loss:",avg_loss)
        output_and_label.append((output,img))

        #20epochごとにモデルを保存
        # 学習途中の状態を保存する。
        if epoch % 20 == 0:
            save_folder = './doukutsu_coloring/'
            model_save_name = f'model_bat-{batch_size}_epo-{epoch}_BN_totyu.pt' 
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                save_folder + model_save_name,
            )
        
        """
        #学習途中を読み込む場合
        # 学習途中の状態を読み込む。
        checkpoint = torch.load("model.tar")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        """
    print("finished")
    return output_and_label,losses

def test(net, trainloader):
    losses = []
    output_and_label = []

    #gpuを使うように設定
    device = torch.device('cuda:0' if torch.cuda.is_available() else'cpu')
    print('using device:',device)

    #ネットワークをGPU上にする
    net = net.to(device)

    for counter,(img, label) in enumerate(trainloader, 1):
            optimizer.zero_grad()
            #入力のために1次元データにする
            #cpuの場合 畳み込みなのでreshapeしない
            #img = img.cpu()
            #gpuの場合
            img = img.to(device)
            label = label.to(device)
            #入力して出力
            output = net(img)
            #損失関数から計算
            loss = criterion(output, label)
            #誤差を逆伝搬
            loss.backward()
            #最適化を進める
            optimizer.step()
            #ロスを計算
            running_loss += loss.item()
        avg_loss = running_loss / counter
        losses.append(avg_loss)
        print("loss:",avg_loss)
        output_and_label.append((output,img))




#main関数
if __name__ == '__main__':
    
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


    #学習内容保存用
    losses = []
    output_and_label = []

    #学習パラメータ
    #Unetクラスを読み込み
    net = Doukutsu_Unet()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
    EPOCHS = 50
    
    
    #学習
    output_and_label, losses = train(net, criterion, optimizer, EPOCHS, trainloader)

    #学習したモデルの保存
    save_folder = './doukutsu_coloring/'
    model_save_name = f'model_bat-{batch_size}_epo-{EPOCHS}_BN_.pt' 
    #state_dictをつけないとGPUが使用中や替わった時にエラー
    #loadはmodel.load_state_dict(torch.load(PATH))
    torch.save(net.cpu().state_dict(), save_folder + model_save_name)

    #結果の表示(trainデータから抜粋)
    output,org = output_and_label[-1]
    #gpuを使っている場合to('cpu')でデータを戻す
    org_r = org.to('cpu')
    output_r = output.to('cpu')
    imsave2(save_folder + f'bat={batch_size}_epo={EPOCHS}_BN_original.png',org_r)
    imsave2(save_folder + f'bat={batch_size}_epo={EPOCHS}_BN_output.png',output_r)
    