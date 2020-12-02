import torch
import torch.nn as nn

"""
研究用
128*128の画像を学習するUnetクラス
"""

#U-netで作ったオートエンコーダ
class Doukutsu_Unet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        ###エンコーダ
        #1層
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(num_features=64,affine=False)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(num_features=64,affine=False)
        self.relu1_2 = nn.ReLU()

        #層移動
        self.pool1_2 = torch.nn.MaxPool2d(2)

        #2層
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(num_features=128,affine=False)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(num_features=128,affine=False)
        self.relu2_2 = nn.ReLU()

        #層移動
        self.pool2_3 = torch.nn.MaxPool2d(2)

        #3層
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(num_features=256,affine=False)
        self.relu3_1 = nn.ReLU()
        self.bn3_2 = nn.BatchNorm2d(num_features=256,affine=False)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU()

        #層移動
        self.pool3_contact = torch.nn.MaxPool2d(2)

        #contacting path
        #4層
        self.convC_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bnC_1 = nn.BatchNorm2d(num_features=512,affine=False)
        self.reluC_1 = nn.ReLU()
        self.convC_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bnC_2 = nn.BatchNorm2d(num_features=512,affine=False)
        self.reluC_2 = nn.ReLU()


        ###デコーダ
        #層移動
        self.convT_1 = torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)

        #5層    
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(num_features=256,affine=False)
        self.relu5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(num_features=256,affine=False)
        self.relu5_2 = nn.ReLU()

        #層移動
        self.convT_2 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        
        #6層
        self.conv6_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.bn6_1 = nn.BatchNorm2d(num_features=128,affine=False)
        self.relu6_1 = nn.ReLU()
        self.conv6_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn6_2 = nn.BatchNorm2d(num_features=128,affine=False)
        self.relu6_2 = nn.ReLU()

        #層移動
        self.convT_3 = torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
    
        #7層
        self.conv7_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.bn7_1 = nn.BatchNorm2d(num_features=64,affine=False)
        self.relu7_1 = nn.ReLU()
        self.conv7_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn7_2 = nn.BatchNorm2d(num_features=64,affine=False)
        self.relu7_2 = nn.ReLU()

        ###出力
        self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)
    
    def forward(self, x):
        h = x
        h = self.relu1_1(self.bn1_1(self.conv1_1(h)))
        h = self.relu1_2(self.bn1_2(self.conv1_2(h)))
        skip1 = h

        h = self.pool1_2(h)

        h = self.relu2_1(self.bn2_1(self.conv2_1(h)))
        h = self.relu2_2(self.bn2_2(self.conv2_2(h)))
        skip2 = h

        h = self.pool2_3(h)

        h = self.relu3_1(self.bn3_1(self.conv3_1(h)))
        h = self.relu3_2(self.bn3_2(self.conv3_2(h)))
        skip3 = h

        h = self.pool3_contact(h)

        h = self.reluC_1(self.bnC_1(self.convC_1(h)))
        h = self.reluC_2(self.bnC_2(self.convC_2(h)))

        h = self.convT_1(h)

        #次元の結合 tensors:結合したいデータ dim:結合する次元
        h = torch.cat(tensors=(h,skip3),dim=1)
        h = self.relu5_1(self.bn5_1(self.conv5_1(h)))
        h = self.relu5_2(self.bn5_2(self.conv5_2(h)))

        h = self.convT_2(h)

        h = torch.cat(tensors=(h,skip2),dim=1)
        h = self.relu6_1(self.bn6_1(self.conv6_1(h)))
        h = self.relu6_2(self.bn6_2(self.conv6_2(h)))

        h = self.convT_3(h)

        h = torch.cat(tensors=(h,skip1),dim=1)
        h = self.relu7_1(self.bn7_1(self.conv7_1(h)))
        h = self.relu7_2(self.bn7_2(self.conv7_2(h)))

        h = torch.tanh(self.conv_output(h))

        return h