import torch
import torch.nn as nn
import pytorch_lightning as pyl
import cv2


class lstm_cnn(pyl.LightningModule):
    cov11 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(8, 8), stride=2)
    cov22 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(8, 8), stride=2)
    cov33 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(8, 8), stride=2)
    cov44 = nn.Conv2d(in_channels=128, out_channels=150, kernel_size=(11, 11), stride=1)
    cov4 = nn.Conv2d(in_channels=3, out_channels=512, kernel_size=(3, 3),padding=1)
    poll = nn.AvgPool2d(kernel_size=(2, 2))

    def __init__(self):
        super().__init__()
        # self.cov1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(8, 8), stride=2)
        # self.cov2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(8, 8), stride=2)
        # self.cov3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(8, 8), stride=2)
        # self.cov4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(8, 8), stride=2)
        # self.avp = nn.AvgPool2d(kernel_size=(2, 2))
        self.cov = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(8, 8), stride=2),
                                 nn.LeakyReLU(),
                                 nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(8, 8), stride=2),
                                 nn.LeakyReLU(),
                                 nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(8, 8), stride=2),
                                 nn.LeakyReLU(),
                                 nn.Conv2d(in_channels=128, out_channels=150, kernel_size=(11, 11), stride=1),
                                 nn.LeakyReLU(),nn.AvgPool2d(kernel_size=(2, 2)))
        self.lstmm =nn.LSTM(input_size=64,hidden_size=128,num_layers=6)
        self.nne =nn.Embedding(36,128)

    def forward(self, x):
        pass
        # nn.Embedding()

aa = cv2.imread('11112.png')
image = cv2.cvtColor(aa, cv2.COLOR_BGR2RGB)  # 转化为RGB，也可以用img = img[:, :, (2, 1, 0)]
# 这时的image是H,W,C的顺序，因此下面需要转化为：C, H, W
image = torch.FloatTensor(image).permute(2, 0, 1) / 255

a = lstm_cnn
b = a.cov11(image)
bb =a.cov4(image)
print(bb)
# c = a.cov22(b)
# c = a.cov33(c)
# c = a.cov44(c)
# c : torch.Tensor = a.poll(c)
# # print(c)
# lstmm =nn.LSTM(input_size=64,hidden_size=128,num_layers=6,batch_first=True)
# cvff=c.view(150,64)
# for i in cvff:
#     print(i)
#     aaa = lstmm(i.view(1,64))
#     print(aaa)
