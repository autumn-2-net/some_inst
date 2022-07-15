import random
from io import BytesIO

import cv2
import numpy as np
import pytorch_lightning as pyl
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from tqdm import tqdm

from base_modle.modle_pack import part_of_main_modle
from base_modle.bert_modle import main_config
import  json
from base_modle.res_modle import res_modle
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
from fontTools.ttLib import TTFont
from PIL import ImageFont, Image, ImageDraw
from torchvision import transforms
class configs(main_config):
    n_head =8

# # ###############################test######################## #
#
# a ='三'
# bhl=['㇐','㇐','㇐']
# l = len(bhl)
# mask=[]
# aa =configs().n_vebs
# if l<aa:
#     bhl.extend('[pad]' for _ in range(aa -l))
# if l==aa:
#     mask.extend(1 for _ in range(aa))
# else:
#     cccsd = []
#     cccsd.extend(1 for _ in range(l))
#     cccsd2 = []
#     cccsd2.extend(0 for _ in range(aa -l))
#     mask =cccsd+cccsd2
#
# print(bhl)
#
# with open('映射.json','r',encoding='utf-8') as f:
#     aas =f.read()
# ccd =json.loads(aas)
# linss = []
# for i in bhl:
#     asas =ccd.get(i)
#     if asas is None:
#         exit('errrrrrrrrr')
#     linss.append(asas)
# print(linss)
# image = cv2.imdecode(np.fromfile('test.png', dtype=np.uint8), 1)  # 1：彩色；2：灰度
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转化为RGB，也可以用img = img[:, :, (2, 1, 0)]
# # 这时的image是H,W,C的顺序，因此下面需要转化为：C, H, W
# # image = torch.from_numpy(image).permute(2, 0, 1)
# image = torch.from_numpy(image).permute(2, 0, 1) / 255
# image = image.unsqueeze(0)
# models = part_of_main_modle(configs())
# linss1 = torch.tensor(linss,dtype=torch.long).unsqueeze(0)
# masks =torch.tensor(mask,dtype=torch.long).unsqueeze(0)
# aaa =models(image,linss1,masks)
# print(aaa)
# # nn.Linear(config.n_embd, config.vocab_size, bias=False)

# # #############################################end################################## #
class dataloadd(Dataset):
    def __init__(self,len11,v_words_path,mapping_path,path_ttf,config):
        self.lenrr=len11
        self.config =config
        with open(v_words_path, 'r', encoding='utf-8') as f:
            # words = f.read()
            words=json.loads(f.read())
        with open(mapping_path, 'r', encoding='utf-8') as f:
            # words = f.read()
            mappi=json.loads(f.read())
        self.ttf_name =os.listdir(path_ttf)
        self.ttf_path=path_ttf
        self.ttf_name_len =len(self.ttf_name)

        jsonnn = {}
        for i in words:
            list1 = []
            cc = words[i]
            for j in cc:
                asda = mappi.get(j)
                if asda is None:
                    list1.append(47)
                    continue
                list1.append(asda)
            jsonnn[i] = list1
        self.wordd =jsonnn
        ppp =''
        for i in words:
            ppp =ppp+i
        self.worddlist =ppp
        self.worddlist_len=len(self.worddlist)
        self.ttf_bet,self.TTF_truetype_fast  =self.fastt()



    def fastt(self):
        # path_ttfs = self.ttf_path + self.ttf_name[ttstk]
        liyy =[]
        TTF_truetype_fast =[]
        print('初始化')
        for i in tqdm(self.ttf_name):
            with open(self.ttf_path +i,'rb') as f:
                liyy.append(f.read())
                text_size = 255  # text_size 是字号
                font = ImageFont.truetype(self.ttf_path +i, text_size)
                TTF_truetype_fast.append(font)

        return liyy ,TTF_truetype_fast

    def get_img_bh(self):
        while True:
            numm = np.random.randint(0, self.worddlist_len)
            ttstk =np.random.randint(0, self.ttf_name_len)
            # path_ttfs =self.ttf_path +self.ttf_name[ttstk]
            path_ttfs = BytesIO(self.ttf_bet[ttstk])
            tocken = self.worddlist[numm]
            # font = TTFont(path_ttfs)
            font = TTFont(path_ttfs)
            # 输出的uniMap是一个字典，key代表的unicode的int值，value代表unicode的名字
            uniMap = font['cmap'].tables[0].ttFont.getBestCmap()
            # print(ord(tocken) in uniMap.keys())
            tfss =ord(tocken) in uniMap.keys()
            if tfss:
                imggg =self.get_img(path_ttfs,tocken)
                return tocken,imggg
    def get_img(self,ttf__path,word):


        # 生成font
        ttf_path = ttf__path
        text_size = 255  # text_size 是字号
        font = ImageFont.truetype(ttf_path, text_size)

        x, y = font.getsize(word)

        y = max(y, 256)
        x = max(x, 256)
        cavv = Image.new('RGB', (x, y), (255, 255, 255))

        ddd = ImageDraw.Draw(cavv)
        ddd.text((0, 0), word, font=font, fill='#000000')

        return cavv.resize((256, 256), Image.ANTIALIAS)
        # image.save('aaa.png')

    def get_tk_list(self,tocken):
        listtt :list= self.wordd[tocken]

        bhl:list =listtt
        mask_bh =self.bh_mask(listtt.copy())

        l = len(listtt)
        mask = []
        aa = self.config.n_vebs
        if l < aa:
            bhl.extend(0 for _ in range(aa - l))
            mask_bh.extend(0 for _ in range(aa - l))
        if l == aa:
            mask.extend(1 for _ in range(aa))
        else:
            cccsd = []
            cccsd.extend(1 for _ in range(l))
            cccsd2 = []
            cccsd2.extend(0 for _ in range(aa - l))
            mask = cccsd + cccsd2

        return bhl, mask,mask_bh

    def bh_mask(self,bh):
        mask_num=0
        out_bh =bh
        for inx, val in enumerate(bh):
            cdf =random.random()

            if cdf <= 0.15:
                rdf = random.random()
                if rdf>=0.2:
                    out_bh[inx] =43
                    mask_num =mask_num + 1
                elif rdf>=0.1:
                    out_bh[inx] = np.random.randint(0, self.config.n_vebs_long)
                    mask_num = mask_num + 1

        if mask_num ==0:
            lenss =len(out_bh)
            out_bh[np.random.randint(0, lenss)] =43

        return out_bh








    def __getitem__(self ,index):
        tocken_word,imgg =self.get_img_bh()

        bh_list,mask,mask_bh =self.get_tk_list(tocken_word)

        tkm =torch.zeros((self.config.n_vebs,self.config.n_vebs_long))

        for inx, val in enumerate(bh_list):
            tkm[inx][val]=1.0
        transform1 = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        ]
        )

        img_tensor =transform1(imgg)
        mask=torch.tensor(mask,dtype=torch.long)
        bh_list=torch.tensor(bh_list,dtype=torch.long)
        mask_bh=torch.tensor(mask_bh,dtype=torch.long)
        return img_tensor,mask_bh,tkm,mask,'None'


    def __len__(self):
        return self.lenrr


dataa =dataloadd(len11=5000,v_words_path='fix1.json',mapping_path='映射.json',path_ttf='./datas/ttfs/',config=configs())
print('生成数据ing')
fast_ran_list = [dataa[0] for _ in tqdm(range(400000))]


class dataloadd_io_fast(Dataset):
    def __init__(self, tump):
        self.lenrr =len(tump)
        self.tunnn =tump

    def __getitem__(self, index):
        return self.tunnn[index]

    def __len__(self):
        return self.lenrr
dataa_fast =dataloadd_io_fast(tump=fast_ran_list)

class main_part_one(pyl.LightningModule):
    def __init__(self,config):
        super().__init__()
        self.encode =part_of_main_modle(config)
        self.outline = nn.Linear(config.n_embd,config.n_vebs_long)
        nn.Sequential(nn.Linear(config.n_embd,250),nn.SELU(),nn.Linear(250,config.n_vebs_long))
        self.F_de =res_modle(n_layer=4,chanal=512)
        self.LR =config.LR
        self.L1loss1 =nn.L1Loss()
        self.Crosslsoss =nn.CrossEntropyLoss()
        self.decode =nn.Sequential(nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(16, 16), stride=2,padding=1), nn.SELU(),
                                   nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(8, 8), stride=2,
                                                      padding=0), nn.SELU(),
                                   nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 2), stride=2,
                                                      padding=0), nn.SELU(),
                                   nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=(10, 10), stride=2,
                                                      padding=0), nn.SELU(),
                                   )

    def forward(self,x,y,att_mask=None,att_mask_img=None):
        img,bh =self.encode(x,y,att_mask,att_mask_img)
        img1 =img.transpose(1, 2).view(img.size(0),512,8,8)
        img =self.decode(self.F_de(img.transpose(1, 2).view(img.size(0),512,8,8)))
        bh = self.outline(bh)
        sfmax =nn.Softmax()
        bh =sfmax(bh)

        return img,bh

    def training_step(self, batch, batch_idx):
        img,bh,r_bh,att_mask,att_mask_img =batch
        if att_mask_img=='None':
            att_mask_img=None
        x =img
        img,bh =self.forward(img,bh,att_mask,att_mask_img)
        imggm =img.detach().cpu().numpy()*255
        if batch_idx%100==0:
            tpp =transforms.ToPILImage()
            image = tpp(img.detach().cpu().clone()[0])
            image.save('./a/'+str(batch_idx)+'.png')
        img_loss =self.L1loss1(img,x)

        bh_loss =self.Crosslsoss(bh,r_bh)

        main_loss =(img_loss + bh_loss)/2
        loss =main_loss

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.LR)

eeeee =main_part_one(configs())
# asasd =eeeee.training_step((image,linss1,1,None,None),1)
trainer = Trainer(gpus=1)
trainer.fit(eeeee,train_dataloaders=DataLoader(dataset=dataa_fast,batch_size=8))






