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
from base_modle.loss_ff import FeatureLoss
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
from fontTools.ttLib import TTFont
from PIL import ImageFont, Image, ImageDraw
from torchvision import transforms
from visdom import Visdom

viz = Visdom(env='FONT_DL')
viz.line([0.],    ## Y的第一个点坐标
        [0.],    ## X的第一个点坐标
        win="train loss all",    ##窗口名称
        opts=dict(title='train_loss')  ## 图像标例
        )
viz.line([0.],    ## Y的第一个点坐标
        [0.],    ## X的第一个点坐标
        win="train loss img",    ##窗口名称
        opts=dict(title='img_loss')  ## 图像标例
        )
viz.line([0.],    ## Y的第一个点坐标
        [0.],    ## X的第一个点坐标
        win="train loss bh",    ##窗口名称
        opts=dict(title='bh_loss')  ## 图像标例
        )



class configs(main_config):
    n_head =16
    LR = 0.0003
    img_bert_lay = 9
    bh_bert_lay = 5
    co_att_lay = 8
    # n_img_embd = 1024  # 图片特征维度
    # n_bh_embd = 1024  # 笔画特征维度
    #
    # n_embd = 1024

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
                imggg =self.get_img(ttstk,tocken)
                return tocken,imggg
    def get_img(self,ttstk,word):


        # 生成font
        # ttf_path = ttf__path
        text_size = 255  # text_size 是字号
        # font = ImageFont.truetype(ttf_path, text_size)
        font =self.TTF_truetype_fast[ttstk]

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
        mask_bh,bh_mask_l =self.bh_mask(listtt.copy())

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

        return bhl, mask,mask_bh,bh_mask_l

    def bh_mask(self,bh):
        mask_num=0
        out_bh =bh
        mask_list = []
        for inx, val in enumerate(bh):
            cdf =random.random()

            if cdf <= 0.15:
                rdf = random.random()
                if rdf>=0.2:
                    out_bh[inx] =43
                    mask_list.append(inx)
                    mask_num =mask_num + 1
                elif rdf>=0.1:
                    out_bh[inx] = np.random.randint(0, self.config.n_vebs_long)
                    mask_list.append(inx)
                    mask_num = mask_num + 1

        if mask_num ==0:
            lenss =len(out_bh)
            inx =np.random.randint(0, lenss)
            out_bh[inx] =43
            mask_list.append(inx)

        return out_bh,mask_list








    def __getitem__(self ,index):
        tocken_word,imgg =self.get_img_bh()

        bh_list,mask,mask_bh,mask_list =self.get_tk_list(tocken_word)

        tkm =torch.zeros((self.config.n_vebs,self.config.n_vebs_long))
        mskk_inx = torch.zeros(self.config.n_vebs)
        for i in mask_list:
            mskk_inx[i]=1

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
        mskk_inx =torch.tensor(mskk_inx,dtype=torch.long)
        return img_tensor,mask_bh,tkm,mask,'None',mskk_inx


    def __len__(self):
        return self.lenrr



# print('生成数据ing')
# fast_ran_list = [dataa[0] for _ in tqdm(range(400000))]


class dataloadd_io_fast(Dataset):
    def __init__(self, tump):
        self.lenrr =len(tump)
        self.tunnn =tump

    def __getitem__(self, index):
        return self.tunnn[index]

    def __len__(self):
        return self.lenrr
# dataa_fast =dataloadd_io_fast(tump=fast_ran_list)

class main_part_one(pyl.LightningModule):
    def __init__(self,config):
        super().__init__()
        self.encode =part_of_main_modle(config)
        self.outline = nn.Linear(config.n_embd,config.n_vebs_long)
        nn.Sequential(nn.Linear(config.n_embd,2048),nn.SELU(),nn.Linear(2048,config.n_vebs_long,bias=False))
        self.F_de =res_modle(n_layer=4,chanal=512)
        self.LR =config.LR
        self.L1loss1 =nn.L1Loss()
        self.Crosslsoss =nn.CrossEntropyLoss()
        self.MSElsoss = nn.MSELoss()
        # self.perceptual_loss =FeatureLoss(loss=self.MSElsoss ,blocks=[1,2,3],weights=[0.2,0.3,0.5],device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.decode =nn.Sequential(nn.ConvTranspose2d(in_channels=config.n_img_embd , out_channels=256, kernel_size=(15, 15), stride=2,padding=0), nn.SELU(),
                                   nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(8, 8), stride=2,
                                                      padding=1), nn.SELU(),
                                   nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(8, 8), stride=2,
                                                      padding=2), nn.SELU(),
                                   nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=(8, 8), stride=2,
                                                      padding=1), nn.SELU(),
                                   )
        self.enco =0
        self.eph = 0
        self.config=config

    def forward(self,x,y,att_mask=None,att_mask_img=None):
        img,bh =self.encode(x,y,att_mask,att_mask_img)
        # img1 =img.transpose(1, 2).view(img.size(0),512,8,8)
        img =self.decode(self.F_de(img.transpose(1, 2).view(img.size(0),self.config.n_img_embd,8,8)))
        # img = self.decode(img.transpose(1, 2).view(img.size(0), 512, 8, 8))
        bh = self.outline(bh)
        sfmax =nn.Softmax()
        bh =sfmax(bh)

        return img,bh

    def training_step(self, batch, batch_idx):
        img,bh,r_bh,att_mask,att_mask_img ,mask_List=batch
        if att_mask_img=='None':
            att_mask_img=None
        seq_length = bh.size(1)
        bach_size = bh.size(0)
        maskksw =bh.detach().cpu().clone().numpy()[0]

        loss_mask = att_mask.unsqueeze(2).expand((bach_size,seq_length,48),).to(bh.device)
        mask_List = mask_List.unsqueeze(2).expand((bach_size, seq_length, 48), ).to(bh.device)
        # loss_mask= mask_List+loss_mask
        # loss_mask = loss_mask.ge(1.5).to(bh.device)
        # loss_mask =  loss_mask
        # loss_mask = loss_mask.ge(0.5).to(bh.device)

        x =img
        sssss = loss_mask.detach().cpu().clone().numpy()
        img,bh =self.forward(img,bh,att_mask,att_mask_img)
        imggm =img.detach().cpu().numpy()*255
        if batch_idx%10==0:
            tpp =transforms.ToPILImage()
            image = tpp(img.detach().cpu().clone()[0])
            imgy=img.detach().cpu().clone()[0]*255
            imgy=imgy.round()
            viz.image(
                imgy,
                opts=dict(title='OUT!', caption='How random.'),win="out_img"
            )
            viz.image(
                x.detach().cpu().clone()[0],
                opts=dict(title='IN!', caption='How random.'),win="in_img"
            )
            if batch_idx % 100 == 0:
                image.save('./a/'+str(self.eph)+'__'+str(batch_idx)+'.png')
        img_loss1 =self.L1loss1(img,x)
        img_loss2 =self.MSElsoss(img,x)
        # img_loss =(img_loss1 + img_loss2)/2
        img_loss = img_loss1
        bh_maskk =loss_mask*bh
        rh_mask =r_bh*loss_mask
        # bh_maskk = torch.masked_select(bh , loss_mask)
        # rh_mask =  torch.masked_select(r_bh , loss_mask)
        bh_loss =self.Crosslsoss(bh_maskk,rh_mask)

        main_loss =(img_loss + bh_loss)/2
        # main_loss = img_loss
        # mask = (bh == bh.max(dim=1, keepdim=True)[0])
        # mskkk =bh.argmax(dim=2)

        if batch_idx%10==0:
            if batch_idx==0:
                self.eph =self.eph+1
            else:
                self.enco =self.enco+10
                lioi =[]
                for inx,var in enumerate(maskksw):
                    if var ==43:
                        lioi.append(inx)
                viz.text('mask'+str(lioi), win='text')
                # viz.close(win='hotmapss')
                # viz.heatmap(
                #     X=bh_maskk.detach().cpu().clone()[0],
                #     opts=dict(
                #         columnnames=['y'+str(i) for i in range(48)],
                #         rownames=[str(i) for i in range(32)],
                #         colormap='Viridis',
                #     ),win="hotmapss"
                # )
                # viz.heatmap(
                #     X=rh_mask.detach().cpu().clone()[0],
                #     opts=dict(
                #         columnnames=['y'+str(i) for i in range(48)],
                #         rownames=[str(i) for i in range(32)],
                #         colormap='Viridis',
                #     ),win="hotmapss——ral"
                # )
                mskkk = bh.argmax(dim=2)[0]
                mskkk_rl = r_bh.argmax(dim=2)[0]

                viz.heatmap(
                    X=bh.detach().cpu().clone()[0],
                    opts=dict(
                        columnnames=['y'+str(i) for i in range(48)],
                        rownames=[str(i) for i in range(32)],
                        colormap='Viridis',
                    ),win="hotmapss"
                )
                viz.heatmap(
                    X=r_bh.detach().cpu().clone()[0],
                    opts=dict(
                        columnnames=['y'+str(i) for i in range(48)],
                        rownames=[str(i) for i in range(32)],
                        colormap='Viridis',
                    ),win="hotmapss——ral"
                )
                viz.line([main_loss.detach().cpu()],  ## Y的下一个点坐标
                         [float(self.enco)],  ## X的下一个点坐标
                         win="train loss all",  ## 窗口名称 与上个窗口同名表示显示在同一个表格里
                         update='append'  ## 添加到上一个点后面
                         )
                viz.line([img_loss.detach().cpu()],  ## Y的第一个点坐标
                         [float(self.enco)],  ## X的第一个点坐标
                         win="train loss img",  ##窗口名称
                         update='append'   ## 图像标例
                         )
                viz.line([bh_loss.detach().cpu()],  ## Y的第一个点坐标
                         [float(self.enco)],  ## X的第一个点坐标
                         win="train loss bh",  ##窗口名称
                         update='append'   ## 图像标例
                         )
        loss =main_loss

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.LR)

if __name__ == '__main__':
    eeeee =main_part_one(configs())
    dataa = dataloadd(len11=10000, v_words_path='fix1.json', mapping_path='映射.json', path_ttf='./datas/ttfs/',
                      config=configs())
    # asasd =eeeee.training_step((image,linss1,1,None,None),1)
    # eeeee.load_state_dict(torch.load(r'C:\Users\autumn\Desktop\poject_all\Font_DL\lightning_logs\version_3\checkpoints\epoch=5-step=3750.ckpt'))
    trainer = Trainer(gpus=1)
    trainer.fit(eeeee,train_dataloaders=DataLoader(dataset=dataa,batch_size=5 ,#num_workers=1
                                                   ))






