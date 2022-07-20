from tqdm import tqdm

from base_modle.bert_modle import main_config
from run_part_modle import dataloadd
from torch.utils.data import Dataset, DataLoader


class configs(main_config):
    n_head =16
    LR = 0.0003
    img_bert_lay = 6
    bh_bert_lay = 4
    co_att_lay = 5
dataa = dataloadd(len11=10000, v_words_path='fix1.json', mapping_path='映射.json', path_ttf='./datas/ttfs/',
                      config=configs())


# for i in dataa:
#     print(i)
#     img, bh, r_bh, att_mask, att_mask_img, mask_List = i

for ai in tqdm(DataLoader(dataset=dataa,batch_size=1 , )):
    # print(ai)
    img, bh, r_bh, att_mask, att_mask_img, mask_List = ai
    for inx, var in enumerate(bh[0]):
        if var == 43:
            # assert 'errrrr'
            # print('rrrrrrrrrrr')
            # raise '1111'
            if inx >20:
                raise '1111'
        if inx > 20:
            if var!=0:
                raise 'emmmmmm'