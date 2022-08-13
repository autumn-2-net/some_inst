import torch
from base_modle.modle_pack import part_of_main_modle
from base_modle.bert_modle import main_config
class configs(main_config):
    n_head =16
    LR = 0.00005
    img_bert_lay = 7
    bh_bert_lay = 5

    co_att_lay = 8

ccd = part_of_main_modle(configs)
a =torch.load(r'C:\Users\autumn\Desktop\poject_all\Font_DL\lightning_logs\version_43\checkpoints\epoch=41-step=84000.ckpt')
lisss =[]
aaa ='111'

for i in a['state_dict']:
    if 'encode.' in i:

        lisss.append(i)
    print(i)
part_sd = {k[7:]: v for k, v in a['state_dict'].items() if k  in lisss}
for i in part_sd:
    print(i)
# ddddd ={}
# for k, v in a['state_dict'].items():
#     if k in lisss:
#         ddddd[k[7:]] =v


ccd.load_state_dict(part_sd)

torch.save(ccd.state_dict(), 'part_one.pkl')
print(ccd)