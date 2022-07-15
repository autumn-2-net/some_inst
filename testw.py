import math
import torch
import numpy as np
q = torch.randn((1,23,512,64))
k= torch.randn((1,23,256,64))
v= torch.randn((1,23,256,64))
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
print(att)
y = att @ v
print(y)
x =torch.randn((10,64))
seq_length = x.size(1)
bach_size = x.size(0)
position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
position_ids = position_ids.unsqueeze(0).expand_as(x)
position_ids
token_type_ids = torch.ones_like(x)
x

asssw = torch.arange(64, dtype=torch.long,).unsqueeze(0)
y_1 = torch.chunk(asssw, 8, dim=1)
z = torch.cat(y_1, dim=0)
asssw = z.unsqueeze(0).expand_as(torch.randn((1,512,8,8), device=x.device))
asssw1 =asssw.view(1,512,64).transpose(1, 2)
bbb =asssw1.numpy()
asssw2 =asssw1.transpose(1, 2).view(1,512,8,8)
bbb1 =asssw2.numpy()
tkm =torch.zeros((32,32))

tkm[0][0]=1.0

tkm[1][12]=1.0
tkm[2][14]=1.0
tkm[3][18]=1.0

x