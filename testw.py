import math
import torch
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