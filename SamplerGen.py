import torch
import structure
import os

if not os.path.exists("./sampler"):
    os.makedirs("./sampler")
    
for i in range(10):
    S = structure.Sampler()
    File ="./sampler/Sampler_%d.pt" % i
    torch.save(S.points,File)
