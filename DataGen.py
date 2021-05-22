import torch
import structure
import os

sampler_idx = 0

S = structure.Sampler()
S.points = torch.load("./sampler/Sampler_%d.pt" % sampler_idx)

if not os.path.exists("./data"):
    os.makedirs("./data")

for i in range(2):
    data,para = S.DataGen(3000,i)
    torch.save(data, "./data/data_%d.pt" % i)
    torch.save(para, "./data/para_%d.pt" % i)

    data_new,para_new,dist_new = structure.DataPerturb(data,para)
    torch.save(data_new, "./data/pert_data_%d.pt" % i)
    torch.save(para_new, "./data/pert_para_%d.pt" % i)
    torch.save(dist_new, "./data/pert_dist_%d.pt" % i)

    data,para = S.DataGen(500,i)
    torch.save(data, "./data/test_data_%d.pt" % i)
    torch.save(para, "./data/test_para_%d.pt" % i)

    data_new,para_new,dist_new = structure.DataPerturb(data,para)
    torch.save(data_new, "./data/test_pert_data_%d.pt" % i)
    torch.save(para_new, "./data/test_pert_para_%d.pt" % i)
    torch.save(dist_new, "./data/test_pert_dist_%d.pt" % i)
