from torch import nn
import torch.nn.functional as F
import torch
import math

class Sampler():

    def __init__(self, samplenumber=100, scope=200):

        self.points = torch.randn(samplenumber)*scope

    def sample(self,a):
        
        x = torch.sin(a[0]*self.points+a[1])
        return x

    def PairGen(self,idx=0):
        p=torch.randn(2,2)
        p[0,idx]=p[1,idx]
        x0 = self.sample(p[0,:])
        x1 = self.sample(p[1,:])
        return torch.stack((x0,x1)), p

    def DataGen(self,datasize=2000,idx=0):
        data = torch.empty(0,2,self.points.size()[0])
        para = torch.empty(0,2,2)
        for i in range(datasize):
            d,p = self.PairGen()
            data = torch.cat((data,d.unsqueeze(0)))
            para = torch.cat((para,p.unsqueeze(0)))
        return data,para


def Perturb(x,r):
    y = x+torch.randn_like(x)*r
    return(y)

def PairPerturb(X,r=.5,k=0):
    Y = X
    Y[k] = Perturb(Y[k],r)
    return Y

def DataPerturb(data,para, res = 20):
    data_new = torch.empty(0,2,data.size()[2])
    para_new = torch.empty(0,2,2)
    dist_new = torch.empty(0,1)
    for n in range(data.size()[0]):
        for j in range(res):
            for i in range(2):
                j0 = torch.empty(1).uniform_(0,1)
                r = 2**(-j0*res)
                data_new = torch.cat((data_new,PairPerturb(data[n,:,:],r,i).unsqueeze(0)))
                para_new = torch.cat((para_new,para[n,:,:].unsqueeze(0)))
                dist_new = torch.cat((dist_new,r.unsqueeze(0)))
    print(data_new.size(),para_new.size(),dist_new.size())
    return data_new, para_new, dist_new
    


def normalize(v):
    with torch.no_grad():
        R = v.norm()
        v1 = v/R
        v.copy_(v1)
                   
def flow(v,lr = 0):
    with torch.no_grad():
        v1 = v-lr*(v.grad+0.0*torch.randn_like(v))
        v = v1
            
class Net(nn.Module):

    def __init__(self, xdim=100, hdim=200):

        super(Net,self).__init__()

        self.fc = nn.Linear(xdim,hdim).cuda()
        self.fc.bias.data.fill_(0.)
        for p in self.parameters():
            p.requires_grad=False

            
    def forward(self,x0):

        x = x0/x0.norm()
        
        return torch.sin(self.fc(x)).norm()

def run(vec, net,lr, train=True):
    
    G=torch.zeros(1)
    
    if train==True:
        L = net(vec)
        L.backward(retain_graph=True)
        #print(vec.grad[1])
        G = vec.grad.norm()
        #print(G)
        with torch.no_grad():
            vec.copy_(vec-lr*(vec.grad+0.01*torch.randn_like(vec)))
            vec.copy_(vec/vec.norm())
        vec.grad.zero_()
        
    if train==False:
        with torch.no_grad():
            L = net(vec)

    return L.detach(), G.detach()
