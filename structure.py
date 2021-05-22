import torch

class Sampler():

    def __init__(self, samplenumber=100, width=200):

        self.points = torch.empty(samplenumber).uniform_(-width,width).sort().values.cuda()

    def sample(self,a):
        
        x = torch.sin(a[0]*self.points+a[1])
        return x

    def PairGen(self,idx=0):
        p=torch.randn(2,2).cuda()
        p[0,idx]=p[1,idx]
        x0 = self.sample(p[0,:])
        x1 = self.sample(p[1,:])
        return torch.stack((x0,x1)), p

    def DataGen(self,datasize=2000,idx=0):
        data = torch.empty(0,2,self.points.size()[0]).cuda()
        para = torch.empty(0,2,2).cuda()
        for n in range(datasize):
            d,p = self.PairGen(idx)
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

def DataPerturb(data,para):
    data_new = torch.empty(0,2,data.size()[2]).cuda()
    para_new = torch.empty(0,2,2).cuda()
    dist_new = torch.empty(0,1).cuda()
    for n in range(data.size()[0]):
        for j in range(10):
            for i in range(2):
                j0 = torch.empty(1).uniform_(0,1).cuda()
                r = j0**(2)
                data_new = torch.cat((data_new,PairPerturb(data[n,:,:],r,i).unsqueeze(0)))
                para_new = torch.cat((para_new,para[n,:,:].unsqueeze(0)))
                dist_new = torch.cat((dist_new,r.unsqueeze(0)))
    return data_new, para_new, dist_new
