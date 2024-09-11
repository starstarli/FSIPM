import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        x = self.net(x)
        x = self.norm(x + residual)
        return x

class MyAttention(nn.Module):
    def __init__(self, dim, heads, dim_head):
        super(MyAttention, self).__init__()
        self.heads = heads
        self.dim = dim
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.to_Q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_K = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_V = nn.Linear(dim, dim_head * heads, bias=False)
        self.norm = nn.LayerNorm(dim_head)

    def forward(self, x, y):
        # b, n, d, c
        residual = x.permute(0,3,1,2)
        b,n_x, d_x, h_x = x.shape
        b,n_y, d_y, h_y = y.shape
        x = x.reshape(b, n_x, -1)
        y = y.reshape(b, n_y, -1)
        q = self.to_Q(x).view(b, -1, self.heads, d_x).transpose(1, 2)
        k = self.to_K(y).view(b, -1, self.heads, d_y).transpose(1, 2)
        v = self.to_V(y).view(b, -1, self.heads, d_y).transpose(1, 2)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = self.norm(out + residual)

        return out


class FCN_SAtt(nn.Module):
    def __init__(self,rois,channel):
        super(FCN_SAtt,self).__init__()

        self.Channel_AvgPool = nn.AvgPool2d(kernel_size=(1,channel))
        self.MLP = nn.Sequential(
            nn.Linear(rois , rois // 4),
            nn.ReLU(),
            nn.Linear(rois // 4,rois)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.Channel_AvgPool(x)
        x = x.permute(0,2,3,1)
        x = self.MLP(x)
        x = x.permute(0,3,1,2)
        attention = self.sigmoid(x)
        return attention


class FCN_conv(nn.Module):
    def __init__(self,rois,C,C1,C2):
        super(FCN_conv,self).__init__()
        self.EdgeConv = nn.Conv2d(C,C1,kernel_size=(1,rois))
        # self.EdgeConv = DynamicConv2d(C,C1,kernel_size=(1,rois))
        self.relu_1 = nn.LeakyReLU()
        self.FCN_SAtt = FCN_SAtt(rois,channel=C1)
        self.NodeConv = nn.Conv2d(C1,C2,kernel_size=(rois,1))
        # self.NodeConv = DynamicConv2d(C,C1,kernel_size=(rois,1))
        self.relu_2 = nn.LeakyReLU()
    def Pearson(self,x):
        x = x.permute(0,3,1,2)
        mean = torch.sum(x,dim=-1) / x.shape[-1]
        mean = mean.unsqueeze(-1)
        x = x - mean
        vect = torch.einsum("b c r t , b c i t -> b c r i", x, x)
        norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
        fab = torch.einsum("b c r t , b c i t -> b c r i", norm, norm)
        pcc = vect / fab
        return pcc
    def forward(self,x):
        x = self.Pearson(x)
        x = self.EdgeConv(x)
        x = x.permute(0,2,3,1)
        x = self.relu_1(x)
        att = self.FCN_SAtt(x)
        out = torch.einsum("b r q k , b r q c -> b r k c", att, x)
        out = out.permute(0,3,1,2)
        out = self.NodeConv(out)
        out = self.relu_2(out)
        out = out.permute(0,2,3,1)
        return out,att

class SSN_SAtt(nn.Module):
    def __init__(self,rois,channel):
        super(SSN_SAtt,self).__init__()
        self.c = channel
        self.Channel_AvgPool = nn.AvgPool2d(kernel_size=(1,channel))
        self.MLP = nn.Sequential(
            nn.Linear(rois , rois // 4),
            nn.ReLU(),
            nn.Linear(rois // 4,rois)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.Channel_AvgPool(x)
        x = x.permute(0,2,3,1)
        x = self.MLP(x)
        x = x.permute(0,3,1,2)
        attention = self.sigmoid(x)
        return attention

class SSN_conv(nn.Module):
    def __init__(self,rois,C,C1,C2):
        super(SSN_conv,self).__init__()
        self.EdgeConv = nn.Conv2d(C,C1,kernel_size=(1,rois))
        self.relu_1 = nn.LeakyReLU()
        self.SSN_SAtt = SSN_SAtt(rois,channel=C1)
        self.NodeConv = nn.Conv2d(C1,C2,kernel_size=(rois,1))
        self.relu_2 = nn.LeakyReLU()
    def forward(self,x):
        x = x.permute(0,3,1,2)
        x = self.EdgeConv(x)
        x = self.relu_1(x)
        x = x.permute(0,2,3,1)
        att = self.SSN_SAtt(x)
        out = torch.einsum("b r q k , b r q c -> b r k c", att, x)
        out = out.permute(0,3,1,2)
        out = self.NodeConv(out)
        out = self.relu_2(out)
        out = out.permute(0,2,3,1)
        return out,att


class FS_AttPool(nn.Module):
    def __init__(self,ps= 4 ):
        super(FS_AttPool,self).__init__()
        self.w1 = 0.5
        self.w2 = 0.5
        self.ps = ps
    def getM(self,att):
        attention = torch.flatten(att,start_dim=1)
        values,_ = torch.sort(attention,dim=1,descending = True)
        Threshold = values[:,self.ps,None]
        mask = torch.greater_equal(attention, Threshold)       
        m = torch.ones_like(attention)  * mask
        M = torch.sum(m,dim=0)
        return M
    def forward(self,x,attention,att_off=None):
        M = self.getM(attention)
        if att_off is not None:
            M_off = self.getM(att_off)
            M = self.w1 * M + self.w2 * M_off
        _,indices = torch.sort(M ,descending = True)
        top_m = indices[:M.shape[0] // self.ps]
        return x[:,top_m],top_m

class ST_trans(nn.Module):
    def __init__(self,rois,C,C1,C2,dim,mlp_dim,dropout):
        super(ST_trans,self).__init__()
        self.rois = rois
        self.selfAttention = MyAttention(dim,heads=C,dim_head=dim // C)
        self.FeedForward = FeedForward(dim // C,mlp_dim,dropout) 
        self.EdgeConv = nn.Conv2d(C,C1,kernel_size=(1,rois))
        self.relu_1 = nn.LeakyReLU()
        self.NodeConv = nn.Conv2d(C1,C2,kernel_size=(rois,1))
        self.relu_2 = nn.LeakyReLU()
    def forward(self,x):
        x = self.selfAttention(x,x)
        x = self.FeedForward(x)
        x_out = torch.matmul(x,x.transpose(-1,-2) / torch.sqrt(torch.tensor(self.rois)))
        out = self.EdgeConv(x_out)
        out = self.relu_1(out)
        out = self.NodeConv(out)
        out = self.relu_2(out)
        return out.permute(0,2,3,1)


class ST_aggregation(nn.Module):
    def __init__(self,rois,in_channel,out_channel,dim,pt=3):
        super(ST_aggregation,self).__init__()
        self.TimeAvgPool = nn.AvgPool2d(kernel_size=(1,dim // pt))
        self.TimeSDPool = nn.AvgPool2d(kernel_size=(1,dim // pt))
        self.NodeConv = nn.Conv2d(2*in_channel,out_channel,kernel_size=(rois,1))
        self.relu = nn.LeakyReLU()
    def forward(self,x):
        x=x.permute(0,3,1,2)
        x1 = self.TimeAvgPool(x)
        x2 = self.TimeSDPool(x)
        out = torch.cat((x1,x2),dim=1)
        out = self.NodeConv(out)
        out = self.relu(out)
        out = out.permute(0,2,3,1)
        return out

class ST_conv(nn.Module):
    def __init__(self,w,C,C1,C2,pt=3):
        super(ST_conv,self).__init__()
        self.TimeConv = nn.Conv2d(C,C1,kernel_size=(1,w))
        self.relu_1 = nn.LeakyReLU()
        self.TimeAvgPool = nn.AvgPool2d(kernel_size=(1,pt))
        self.SpatialConv = nn.Conv2d(C1,C2,kernel_size=(1,1))
    def forward(self,x):
        x = x.permute(0,3,1,2)
        x = self.TimeConv(x)
        x = self.relu_1(x)
        x = self.TimeAvgPool(x)
        x = self.SpatialConv(x)
        x = x.permute(0,2,3,1)
        return x 


class FSIPM(nn.Module):
    def __init__(self,rois,timepoints,p,num_classes,dropout,w):
        super(FSIPM, self).__init__()
        self.rois = rois
        self.timepoints = timepoints
        self.p = p
        self.ps = 4
        self.pt = 3
        self.dim1 = timepoints - w + 1
        self.dim2 = (self.dim1 - w + 1) // self.ps 
        self.FC_conv_1 = FCN_conv(rois,C=1,C1=8,C2=16)
        self.FC_conv_2 = FCN_conv(rois,C=16,C1=32,C2=64)
        self.FC_conv_3 = FCN_conv(rois // self.ps,C=64,C1=128,C2=256)

        self.trans_dim1 = timepoints
        self.trans_dim2 = (self.trans_dim1 - w + 1) // self.pt
        self.trans_dim3 = (self.trans_dim2 - w +1 ) // self.pt

        self.ST_trans_1 = ST_trans(rois,C=1,C1=8,C2=16,dim=self.trans_dim1 * 1,mlp_dim=self.trans_dim1*3,dropout=dropout)
        self.ST_trans_2 = ST_trans(rois ,C=16,C1=32,C2=64,dim=self.trans_dim2 * 16,mlp_dim=self.trans_dim2*3,dropout=dropout)
        self.ST_trans_3 = ST_trans(rois // self.ps,C=64,C1=128,C2=256,dim=self.trans_dim3 * 64,mlp_dim=self.trans_dim3*3,dropout=dropout)
        
        self.ST_graph_conv_1 = ST_conv(w,C=1,C1=8,C2=16)
        self.ST_graph_conv_2 = ST_conv(w,C=16,C1=32,C2=64)

        self.Spatial_AttPool_1 = FS_AttPool()
        self.Spatial_AttPool_2 = FS_AttPool()
        
        self.ST_aggregation_1 = ST_aggregation(rois // self.ps,in_channel=16,out_channel=64,dim = self.dim1)
        self.ST_aggregation_2 = ST_aggregation(rois // self.ps**2,in_channel=64,out_channel=128,dim = self.dim2)


        self.SC_conv_1 = SSN_conv(rois,C=1,C1=16,C2=64)
        self.SC_conv_2 = SSN_conv(rois // self.ps,C=1,C1=64,C2=256)

        self.MLP = nn.Sequential(
            nn.Linear(4, 16),
            nn.LeakyReLU(0.33),
            nn.Dropout(0.2),
            nn.Linear(16, 32),
            nn.LeakyReLU(0.33)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(1216, 128),
            nn.LeakyReLU(0.33),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.33)
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(128),
            nn.LeakyReLU(0.33),
            nn.Linear(128, num_classes)
        )
    

    def forward(self, x):
        # inputs n*T*1

        inputs = x[:, :self.rois * self.timepoints].reshape(-1,self.rois,self.timepoints)
        inputs_s = x[:, self.rois * self.timepoints: -self.p].reshape(-1,self.rois,self.rois)
        pheno = x[:, -self.p:]

        inputs = inputs.unsqueeze(-1)
        inputs_s = inputs_s.unsqueeze(-1)

        batch_1,att_1 = self.FC_conv_1(inputs)
        batch_trans_1 = self.ST_trans_1(inputs)

        st_gconv_1 = self.ST_graph_conv_1(inputs * att_1)

        batch_2,att_2 = self.FC_conv_2(st_gconv_1)
        batch_trans_2 = self.ST_trans_2(st_gconv_1)

        batch_7,att_off_1 = self.SC_conv_1(inputs_s)
        
        sp_attp_1,index_1 = self.Spatial_AttPool_1(st_gconv_1*att_2,att_2,att_off_1)
        batch_3 = self.ST_aggregation_1(sp_attp_1)

        st_gconv_2 = self.ST_graph_conv_2(sp_attp_1)

        batch_4,att_3 = self.FC_conv_3(st_gconv_2)
        batch_trans_3 = self.ST_trans_3(st_gconv_2)

        batch_8,att_off_2 = self.SC_conv_2((inputs_s*att_off_1)[:,index_1][:,:,index_1])

        sp_attp_2,index_2 = self.Spatial_AttPool_2(st_gconv_2*att_3,att_3,att_off_2)
        batch_5 = self.ST_aggregation_2(sp_attp_2)
        
        batch_6 = self.MLP(pheno)
        
        x_out = torch.cat((batch_1,batch_trans_1,batch_2,batch_trans_2,batch_3,batch_4,batch_trans_3,batch_5),dim=-1)
        x_out = torch.flatten(x_out,start_dim=1)

        y_out = torch.cat((batch_7,batch_8),dim=-1)
        y_out = torch.flatten(y_out,start_dim=1)
        out = torch.cat((x_out,y_out,batch_6),dim=1)
        out = self.fc1(out)

        out_norm = F.normalize(out, p=2, dim=1)
        return out_norm, out
    
    def frozen_forward(self, x):
        with torch.no_grad():
            _, x = self.forward(x)
        x = self.mlp_head(x)
        return torch.softmax(x, dim=-1)


