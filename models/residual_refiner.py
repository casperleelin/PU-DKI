import torch
import torch.nn as nn
from models.pointops.functions import pointops

class LocalResidualRefineUnit(nn.Module):
    def __init__(self,num_neighbors,in_channels):
        super(LocalResidualRefineUnit, self).__init__()

        self.coord_grouper = pointops.QueryAndGroup(nsample=num_neighbors + 1,return_idx=True,use_xyz=True)
        self.feature_grouper = pointops.QueryAndGroupFeature(nsample=num_neighbors + 1,return_idx=False,use_feature=False)

        self.feature_mlps = nn.Sequential(nn.Conv2d(in_channels + 3,2*in_channels,1,bias=False),
                                          nn.BatchNorm2d(2*in_channels),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(2*in_channels,in_channels,1),
                                          nn.BatchNorm2d(in_channels),
                                          nn.ReLU(inplace=True)
                                          )
        self.coord_mlps = nn.Sequential(nn.Conv2d(3,in_channels,1,bias=False),
                                          nn.BatchNorm2d(in_channels),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(in_channels,num_neighbors,1)
                                          )

        self.mlps_maxpool = nn.Sequential(nn.Conv2d(in_channels,2*in_channels,1,bias=False),
                                          nn.BatchNorm2d(2*in_channels),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(2*in_channels,in_channels,1,bias=False),
                                          nn.BatchNorm2d(in_channels),
                                          nn.ReLU(inplace=True),
                                          nn.MaxPool2d([1,num_neighbors])
                                          )

    def forward(self,x,xyz):
        '''
        :param x: features [B,C,N]
        :param xyz: coord [B,N,3]
        :return: [B,C,N]
        '''

        grouped_diff, _, indices = self.coord_grouper(xyz=xyz.contiguous())

        grouped_diff = grouped_diff[...,1:]

        grouped_feature = self.feature_grouper(xyz=xyz.contiguous(), features=x.contiguous(), idx=indices.int())
        grouped_feature = grouped_feature[...,1:]
        #print(grouped_diff.shape,grouped_xyz.shape,grouped_feature.shape)
        #torch.Size([2, 3, 1024, 15]) torch.Size([2, 3, 1024, 15]) torch.Size([2, 32, 1024, 15])

        weights = self.coord_mlps(grouped_diff)
        #print(weights.shape) torch.Size([2, 15, 1024, 15])

        cat_feature = torch.cat([grouped_feature,grouped_diff],dim=1)
        #print(cat_feature.shape) torch.Size([2, 35, 1024, 15])

        features = self.feature_mlps(cat_feature)
        #print(features.shape) torch.Size([2, 32, 1024, 15])

        rNxC_feature = self.mlps_maxpool(features).squeeze()
        #print(rNxC_feature.shape) torch.Size([2, 32, 1024])

        # torch.Size([2,1024,15,15]) torch.Size([2,1024,15,32])
        rNxC_weight = torch.matmul(weights.permute(0,2,3,1),features.permute(0,2,3,1))
        rNxC_weight = rNxC_weight.sum(dim=-2).squeeze()
        # print(rNxC_weight.shape) torch.Size([2, 1024, 32])

        return rNxC_weight.permute(0,2,1) + rNxC_feature

class SelfAttentionUnit(nn.Module):
    def __init__(self,in_channels):
        super(SelfAttentionUnit, self).__init__()

        self.to_q = nn.Sequential(nn.Conv1d(3 + in_channels,2 * in_channels,1,bias=False),
                                  nn.BatchNorm1d(2 * in_channels),
                                  nn.ReLU(inplace=True))
        self.to_k = nn.Sequential(nn.Conv1d(3 + in_channels,2 * in_channels, 1, bias=False),
                                  nn.BatchNorm1d(2 * in_channels),
                                  nn.ReLU(inplace=True))
        self.to_v = nn.Sequential(nn.Conv1d(3 + in_channels,2 * in_channels, 1, bias=False),
                                  nn.BatchNorm1d(2 * in_channels),
                                  nn.ReLU(inplace=True))

        self.fusion = nn.Sequential(nn.Conv1d(2 * in_channels, in_channels, 1, bias=False),
                                  nn.BatchNorm1d(in_channels),
                                  nn.ReLU(inplace=True))

    def forward(self,x):
        '''
        :param x: [B,C + 3,N]
        :return: [B,C,N]
        '''

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        #print(q.shape,k.shape,v.shape)
        attention_map = torch.matmul(q.permute(0,2,1),k)
        #print(attention_map.shape) torch.Size([2, 1024, 1024])
        value = torch.matmul(attention_map,v.permute(0,2,1)).permute(0,2,1)

        value = self.fusion(value)
        #print(value.shape) torch.Size([2, 64, 1024])

        return value


class GlobalResidualRefineUnit(nn.Module):
    def __init__(self,in_channels):
        super(GlobalResidualRefineUnit, self).__init__()
        self.sa = SelfAttentionUnit(in_channels=in_channels)

    def forward(self,x,xyz):
        '''
        :param x: [B,C,N]
        :param xyz: [B,N,3]
        :return: [B,C,N]
        '''
        cat = torch.cat([xyz.permute(0,2,1),x],dim=1)

        return self.sa(cat)

class OffsetRegression(nn.Module):
    def __init__(self,in_channels):
        super(OffsetRegression, self).__init__()
        self.coordinate_regression = nn.Sequential(nn.Conv1d(in_channels, 256, 1),
                                                   nn.ReLU(inplace=True),
                                                   nn.Conv1d(256, 64, 1),
                                                   nn.ReLU(inplace=True),
                                                   nn.Conv1d(64, 3, 1),
                                                   nn.Sigmoid())
        self.range_max = 0.5

    def forward(self,x):
        '''
        :param x: [B,C,N]
        :return: [B,N,3]
        '''

        offset = self.coordinate_regression(x)

        return (offset * self.range_max * 2 - self.range_max).permute(0,2,1)

class ResidualRefiner(nn.Module):
    def __init__(self,num_neighbors,in_channels):
        super(ResidualRefiner, self).__init__()

        self.local_refinement =  LocalResidualRefineUnit(num_neighbors=num_neighbors,in_channels=in_channels)
        self.global_refinement = GlobalResidualRefineUnit(in_channels=in_channels)
        self.offset = OffsetRegression(in_channels=in_channels)

    def forward(self,x,xyz):
        '''
        :param x: [B,C,4N]
        :param xyz: [B,4N,3]
        :return: [B,4N,3]
        '''
        coarse_coords = xyz.permute(0,2,1)
        local_refine = self.local_refinement(x, coarse_coords)
        global_refine = self.global_refinement(x, coarse_coords)

        refine = local_refine + global_refine

        return self.offset(refine).permute(0,2,1)