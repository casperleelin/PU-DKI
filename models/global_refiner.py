import torch
import torch.nn as nn
from torch.nn import functional as F

class AttentionUnit(nn.Module):
    def __init__(self, args):
        super(AttentionUnit, self).__init__()
        self.kernel_output_dim = args.kernel_output_dim
        self.queries = nn.Sequential(nn.Conv1d(3 + self.kernel_output_dim, 2 * self.kernel_output_dim, 1, bias=False),
                                  nn.BatchNorm1d(2 * self.kernel_output_dim),
                                  nn.ReLU(inplace=True))
        
        self.keys = nn.Sequential(nn.Conv1d(3 + self.kernel_output_dim, 2 * self.kernel_output_dim, 1, bias=False),
                                  nn.BatchNorm1d(2 * self.kernel_output_dim),
                                  nn.ReLU(inplace=True))
        
        self.values = nn.Sequential(nn.Conv1d(3 + self.kernel_output_dim, 2 * self.kernel_output_dim, 1, bias=False),
                                  nn.BatchNorm1d(2 * self.kernel_output_dim),
                                  nn.ReLU(inplace=True))
        
        self.fusion = nn.Sequential(nn.Conv1d(2 * self.kernel_output_dim, self.kernel_output_dim, 1, bias=False),
                                  nn.BatchNorm1d(self.kernel_output_dim),
                                  nn.ReLU(inplace=True))
        
    def forward(self, features):
        queries = self.queries(features)
        keys = self.keys(features)
        values = self.values(features)
        
        attention_scores = torch.matmul(queries.transpose(1, 2), keys)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        attention_output = torch.matmul(attention_weights, values.transpose(1, 2)).permute(0,2,1)
        
        fused_values = self.fusion(attention_output)
        
        return fused_values

class GlobalRefiner(nn.Module):
    def __init__(self, args):
        super(GlobalRefiner, self).__init__()
        
        self.global_attention = AttentionUnit(args)
        
    def forward(self, features, coords):
        concatenated_features = torch.cat([features, coords], dim=1)
        refined_features = self.global_attention(concatenated_features)
        
        return refined_features