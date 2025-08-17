import torch
import torch.nn as nn
import numpy as np

# Multi-head Attention
class MultipleHeadAttention(nn.Module):
    def __init__(self, args):
        super(MultipleHeadAttention, self).__init__()
        self.trans_dim = args.trans_dim
        self.num_heads = args.head_num
        self.shell_sizes = args.shell_sizes
        self.num_kernel_points = np.sum(self.shell_sizes).item()
        # self.upsampling_rate = args.upsampling_rate
        self.conv_queries = nn.Conv2d(self.trans_dim, self.trans_dim, 1)
        self.conv_keys = nn.Conv2d(self.trans_dim, self.trans_dim, 1)
        self.conv_values = nn.Conv2d(self.trans_dim, self.trans_dim, 1)
        self.ffn = nn.Sequential(
            nn.Conv2d(self.trans_dim, self.trans_dim*2, 1),
            nn.BatchNorm2d(self.trans_dim*2) if args.is_attn_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.trans_dim*2, self.trans_dim, 1)
        )
        # self.attn_conv = nn.Conv2d(self.dim, self.dim, 1)
        self.ffn_act = nn.ReLU(inplace=True)

    def forward(self, queries, feats):
        B, _, N, __ = feats.shape
        queries_count = queries.shape[3]
        queries_iden = queries
        queries = self.conv_queries(queries).view(B, self.num_heads, self.trans_dim//self.num_heads, N, -1).permute(0, 3, 1, 4, 2).contiguous() # (B, n, h, r, d//h)
        keys = self.conv_keys(feats).view(B, self.num_heads, self.trans_dim//self.num_heads, N, -1).permute(0, 3, 1, 2, 4).contiguous() # (B, n, h, d//h, nkp)
        values = self.conv_values(feats).view(B, self.num_heads, self.trans_dim//self.num_heads, N, -1).permute(0, 3, 1, 4, 2).contiguous() # (B, n, h, nkp, d//h) 

        logits = torch.matmul(queries, keys) * ((self.trans_dim // self.num_heads)**(-0.5)) # (B, n, h, r, nkp) 
        soft_logits = torch.softmax(logits, dim=-1)
        agg_feats = torch.matmul(soft_logits, values).permute(0, 1, 2, 4, 3).contiguous().view(B, N, -1, queries_count) # (B, n, d, r) 
        agg_feats = agg_feats.permute(0,2,1,3).contiguous()# (B, n, d, r)
        # agg_feats = self.attn_conv(agg_feats) + queries_iden# (B, d, N, r) 
        agg_feats = agg_feats + queries_iden
        agg_feats = self.ffn_act(self.ffn(agg_feats) + agg_feats)
        return agg_feats

# Cross-Attention Transformer
class CrossAttentionTransformer(nn.Module):
    def __init__(self, args):
        super(CrossAttentionTransformer, self).__init__()    
        self.attentions = nn.ModuleList()
        for i in range(args.trans_num):
            self.attentions.append(MultipleHeadAttention(args))

    def forward(self, q, k):
        for m in self.attentions:
            q = m(q, k)
        return q