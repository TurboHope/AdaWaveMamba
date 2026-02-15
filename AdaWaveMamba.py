import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.MambaLifting import MambaLiftingBlock

class RevIN(nn.Module):
    """Reversible Instance Normalization"""
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps*1e-5)
            x = x * self.stdev + self.mean
            return x

class AdaptiveScaleFusion(nn.Module):
    """
    [创新点 C] 自适应多尺度融合 (Adaptive Scale Fusion)
    不只是简单相加，而是学习每个尺度的重要性权重
    """
    def __init__(self, num_levels, d_model):
        super(AdaptiveScaleFusion, self).__init__()
        self.num_scales = num_levels + 1 # Details * num_levels + 1 Approx
        # 一个简单的门控网络，计算每个尺度的权重
        self.gate = nn.Sequential(
            nn.Linear(d_model * self.num_scales, d_model),
            nn.Tanh(),
            nn.Linear(d_model, self.num_scales),
            nn.Softmax(dim=-1)
        )

    def forward(self, preds):
        # preds list: [Tensor(B, L, D), Tensor(B, L, D)...]
        # Stack 起来: [B, L, D, Scales]
        stacked = torch.stack(preds, dim=-1) 
        B, L, D, S = stacked.shape
        
        # 计算权重: [B, L, D, S] -> Flatten -> [B, L, D*S] -> Gate -> [B, L, S]
        flat = stacked.view(B, L, -1)
        weights = self.gate(flat) # [B, L, S]
        weights = weights.unsqueeze(2) # [B, L, 1, S]
        
        # 加权求和
        # [B, L, D, S] * [B, L, 1, S] -> Sum over S -> [B, L, D]
        fused = torch.sum(stacked * weights, dim=-1)
        return fused

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.num_levels = configs.num_levels

        self.revin = RevIN(self.enc_in)
        self.embedding = nn.Linear(1, self.d_model)

        # 1. Bi-Mamba Lifting Blocks
        self.lifting_blocks = nn.ModuleList()
        for i in range(self.num_levels):
            self.lifting_blocks.append(MambaLiftingBlock(self.d_model, dropout=0.1))

        # 2. Projections
        self.projection_layers = nn.ModuleList()
        current_len = self.seq_len
        for i in range(self.num_levels):
            current_len = current_len // 2
            self.projection_layers.append(nn.Linear(current_len, self.pred_len))
        self.projection_layers.append(nn.Linear(current_len, self.pred_len))

        # 3. [创新点 C] 自适应融合层
        self.fusion_layer = AdaptiveScaleFusion(self.num_levels, self.d_model)

        self.out_proj = nn.Linear(self.d_model, 1)

    def forward(self, x):
        # 1. Norm & CI
        x = self.revin(x, 'norm')
        B, L, C = x.shape
        x = x.permute(0, 2, 1).reshape(B * C, L, 1)
        
        # 2. Embed
        x_enc = self.embedding(x)

        # 3. Bi-Directional Decomposition
        details = []
        curr_s = x_enc
        for block in self.lifting_blocks:
            curr_s, curr_d = block(curr_s)
            details.append(curr_d)
        final_s = curr_s
        
        # 4. Projection to same length
        all_components = []
        
        # 处理 Details
        for i, d in enumerate(details):
            pred_d = self.projection_layers[i](d.permute(0, 2, 1)).permute(0, 2, 1)
            all_components.append(pred_d)
            
        # 处理 Approx
        pred_s = self.projection_layers[-1](final_s.permute(0, 2, 1)).permute(0, 2, 1)
        all_components.append(pred_s)

        # 5. [创新点 C] Adaptive Fusion
        total_pred = self.fusion_layer(all_components)

        # 6. Output & Denorm
        output = self.out_proj(total_pred) 
        output = output.reshape(B, C, self.pred_len).permute(0, 2, 1)
        output = self.revin(output, 'denorm')

        return output