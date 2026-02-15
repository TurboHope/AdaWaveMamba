import torch
import torch.nn as nn
from mamba_ssm import Mamba

class BiMambaOperator(nn.Module):
    """
    [创新点 A] 双向 Mamba 算子 (Bi-Directional Mamba)
    解决标准 Mamba 只能单向因果建模的问题，实现全局上下文感知。
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super(BiMambaOperator, self).__init__()
        
        # 正向 Mamba (Forward)
        self.mamba_fwd = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        # 反向 Mamba (Backward)
        self.mamba_bwd = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 特征融合层：把正向和反向的结果融合
        self.fusion = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        # x: [Batch, Length, Channel]
        x_in = x
        x = self.norm(x)
        
        # 1. 正向流
        out_fwd = self.mamba_fwd(x)
        
        # 2. 反向流 (翻转输入 -> Mamba -> 再翻转回来)
        x_bwd = x.flip(dims=[1]) # [B, L, C] -> Flip Time
        out_bwd = self.mamba_bwd(x_bwd).flip(dims=[1])
        
        # 3. 双向融合
        # 简单相加或者Concat后映射，这里选用Concat更强一点
        out_cat = torch.cat([out_fwd, out_bwd], dim=-1)
        out = self.fusion(out_cat)
        
        return self.dropout(out) + x_in

class MambaLiftingBlock(nn.Module):
    """
    使用 Bi-Mamba 的提升块
    """
    def __init__(self, d_model, d_state=16, dropout=0.1):
        super(MambaLiftingBlock, self).__init__()
        
        # 使用双向算子
        self.predictor = BiMambaOperator(d_model, d_state=d_state, dropout=dropout)
        self.updater = BiMambaOperator(d_model, d_state=d_state, dropout=dropout)

    def forward(self, x):
        if x.shape[1] % 2 != 0:
            x = x[:, :-1, :]
            
        x_even = x[:, 0::2, :] 
        x_odd = x[:, 1::2, :]  

        # Predict
        pred = self.predictor(x_even)
        d = x_odd - pred

        # Update
        update = self.updater(d)
        s = x_even + update

        return s, d