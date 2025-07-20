import torch
from torch import nn

# MLPAer : for univariate

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.channel = configs.enc_in
        self.t_ff = configs.t_ff
        self.c_ff = configs.c_ff
        self.norm = configs.usenorm
        self.embed_dropout = configs.embed_dropout
        self.head_dropout = configs.head_dropout
        self.t_dropout = configs.t_dropout
        self.c_dropout = configs.c_dropout
        self.features = configs.features

        
        self.backbone = MLPAerBackBone(
            self.seq_len, self.d_model, self.channel, self.t_ff,
            self.c_ff, self.t_dropout, self.c_dropout, self.embed_dropout
        )
        
        self.head = nn.Sequential(
            nn.Dropout(self.head_dropout),
            nn.Linear(2 * self.d_model, self.pred_len)
        )
        

    def forward(self, x_enc):
        # [batch_size, seq_len, channel]
        if self.norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        
        x_enc = x_enc.permute(0, 2, 1)                                  # [b, c, s]
        
        en = self.backbone(x_enc)                                   # [b, 1, 2d]
        
        dec_out = self.head(en).permute(0, 2, 1)                    # [b, t, 1]
        
        if self.norm:
            dec_out = dec_out * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out

class MLPAerBackBone(nn.Module):
    def __init__(self, seq_len, d_model, channel, t_ff, c_ff, t_dropout, c_dropout, embed_dropout):
        super().__init__()
        self.d_model = d_model
        self.channel = channel
        self.projection = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.Dropout(embed_dropout)
        )
        
        self.global_token = nn.Parameter(
            torch.ones([1, 1, d_model], dtype=torch.float)
        )
        
        self.temproal = Attention(2 * d_model, t_ff, t_dropout)
        
        self.cross = Attention(channel, c_ff, c_dropout)
    
    def forward(self, x):
        b, _, _ = x.shape
        embed = self.projection(x)                              # [b, c, d]
        en = embed[:, -1:, :]                                   # [b, 1, d]
        ex = embed[:, :-1, :]                                   # [b, c-1, d]
        
        glob = self.global_token.repeat([b, 1, 1])              # [b, 1, d]
        en_d = torch.concat([en, glob], dim=-1)                 # [b, 1, 2d]
        
        en_atten = self.temproal(en_d)                          # [b, 1, 2d]
        
        origin_atten = en_atten[:, :, :self.d_model]            # [b, 1, d]
        glob = en_atten[:, :, self.d_model:]                    # [b, 1, d]
        
        ex_d = torch.concat([ex, glob], dim=1)                  # [b, c, d]
        ex_atten = self.cross(ex_d.permute(0, 2, 1))            # [b, d, c]
        
        glob = ex_atten[:, :, -1:]                              # [b, d, 1]
        en_data = torch.concat(
            [origin_atten, glob.permute(0, 2, 1)], dim=-1)      # [b, 1, 2d]
        
        return en_data
    
                                 
class Attention(nn.Module):
    def __init__(self, d_model, hf, dropout=0.):
        super().__init__()
        self.d_model = d_model
        
        self.weight = nn.Sequential(
            nn.Linear(d_model, hf),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hf, d_model),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        weight = self.weight(x)
        return x * weight
