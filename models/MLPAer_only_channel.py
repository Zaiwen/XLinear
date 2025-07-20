import torch
from torch import nn

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
        
        self.backbone = MLPAerBackbone(
            self.seq_len, self.d_model, self.channel, self.t_ff,
            self.c_ff, self.t_dropout, self.c_dropout, self.embed_dropout
        )
        
        self.head = nn.Sequential(
            nn.Dropout(self.head_dropout),
            nn.Linear(self.d_model, self.pred_len)
        )
        

    def forward(self, x_enc):
        # [batch_size, seq_len, channel]
        if self.norm:
            # Normalization
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        
        x_enc = x_enc.permute(0, 2, 1)                                  # [b, c, s]
        
        en = self.backbone(x_enc)                                   # [b, c, d]
        
        dec_out = self.head(en).permute(0, 2, 1)                    # [b, t, c]
        
        if self.norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out


class MLPAerBackbone(nn.Module):
    def __init__(self, seq_len, d_model, channel, t_ff, c_ff, t_dropout, c_dropout, embed_dropout):
        super().__init__()
        self.d_model = d_model
        self.channel = channel
        self.projection = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.Dropout(embed_dropout)
        )
        
        self.glob_token = nn.Parameter(
            torch.ones([1, channel, d_model])
        )
        
        self.en_attention = Attention(2 * d_model, t_ff, t_dropout)
        
        self.ex_attention = Attention(2 * channel, c_ff, c_dropout)
    
    def forward(self, x):
        # in: [batch_size, channel, seq_len]
        # out:[batch_size, channel, 2 * d_model]
        b, _, _ = x.shape
        emb = self.projection(x)                                # [b, c, d]
        
        glob_token = self.glob_token.repeat([b, 1, 1])          # [b, c, d]
        en_emb = torch.concat([emb, glob_token], dim=-1)        # [b, c, 2d]
        
        en_atten = self.en_attention(en_emb)                    # [b, c, 2d]
        
        origin_atten = en_atten[:, :, :self.d_model]            # [b, c, d]
        glob_atten = en_atten[:, :, self.d_model:]              # [b, c, d]
        
        ex_emb = torch.concat([emb, glob_atten], dim=1)         # [b, 2c, d]
        ex_atten = self.ex_attention(ex_emb.permute(0, 2, 1))   # [b, d, 2c]
        
        glob = ex_atten[:, :, self.channel:]                    # [b, d, c]
        
        # en = torch.concat([origin_atten, glob.permute(0, 2, 1)], dim=-1) # [b, c, 2d]
        en = glob.permute(0, 2, 1)                              # [b, c, d]
        
        return en
        

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
