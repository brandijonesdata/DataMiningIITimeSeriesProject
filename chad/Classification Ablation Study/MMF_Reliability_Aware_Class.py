import torch
import torch.nn as nn

class MMF_Reliability_Aware_Class(nn.Module):
    def __init__(self, d_txt: int, C: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.C = C # This will be 32 (hid_dim)
        self.d_txt = d_txt
        
        # Original GR-Add components [cite: 46-50]
        self.gru = nn.GRU(input_size=C + d_txt, hidden_size=hidden_dim, batch_first=True)
        self.residual_head = nn.Linear(hidden_dim, C)
        self.gate_net = nn.Linear(C + d_txt, C)
        
        # Reliability scoring networks [cite: 69]
        self.w_ts = nn.Linear(2, 1)  # For missingness and alignment
        self.w_txt = nn.Linear(2, 1) # For density and recency
        self.alpha = nn.Parameter(torch.tensor(1.0)) # Learned reliability influence [cite: 73]
        
        self.layer_norm = nn.LayerNorm(C)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_ts, h_txt, q_ts, q_txt, m_txt_pooled):
        B, T, C = h_ts.shape
        x = torch.cat([h_ts, h_txt], dim=-1)

        # 1. Compute Residual [cite: 49-50]
        u, _ = self.gru(x)
        r_i = self.residual_head(u)
        r_i = self.dropout(self.layer_norm(r_i))

        # 2. Compute Scalar Reliability Scores r_i (0 to 1) [cite: 69-70]
        r_ts = torch.sigmoid(self.w_ts(q_ts))
        r_txt = torch.sigmoid(self.w_txt(q_txt))

        # 3. Reliability-Aware Gate [cite: 72]
        # Formula: sigmoid(W[h_ts||h_txt] + b + alpha * (r_ts - r_txt))
        gate_logits = self.gate_net(x)
        rel_contrast = self.alpha * (r_ts - r_txt)
        g_tilde = torch.sigmoid(gate_logits + rel_contrast)

        # 4. Mask and Fuse [cite: 76-77]
        mask = m_txt_pooled.unsqueeze(-1).expand(-1, -1, C).bool()
        g_tilde = torch.where(mask, g_tilde, torch.ones_like(g_tilde))
        
        return h_ts + g_tilde * r_i