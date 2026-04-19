import torch
import torch.nn as nn

class RA_Stock_Classifier(nn.Module):
    def __init__(self, args, mmf_module):
        super().__init__()
        from models.tPatchGNN import tPatchGNN
        self.encoder = tPatchGNN(args)
        self.mmf = mmf_module
        
        # Binary Classification Head [cite: 102]
        self.classifier = nn.Sequential(
            nn.Linear(args.hid_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, X, mask, text_emb, text_mask, q_ts, q_txt):
        # 1. Structured Representation h_ts [cite: 32]
        h_ts = self.encoder.IMTS_Model(X, mask)
        
        # 2. Aligned Text Representation h_txt [cite: 41]
        t_mask_expanded = text_mask.unsqueeze(-1)
        pooled_text = (text_emb * t_mask_expanded).sum(dim=1) / (t_mask_expanded.sum(dim=1) + 1e-9)
        h_txt = pooled_text.unsqueeze(1).repeat(1, h_ts.size(1), 1)
        
        # 3. Reliability-Aware Fusion [cite: 76-78]
        m_txt_pooled = torch.ones((h_ts.size(0), h_ts.size(1)), device=X.device)
        h_fuse = self.mmf(h_ts, h_txt, q_ts, q_txt, m_txt_pooled)
        
        # 4. Classification Logit [cite: 104]
        pooled_context = torch.mean(h_fuse, dim=1)
        return self.classifier(pooled_context)