import torch
import torch.nn as nn

class MultimodalStockClassifier(nn.Module):
    def __init__(self, args, mmf_module, supports=None):
        super(MultimodalStockClassifier, self).__init__()
        # 1. Instantiate the original tPatchGNN as the encoder
        from models.tPatchGNN import tPatchGNN
        self.encoder = tPatchGNN(args, supports=supports)
        
        # 2. The Multimodality Fusion (MMF) module (GR-Add)
        self.mmf = mmf_module
        
        # 3. Binary Classification Head
        self.classifier_head = nn.Sequential(
            nn.Linear(args.hid_dim, args.hid_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.hid_dim // 2, 1),
            nn.Sigmoid() 
        )

    def forward(self, X, mask, text_embeddings, text_mask):
        """
        X: Numerical (B*N*M, L, 11)
        mask: Mask (B*N*M, L, 1)
        text_embeddings: GPT-2 (B, 25, d_txt)
        text_mask: Binary mask for the 25 headlines (B, 25)
        """
        # 1. Encode Numerical Data (B, 6, hid_dim)
        numerical_repr = self.encoder.IMTS_Model(X, mask)
        
        # 2. Process Multimodal Fusion
        if self.mmf is not None and text_embeddings is not None:
            # A. Masked Temporal Pooling: Average the 25 headlines into 1
            t_mask_expanded = text_mask.unsqueeze(-1) # (B, 25, 1)
            pooled_text = (text_embeddings * t_mask_expanded).sum(dim=1) / (t_mask_expanded.sum(dim=1) + 1e-9)
            
            # B. Repeat to match the 6 numerical nodes -> (B, 6, d_txt)
            text_repr = pooled_text.unsqueeze(1).repeat(1, numerical_repr.size(1), 1)
            
            # C. Create a matching mask for the pooled features (B, 6)
            # This ensures GR-Add doesn't crash due to the length 25 vs 6 mismatch.
            m_txt_pooled = torch.ones((numerical_repr.size(0), numerical_repr.size(1)), device=X.device)
            
            fused_repr = self.mmf(numerical_repr, text_repr, m_txt_pooled)
        else:
            fused_repr = numerical_repr
            
        # 3. Global Aggregation & Prediction
        pooled_context = torch.mean(fused_repr, dim=1)
        probability = self.classifier_head(pooled_context)
        
        return probability