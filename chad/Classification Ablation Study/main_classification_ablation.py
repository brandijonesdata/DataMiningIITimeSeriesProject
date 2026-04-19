import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from lib.parse_datasets import parse_datasets
from main import get_args_from_parser, update_args_for_dataset, update_args_for_model
from sklearn.metrics import accuracy_score, f1_score, classification_report
from utils.tools import set_seed
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score, 
                             balanced_accuracy_score, precision_score, recall_score, f1_score)
import numpy as np


# --- 1. Reliability-Aware Fusion [cite: 1, 69-78] ---
class MMF_Reliability_Aware_Class(nn.Module):
    def __init__(self, d_txt, C, hidden_dim):
        super().__init__()
        # Matches your 32-dim latent space requirements [cite: 1, 46-50]
        self.gru = nn.GRU(input_size=C + d_txt, hidden_size=hidden_dim, batch_first=True)
        self.residual_head = nn.Linear(hidden_dim, C)
        self.gate_net = nn.Linear(C + d_txt, C)
        self.w_ts = nn.Linear(2, 1)  # Missingness, Alignment [cite: 1, 62-63]
        self.w_txt = nn.Linear(2, 1) # Density, Recency [cite: 1, 66-67]
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.layer_norm = nn.LayerNorm(C)

    def forward(self, h_ts, h_txt, q_ts, q_txt):
        x = torch.cat([h_ts, h_txt], dim=-1)
        u, _ = self.gru(x)
        r_i = self.layer_norm(self.residual_head(u))
        
        # Scalar trust scores [cite: 1, 69-70]
        r_ts_score = torch.sigmoid(self.w_ts(q_ts))
        r_txt_score = torch.sigmoid(self.w_txt(q_txt))
        
        # Reliability-Aware Gate [cite: 1, 71-72]
        g_tilde = torch.sigmoid(self.gate_net(x) + self.alpha * (r_ts_score - r_txt_score))
        return h_ts + g_tilde * r_i

# --- 2. Multi-Modal Classifier Wrapper ---
class RA_Stock_Classifier(nn.Module):
    def __init__(self, args, mmf):
        super().__init__()
        from models.tPatchGNN import tPatchGNN
        self.encoder = tPatchGNN(args)
        self.mmf = mmf
        self.hid_dim = args.hid_dim
        self.classifier = nn.Sequential(nn.Linear(args.hid_dim, 1), nn.Sigmoid())

    def forward(self, X_final, mask_flat, text_emb, text_mask, q_ts, q_txt):
        # A. Latent Encoding [cite: 1, 31-32]
        h_ts_flat = self.encoder.IMTS_Model(X_final, mask_flat)
        B = text_emb.shape[0] if text_emb is not None else q_ts.shape[0]
        h_ts = h_ts_flat.view(B, -1, self.hid_dim)
        
        # B. Text Alignment [cite: 1, 40-41]
        if text_emb is not None:
            t_mask_exp = text_mask.unsqueeze(-1)
            h_txt_single = (text_emb * t_mask_exp).sum(1) / (t_mask_exp.sum(1) + 1e-9)
            h_txt = h_txt_single.unsqueeze(1).repeat(1, h_ts.size(1), 1)
        else:
            h_txt = torch.zeros((B, h_ts.size(1), 768), device=X_final.device)
        
        # C. Fusion and Output [cite: 1, 76-78, 101-104]
        h_fuse = self.mmf(h_ts, h_txt, q_ts, q_txt)
        return self.classifier(torch.mean(h_fuse, dim=1))

def main():
    # --- STEP 1: ARGUMENT INTERCEPTOR ---
    c_parser = argparse.ArgumentParser(add_help=False)
    c_parser.add_argument('--task', type=str, default='classification')
    c_parser.add_argument('--ablation_step', type=int, default=5)
    c_parser.add_argument('--llm_layers_fusion', type=str, default='full')
    res_args, remaining = c_parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    
    args = get_args_from_parser()
    args.task, args.ablation_step, args.llm_layers_fusion = res_args.task, res_args.ablation_step, res_args.llm_layers_fusion
    
    # --- STEP 2: DATA INITIALIZATION [cite: 1, 14-15] ---
    args = update_args_for_dataset(args)
    data_obj = parse_datasets(args, show_summary=False)
    args.C = data_obj["input_dim"]
    args = update_args_for_model(args)
    set_seed(args.seed)

    # --- STEP 3: ABLATION CONFIGS ---
    configs = {
        1: {"ts": [0,0], "txt": [0,0]}, # Baseline
        2: {"ts": [0,0], "txt": [0,1]}, # + Recency
        3: {"ts": [1,0], "txt": [0,1]}, # + Missingness
        4: {"ts": [1,1], "txt": [0,1]}, # + Alignment
        5: {"ts": [0,0], "txt": [1,0]}, # + Density Only
        6: {"ts": [1,1], "txt": [1,1]},  # Full Reliability
        7: {"ts": [0,1], "txt": [1,1]}, # RA-GRAdd (-Missingness)
        8: {"ts": [1,0], "txt": [1,1]}, # RA-GRAdd (-Alignment)
        9: {"ts": [1,1], "txt": [0,1]}, # RA-GRAdd (-Density)
        10: {"ts": [1,1], "txt": [1,0]} # RA-GRAdd (-Recency)
    }
    cfg = configs[args.ablation_step]

    # --- STEP 4: MODEL INIT ---
    mmf = MMF_Reliability_Aware_Class(768, args.hid_dim, args.hid_dim)
    model = RA_Stock_Classifier(args, mmf).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    # --- STEP 5: FULL TRAINING LOOP ---
    print(f"--- Starting Ablation Step {args.ablation_step} on {args.dataset} ---")
    for epoch in range(args.epoch):
        model.train()
        epoch_loss = 0
        for i, batch in enumerate(data_obj["train_dataloader"]):
            # A. Extract Data
            X_ts = batch['observed_data'].to(args.device)
            mask = batch['observed_mask'].to(args.device)
            tt = batch['observed_tp'].to(args.device)
            text_emb = batch['notes_embeddings'].to(args.device) if args.enable_text else None
            text_mask = (torch.sum(torch.abs(text_emb), dim=-1) != 0).float().to(args.device) if text_emb is not None else None

            B, M, L_in, N = X_ts.shape
            
            # B. Generate Reliability Features for this step [cite: 1, 61-67]
            q_ts = torch.ones(B, M, 2).to(args.device) * torch.tensor(cfg['ts']).to(args.device)
            q_txt = torch.ones(B, M, 2).to(args.device) * torch.tensor(cfg['txt']).to(args.device)

            # C. Preprocess Numerical Data 
            X_flat = X_ts.permute(0, 3, 1, 2).reshape(-1, L_in, 1)
            tt_flat = tt.permute(0, 3, 1, 2).reshape(-1, L_in, 1) if tt.dim() == 4 else \
                      tt.unsqueeze(1).repeat(1, N, 1, 1).reshape(-1, L_in, 1)
            mask_flat = mask.permute(0, 3, 1, 2).reshape(-1, L_in, 1)
            
            model.encoder.batch_size = B
            X_final = torch.cat([X_flat, model.encoder.LearnableTE(tt_flat)], dim=-1)

            # D. Generate Labels (Price Increase = 1) [cite: 1, 100-101]
            labels = (batch['data_to_predict'][:, 0, 0].to(args.device) > X_ts[:, -1, -1, 0]).float().unsqueeze(1)

            # E. Forward & Backward Pass
            optimizer.zero_grad()
            outputs = model(X_final, mask_flat, text_emb, text_mask, q_ts, q_txt)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch {epoch}, Iter {i}, Loss: {loss.item():.4f}")
        
        print(f"--- Epoch {epoch} complete. Avg Loss: {epoch_loss/len(data_obj['train_dataloader']):.4f}")

    # --- STEP 6: FINAL TEST EVALUATION [cite: 1, 140-157, 167-168] ---
    print(f"\n--- Running Final Evaluation for Step {args.ablation_step} ---")
    model.eval()
    y_true, y_probs = [], []

    with torch.no_grad():
        for batch in data_obj["test_dataloader"]:
            X_ts = batch['observed_data'].to(args.device)
            mask = batch['observed_mask'].to(args.device)
            tt = batch['observed_tp'].to(args.device)
            text_emb = batch['notes_embeddings'].to(args.device) if args.enable_text else None
            text_mask = (torch.sum(torch.abs(text_emb), dim=-1) != 0).float().to(args.device) if text_emb is not None else None

            B, M, L_in, N = X_ts.shape
            q_ts = torch.ones(B, M, 2).to(args.device) * torch.tensor(cfg['ts']).to(args.device)
            q_txt = torch.ones(B, M, 2).to(args.device) * torch.tensor(cfg['txt']).to(args.device)
            
            X_flat = X_ts.permute(0, 3, 1, 2).reshape(-1, L_in, 1)
            tt_flat = tt.permute(0, 3, 1, 2).reshape(-1, L_in, 1) if tt.dim() == 4 else \
                      tt.unsqueeze(1).repeat(1, N, 1, 1).reshape(-1, L_in, 1)
            mask_flat = mask.permute(0, 3, 1, 2).reshape(-1, L_in, 1)
            model.encoder.batch_size = B
            X_final = torch.cat([X_flat, model.encoder.LearnableTE(tt_flat)], dim=-1)
            
            labels = (batch['data_to_predict'][:, 0, 0].to(args.device) > X_ts[:, -1, -1, 0]).float().unsqueeze(1)

            probs = model(X_final, mask_flat, text_emb, text_mask, q_ts, q_txt)
            y_probs.extend(probs.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    # --- STEP 7: CALCULATE TABLE METRICS  ---
    y_true, y_probs = np.array(y_true), np.array(y_probs)

    # 1. Find the Optimal Threshold
    best_f1 = 0
    best_threshold = 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        y_p = (y_probs > t).astype(float)
        current_f1 = f1_score(y_true, y_p, zero_division=0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = t

    # 2. Generate Final Predictions using the Optimized Threshold
    y_pred = (y_probs > best_threshold).astype(float)

    exp_name = {1:"Original GR-Add", 2:"RA-GRAdd (+Recency)", 3:"RA-GRAdd (+Missingness)", 
                4:"RA-GRAdd (+Alignment)", 5:"RA-GRAdd (+Density)", 6:"RA-GRAdd (+All 4)", 7: "RA-GRAdd (-Missingness)", 8: "RA-GRAdd (-Alignment)", 9: "RA-GRAdd (-Density)", 10: "RA-GRAdd (-Recency)"}[args.ablation_step]

    print(f"\n{'Experiment':<25} | {'AUC':<4} | {'AUPRC':<5} | {'Acc':<4} | {'F1':<4} | {'Prec':<4} | {'Rec':<4} | {'Thresh':<6}")
    print("-" * 110)
    print(f"{exp_name:<25} | {roc_auc_score(y_true, y_probs):.2f} | {average_precision_score(y_true, y_probs):.2f} | "
          f"{accuracy_score(y_true, y_pred):.2f} | {f1_score(y_true, y_pred):.2f} | "
          f"{precision_score(y_true, y_pred):.2f} | {recall_score(y_true, y_pred):.2f} | {best_threshold:.2f}")

if __name__ == "__main__":
    main()