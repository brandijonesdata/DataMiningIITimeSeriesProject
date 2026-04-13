import torch
import torch.nn as nn
import torch.optim as optim
from models.multimodal_classifier import MultimodalStockClassifier
from utils.tools import set_seed
from lib.parse_datasets import parse_datasets

# 1. Import your NEW classification-specific fusion directly
from fusions.MMF_GR_Add_Class import MMF_GR_Add_Class

# Import the argument setup from your main library
from main import get_args_from_parser, update_args_for_dataset, update_args_for_model

from sklearn.metrics import confusion_matrix, classification_report

def train_classification():
    # --- STEP 1: INITIALIZE ARGS FIRST ---
    # This fixes the UnboundLocalError
    args = get_args_from_parser()
    args.llm_layers_fusion = "full" 
    args.task = 'classification' # Custom flag for your research
    args = update_args_for_dataset(args) 
    set_seed(args.seed)

    # --- STEP 2: LOAD DATA ---
    data_obj = parse_datasets(args, show_summary=False)
    args.C = data_obj["input_dim"] 
    
    # Define these variables so the loops can find them
    train_loader = data_obj["train_dataloader"]
    val_loader = data_obj["val_dataloader"]
    test_loader = data_obj["test_dataloader"] # Good to have for later
    
    # --- STEP 3: UPDATE MODEL-SPECIFIC ARGS ---
    # This locks in the latent dimensions (e.g., hid_dim=32)
    args = update_args_for_model(args)   
    train_loader = data_obj["train_dataloader"]

    # --- STEP 4: INITIALIZE CLASSIFICATION FUSION ---
    # We bypass FusionModel.py to keep the forecasting path untouched.
    # We use args.hid_dim (32) for 'C' to fix the 774 vs 800 error.
    mmf_class = MMF_GR_Add_Class(
        d_txt=768, 
        C=args.hid_dim, 
        hidden_dim=args.hid_dim
    ).to(args.device)

    # --- STEP 5: INITIALIZE MODEL ---
    model = MultimodalStockClassifier(args, mmf_class)
    model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss() 

    # --- STEP 6: TRAINING & VALIDATION LOOP ---
    print(f"--- Starting Training on {args.dataset} ---")
    
    for epoch in range(args.epoch):
        # 1. TRAINING PHASE
        model.train()
        epoch_loss = 0
        for i, batch in enumerate(train_loader):
            # A. Data Extraction & Text Masking
            X_ts = batch['observed_data'].to(args.device)      
            mask = batch['observed_mask'].to(args.device)      
            tt = batch['observed_tp'].to(args.device)          
            text_emb = batch['notes_embeddings'].to(args.device) if args.enable_text else None
            
            if text_emb is not None:
                text_mask = (torch.sum(torch.abs(text_emb), dim=-1) != 0).float().to(args.device)
            else:
                text_mask = None

            B, M, L_in, N = X_ts.shape
            
            # B. Encoder Preprocessing (Flattening & Time Embeddings)
            X_flat = X_ts.permute(0, 3, 1, 2).reshape(-1, L_in, 1)
            mask_flat = mask.permute(0, 3, 1, 2).reshape(-1, L_in, 1)
            
            # Handle timestamps dynamically
            if tt.dim() == 4:
                tt_flat = tt.permute(0, 3, 1, 2).reshape(-1, L_in, 1)
            else:
                tt_flat = tt.unsqueeze(1).repeat(1, N, 1, 1).reshape(-1, L_in, 1)

            # Sync batch size for tPatchGNN internal reshaping
            model.encoder.batch_size = B
            te_his = model.encoder.LearnableTE(tt_flat) 
            X_final = torch.cat([X_flat, te_his], dim=-1) 

            # C. Label Generation (1 = Price Increase)
            current_price = X_ts[:, -1, -1, 0] 
            next_price = batch['data_to_predict'][:, 0, 0].to(args.device)
            labels = (next_price > current_price).float().unsqueeze(1).to(args.device)
            
            # D. Optimization Step
            optimizer.zero_grad()
            outputs = model(X_final, mask_flat, text_emb, text_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch {epoch}, Iter {i}, Loss: {loss.item():.4f}")

        # 2. VALIDATION PHASE (Evaluation)
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad(): # Disable gradient tracking to save memory
            for batch in val_loader:
                # Replicate exactly the same prep as Training
                X_ts_v = batch['observed_data'].to(args.device)
                tt_v = batch['observed_tp'].to(args.device)
                mask_v = batch['observed_mask'].to(args.device)
                text_emb_v = batch['notes_embeddings'].to(args.device) if args.enable_text else None
                
                B_v, M_v, L_v, N_v = X_ts_v.shape
                X_flat_v = X_ts_v.permute(0, 3, 1, 2).reshape(-1, L_v, 1)
                
                if tt_v.dim() == 4:
                    tt_flat_v = tt_v.permute(0, 3, 1, 2).reshape(-1, L_v, 1)
                else:
                    tt_flat_v = tt_v.unsqueeze(1).repeat(1, N_v, 1, 1).reshape(-1, L_v, 1)
                
                mask_flat_v = mask_v.permute(0, 3, 1, 2).reshape(-1, L_v, 1)
                
                if text_emb_v is not None:
                    text_mask_v = (torch.sum(torch.abs(text_emb_v), dim=-1) != 0).float().to(args.device)
                else:
                    text_mask_v = None

                model.encoder.batch_size = B_v
                X_final_v = torch.cat([X_flat_v, model.encoder.LearnableTE(tt_flat_v)], dim=-1)

                # Ground Truth
                curr_p_v = X_ts_v[:, -1, -1, 0]
                next_p_v = batch['data_to_predict'][:, 0, 0].to(args.device)
                labels_v = (next_p_v > curr_p_v).float().unsqueeze(1).to(args.device)

                # Accuracy Metrics
                outputs_v = model(X_final_v, mask_flat_v, text_emb_v, text_mask_v)
                preds = (outputs_v > 0.5).float()
                val_correct += (preds == labels_v).sum().item()
                val_total += labels_v.size(0)

        # 3. EPOCH SUMMARY
        epoch_acc = val_correct / val_total
        print(f"--- Epoch {epoch} Final: Avg Loss: {epoch_loss/len(train_loader):.4f} | Val Accuracy: {epoch_acc:.2%}")

    print("--- Training Complete. Moving to Final Test ---")
    evaluate_model(model, test_loader, args, args.device, split_name="Final Test")

def evaluate_model(model, loader, args, device, split_name="Test"):
    model.eval()
    all_labels = []
    all_preds = []
    
    print(f"--- Running {split_name} Evaluation ---")
    with torch.no_grad():
        for batch in loader:
            # A. Data Prep (Mirrors Training Loop)
            X_ts = batch['observed_data'].to(device)
            tt = batch['observed_tp'].to(device)
            mask = batch['observed_mask'].to(device)
            text_emb = batch['notes_embeddings'].to(device) if args.enable_text else None
            
            B, M, L, N = X_ts.shape
            X_flat = X_ts.permute(0, 3, 1, 2).reshape(-1, L, 1)
            
            if tt.dim() == 4:
                tt_flat = tt.permute(0, 3, 1, 2).reshape(-1, L, 1)
            else:
                tt_flat = tt.unsqueeze(1).repeat(1, N, 1, 1).reshape(-1, L, 1)
            
            mask_flat = mask.permute(0, 3, 1, 2).reshape(-1, L, 1)
            text_mask = (torch.sum(torch.abs(text_emb), dim=-1) != 0).float().to(device) if text_emb is not None else None

            # B. Forward Pass
            model.encoder.batch_size = B
            X_final = torch.cat([X_flat, model.encoder.LearnableTE(tt_flat)], dim=-1)
            
            # C. Ground Truth Logic
            curr_p = X_ts[:, -1, -1, 0]
            next_p = batch['data_to_predict'][:, 0, 0].to(device)
            labels = (next_p > curr_p).float().unsqueeze(1).to(device)

            outputs = model(X_final, mask_flat, text_emb, text_mask)
            preds = (outputs > 0.5).float()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # D. Calculate Metrics
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=['No Buy (0)', 'Buy (1)'])
    
    print(report)
    print("Confusion Matrix:")
    print(cm)
    return report, cm

if __name__ == "__main__":
    train_classification()