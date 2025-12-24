import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np

# --- IMPORTS FROM YOUR MODULES ---
import sys
from pathlib import Path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "data"))

from models.alignn import ALIGNN
from build_graphs import ALIGNNDataset #type: ignore

# --- CONFIGURATION ---
BATCH_SIZE = 128  # Increased for better GPU utilization
LEARNING_RATE = 1e-3
EPOCHS = 100
VALIDATE_EVERY = 5  # Validate every N epochs instead of every epoch
# Device selection: CUDA > MPS > CPU
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

def train():
    import os
    from pathlib import Path
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    checkpoint_dir = project_root / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # 1. Prepare Data
    print("Loading Dataset...")
    dataset_root = project_root / "dataset"
    full_dataset = ALIGNNDataset(root=str(dataset_root))
    
    # Stratified split is better, but random is fine for now
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Set num_workers=0 to avoid multiprocessing issues with MPS
    # MPS has known compatibility issues with PyTorch Geometric multiprocessing
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. Initialize Model
    # n_atom_input=4 for physical features: [Electronegativity, Radius, Mass, Melting]
    # hidden_dim=128: Hidden dimension for ALIGNN layers
    # n_layers=4: Number of ALIGNN update layers (atom + line graph)
    model = ALIGNN(n_atom_input=4, hidden_dim=96, n_layers=3).to(DEVICE)
    
    # 1. OPTIMIZER: Use AdamW (Better for deep models like ALIGNN)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # 2. SCHEDULER: OneCycleLR helps ALIGNN converge faster
    # Steps per epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE, 
        steps_per_epoch=len(train_loader), 
        epochs=EPOCHS
    )
    
    # Loss Functions: L1Loss (MAE) for robustness against outliers
    criterion_bg = nn.L1Loss()
    criterion_ehull = nn.L1Loss()

    print("Starting ALIGNN Training (Log-Space Bandgap)...")
    best_val_loss = float('inf')

    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        
        for batch in loop:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            
            # Forward Pass
            # Expecting model to output [Batch, 1] for both
            pred_bg_raw, pred_hull = model(batch)
            
            # --- CRITICAL FIX: Log-Transform Targets ---
            # Real: 0.0 eV -> Log: 0.0
            # Real: 1.5 eV -> Log: 0.91
            # Real: 5.0 eV -> Log: 1.79 (Outliers squashed)
            # Transform model output to log1p space (ensure non-negative)
            pred_bg_log = torch.log1p(torch.clamp(pred_bg_raw.squeeze(), min=0.0))
            target_bg_log = torch.log1p(batch.y_bandgap)
            
            # Clamp Stability (0.0 - 0.5 eV)
            target_hull_clamped = torch.clamp(batch.y_ehull, max=0.5)
            
            # Calculate Loss
            loss_bg = criterion_bg(pred_bg_log, target_bg_log)
            loss_hull = criterion_ehull(pred_hull.squeeze(), target_hull_clamped)
            
            # Weighting: Bandgap is harder, give it more focus now that stability is good
            loss = (1.0 * loss_bg) + (5.0 * loss_hull)
            
            # Backward Pass
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update LR every batch
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # 4. Validation Loop (Un-log to check real MAE)
        avg_train_loss = total_loss / len(train_loader)
        
        if epoch % VALIDATE_EVERY == 0 or epoch == 0:
            val_loss, mae_bg, mae_hull = evaluate(model, val_loader)
            print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f}")
            print(f"   >> Val MAE Bandgap: {mae_bg:.3f} eV | Stability: {mae_hull:.3f} eV")
            
            # Save Best Model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")
                print("   >> Model Saved!")
        else:
            print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f} | (Validation skipped)")

def evaluate(model, loader):
    model.eval()
    mae_bg = 0
    mae_hull = 0
    total_val_loss = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            pred_bg_raw, pred_hull = model(batch)  # Model outputs raw predictions
            
            # --- IMPORTANT: Convert to log1p space then un-log for human metrics ---
            # 1. Convert raw output to log1p space
            pred_bg_log = torch.log1p(torch.clamp(pred_bg_raw.squeeze(), min=0.0))
            target_bg_log = torch.log1p(batch.y_bandgap)
            
            # 2. Convert back to eV: exp(x) - 1
            pred_bg_real = torch.expm1(pred_bg_log)
            
            # 3. Calculate real MAE in eV
            mae_bg += (pred_bg_real - batch.y_bandgap).abs().sum().item()
            
            # 4. Calculate stability MAE
            target_hull_clamped = torch.clamp(batch.y_ehull, max=0.5)
            mae_hull += (pred_hull.squeeze() - target_hull_clamped).abs().sum().item()
            
            # 5. Calculate validation loss (same as training)
            loss_bg = nn.L1Loss()(pred_bg_log, target_bg_log)
            loss_hull = nn.L1Loss()(pred_hull.squeeze(), target_hull_clamped)
            total_val_loss += ((1.0 * loss_bg) + (5.0 * loss_hull)).item()
            
    num_samples = len(loader.dataset)
    avg_val_loss = total_val_loss / len(loader)
    return avg_val_loss, mae_bg / num_samples, mae_hull / num_samples

if __name__ == "__main__":
    train()