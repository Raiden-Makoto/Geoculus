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

from models.crystalgnn import CrystallGNN
from build_graphs import PerovskiteDataset #type: ignore

# --- CONFIGURATION ---
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 100
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
    full_dataset = PerovskiteDataset(root=str(dataset_root))
    
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
    # n_global_feats=2 for global features: [tolerance_factor, packing_fraction]
    model = CrystallGNN(n_atom_input=4, n_atom_feats=64, n_global_feats=2, n_rbf=50, n_conv=3).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Loss Functions: L1Loss (MAE) for robustness against outliers
    criterion_bg = nn.L1Loss()
    criterion_ehull = nn.L1Loss()
    
    # Scheduler: Reduce LR if validation loss stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    print(f"Starting training on {DEVICE}...")
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
            pred_bg_log, pred_ehull = model(batch)  # Model outputs log(Eg) for band gap
            
            # --- FIX: Log-Space Training ---
            # 1. Transform Targets: Compress the high values
            target_bg_log = torch.log1p(batch.y_bandgap)
            
            # 2. Compute Loss in Log-Space
            # We use L1Loss (MAE) on the logs for maximum robustness against outliers
            loss_bg = criterion_bg(pred_bg_log.squeeze(), target_bg_log)
            
            # Stability uses standard linear clamping (as before)
            loss_ehull = criterion_ehull(pred_ehull.squeeze(), torch.clamp(batch.y_ehull, max=0.5))
            
            # Combine with weighted loss (stability is 5x more important)
            loss = loss_bg + (5.0 * loss_ehull)
            
            # Backward Pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # 4. Validation Loop
        avg_train_loss = total_loss / len(train_loader)
        mae_bg, mae_ehull = evaluate(model, val_loader)
        
        # Calculate validation loss for scheduler
        val_loss = mae_bg + (5.0 * mae_ehull)
        scheduler.step(val_loss)

        # Print Stats
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"    >> MAE Bandgap: {mae_bg:.3f} eV | MAE Stability: {mae_ehull:.3f} eV/atom")

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")
            print("    >> Model Saved!")

def evaluate(model, loader):
    model.eval()
    mae_bg = 0
    mae_ehull = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            pred_bg_log, pred_ehull = model(batch)  # Model outputs Log(Eg) for band gap
            
            # --- IMPORTANT: Un-Log for Human Metrics ---
            # Convert predictions back to eV: exp(x) - 1
            pred_bg_real = torch.expm1(pred_bg_log.squeeze())
            
            # Calculate real MAE in eV
            mae_bg += (pred_bg_real - batch.y_bandgap).abs().sum().item()
            mae_ehull += (pred_ehull.squeeze() - batch.y_ehull).abs().sum().item()
            
    num_samples = len(loader.dataset)
    return mae_bg / num_samples, mae_ehull / num_samples

if __name__ == "__main__":
    train()