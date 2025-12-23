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
sys.path.insert(0, str(project_root / "data"))

from models.crystalgnn import CrystallGNN
from build_graphs import PerovskiteDataset

# --- CONFIGURATION ---
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Initialize Model
    # These params match the defaults in crystalgnn.py
    model = CrystallGNN(n_atom_feats=64, n_rbf=50, n_conv=3).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Loss Function: MSE is standard for regression
    criterion = nn.MSELoss()
    
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
            pred_bg, pred_ehull = model(batch)
            
            # Compute Loss
            # We predict two things, so we sum the losses
            loss_bg = criterion(pred_bg.squeeze(), batch.y_bandgap)
            loss_ehull = criterion(pred_ehull.squeeze(), batch.y_ehull)
            
            # Weighted Loss: You might care more about stability than bandgap
            # loss = 1.0 * loss_bg + 5.0 * loss_ehull 
            loss = loss_bg + loss_ehull
            
            # Backward Pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # 4. Validation Loop
        avg_train_loss = total_loss / len(train_loader)
        val_loss, mae_bg, mae_ehull = evaluate(model, val_loader, criterion)
        
        scheduler.step(val_loss)

        # Print Stats
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"    >> MAE Bandgap: {mae_bg:.3f} eV | MAE Stability: {mae_ehull:.3f} eV/atom")

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")
            print("    >> Model Saved!")

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    abs_err_bg = 0
    abs_err_ehull = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            pred_bg, pred_ehull = model(batch)
            
            loss_bg = criterion(pred_bg.squeeze(), batch.y_bandgap)
            loss_ehull = criterion(pred_ehull.squeeze(), batch.y_ehull)
            
            total_loss += (loss_bg + loss_ehull).item()
            
            # Calculate Mean Absolute Error (MAE) for human readability
            abs_err_bg += (pred_bg.squeeze() - batch.y_bandgap).abs().sum().item()
            abs_err_ehull += (pred_ehull.squeeze() - batch.y_ehull).abs().sum().item()
            
    num_samples = len(loader.dataset)
    return total_loss / len(loader), abs_err_bg / num_samples, abs_err_ehull / num_samples

if __name__ == "__main__":
    train()