# train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data import generate_transitions
from model import TinyWorldModelMLP


def main():
    print("Generating dataset...")
    X, A, Y = generate_transitions(
        num_episodes=64,
        steps_per_episode=100,
    )
    
    # Create a PyTorch DataLoader
    dataset = TensorDataset(X, A, Y)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Initialize model, device, and optimizer
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = TinyWorldModelMLP().to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Calculate pos_weight and define the loss function
    pos_count = float(Y.sum().item())
    neg_count = float(Y.numel() - pos_count)
    
    pos_weight_value = min(neg_count / max(pos_count, 1.0), 50.0)
    
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"Dataset size: {len(dataset)}")
    print(f"pos_weight: {pos_weight_value:.2f}")

    epochs = 100
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        
        for batch_x, batch_a, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_a = batch_a.to(device)
            batch_y = batch_y.to(device)
            
            logits = model(batch_x, batch_a)
            
            loss = loss_fn(logits, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
            total_items += batch_x.size(0)

        avg_loss = total_loss / max(total_items, 1)
        print(f"epoch={epoch:02d} loss={avg_loss:.4f}")

    torch.save(model.state_dict(), "world_model_mlp.pt")
    print("Saved model to world_model_mlp.pt")


if __name__ == "__main__":
    main()