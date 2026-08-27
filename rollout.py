# rollout.py

import torch
import numpy as np

from env import MovingSquareEnv
from model import TinyWorldModelMLP


def save_image_strip(frames_list, filename):
    """Helper to save a list of (1, 32, 32) frames as a single horizontal image."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping image save.")
        return

    n = len(frames_list)
    fig, axes = plt.subplots(1, n, figsize=(n * 1.2, 1.2))
    if n == 1:
        axes = [axes]
        
    for ax, frame in zip(axes, frames_list):
        # frame shape is (1, 32, 32), we want (32, 32) for imshow
        ax.imshow(frame[0], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    env = MovingSquareEnv(size=32, square_size=4, speed=2.0, max_steps=100)
    model = TinyWorldModelMLP().to(device)
    
    state_dict = torch.load("world_model_mlp.pt", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    
    model.eval() # Put model in evaluation mode
    
    obs = env.reset(seed=42)
    
    # We will record both the true environment frames and the model's imagined frames
    true_frames = [obs]
    pred_frames = [obs]
    
    # Let's do: Right (4) x 5, Down (2) x 5, Left (3) x 5, Up (1) x 5
    actions = [4]*5 + [2]*5 + [3]*5 + [1]*5
    
    # This will be the input to the model for the *next* step
    current_model_input = obs 
    
    for action in actions:
        
        # --- TRUE ENVIRONMENT STEP ---
        true_obs, _, _, _ = env.step(action)
        true_frames.append(true_obs)
        
        # --- MODEL IMAGINATION STEP ---
        x_tensor = torch.from_numpy(current_model_input).unsqueeze(0).to(device)
        a_tensor = torch.tensor([action], dtype=torch.long, device=device)
        
        with torch.no_grad():
            logits = model(x_tensor, a_tensor)
            probs = torch.sigmoid(logits)
            
        pred_binary = (probs > 0.5).float()
        pred_np = pred_binary.cpu().numpy()[0]
        
        pred_frames.append(pred_np)
        current_model_input = pred_np

    save_image_strip(true_frames, "rollout_true.png")
    save_image_strip(pred_frames, "rollout_pred.png")


if __name__ == "__main__":
    main()