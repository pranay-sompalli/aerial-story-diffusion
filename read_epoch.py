import torch
import sys
import os

def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "ckpts/visdrone_run/last.ckpt"
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint file not found at {ckpt_path}")
        return

    print(f"Loading checkpoint metadata from: {ckpt_path}...")
    try:
        # Load checkpoint to CPU (weights are not needed, just metadata)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # PyTorch Lightning stores 'epoch' and 'global_step' at the top level
        epoch = checkpoint.get('epoch', None)
        global_step = checkpoint.get('global_step', None)
        pl_version = checkpoint.get('pytorch-lightning_version', 'Unknown')
        
        print("\n=== Checkpoint Metadata ===")
        print(f"Last Epoch completed: {epoch}")
        print(f"Global Step:          {global_step}")
        print(f"PL Version:           {pl_version}")
        print("===========================")
    except Exception as e:
        print(f"Failed to read checkpoint: {e}")

if __name__ == '__main__':
    main()
