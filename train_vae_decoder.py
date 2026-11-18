"""
train_vae_decoder.py
────────────────────
Fine-tune ONLY the VAE Decoder on VisDrone aerial imagery.

Key architectural guarantees:
  1. VAE Encoder is FROZEN: Latent distribution z ~ N(mu, sigma) remains 100% identical.
     All 150 epochs of trained DiT diffusion weights remain 100% compatible!
  2. VAE Decoder is TRAINABLE: Learns sharp reconstruction of aerial buildings,
     curbs, road markings, and drone vehicles, eliminating the SD 1.5 8x blur.
  3. Loss function: L1 reconstruction + 2D Sobel edge gradient loss directly in RGB space.
"""

import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL
from omegaconf import OmegaConf
from torchvision import transforms
from PIL import Image
import numpy as np


class SobelEdgeLoss(nn.Module):
    """Computes L1 loss on 2D image gradients in X and Y to enforce sharp edges."""
    def __init__(self, device):
        super().__init__()
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device).view(1, 1, 3, 3)
        ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=device).view(1, 1, 3, 3)
        self.kx = kx.repeat(3, 1, 1, 1)  # 3 channels
        self.ky = ky.repeat(3, 1, 1, 1)

    def forward(self, pred, target):
        grad_p_x = F.conv2d(pred, self.kx, padding=1, groups=3)
        grad_p_y = F.conv2d(pred, self.ky, padding=1, groups=3)
        grad_t_x = F.conv2d(target, self.kx, padding=1, groups=3)
        grad_t_y = F.conv2d(target, self.ky, padding=1, groups=3)
        return 0.5 * (F.l1_loss(grad_p_x, grad_t_x) + F.l1_loss(grad_p_y, grad_t_y))


def train_vae_decoder(
    h5_path: str = "visdrone.h5",
    output_dir: str = "ckpts/visdrone_vae",
    epochs: int = 2,
    start_epoch_arg: int = None,
    batch_size: int = 2,
    accumulate_grad_batches: int = 2,
    lr: float = 1e-4,
    edge_weight: float = 0.25,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print("\n" + "=" * 65)
    print(">>> VISDRONE VAE DECODER FINE-TUNING PIPELINE")
    print(f"  • Target Epochs:       {epochs}")
    print(f"  • Micro-batch (stories):{batch_size} (x 4 frames = {batch_size * 4} images/step)")
    print(f"  • Grad Accumulation:   {accumulate_grad_batches} (effective batch = {batch_size * accumulate_grad_batches * 4} images)")
    print(f"  • Learning Rate:       {lr}")
    print(f"  • Edge Weight:         {edge_weight}")
    print(f"  • Output Directory:    {output_dir}")
    print(f"  • Compute Device:      {device}")
    print("=" * 65 + "\n", flush=True)

    # 1. Load Pretrained SD 1.5 VAE or Resume Existing Fine-Tuned Checkpoint
    meta_path = os.path.join(output_dir, "training_meta.pt")
    start_epoch = 1
    if os.path.exists(output_dir) and os.path.exists(os.path.join(output_dir, "config.json")):
        print(f">>> Found existing fine-tuned VAE at '{output_dir}'. Resuming weights...", flush=True)
        vae = AutoencoderKL.from_pretrained(output_dir)
        if os.path.exists(meta_path):
            try:
                meta = torch.load(meta_path, map_location="cpu")
                completed_epoch = meta.get("epoch", 0)
                start_epoch = completed_epoch + 1
                print(f">>> Resuming fine-tuning from Epoch {start_epoch} (previously completed {completed_epoch} epochs).", flush=True)
            except Exception:
                start_epoch = 3
        else:
            start_epoch = 3
            print(f">>> Existing fine-tuned weights detected (training_meta.pt missing). Defaulting start to Epoch {start_epoch}.", flush=True)
    else:
        print(">>> Loading base SD 1.5 VAE from 'runwayml/stable-diffusion-v1-5'...", flush=True)
        vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")

    if start_epoch_arg is not None:
        start_epoch = start_epoch_arg
        print(f">>> Explicit start epoch provided: Starting at Epoch {start_epoch}.", flush=True)

    vae.to(device)

    # Enable gradient checkpointing to keep VRAM around 3-4 GB
    if hasattr(vae, "enable_gradient_checkpointing"):
        vae.enable_gradient_checkpointing()

    # 2. Freeze Encoder & Quant-Conv (Latent distribution must remain strictly frozen!)
    vae.encoder.eval()
    vae.quant_conv.eval()
    for p in vae.encoder.parameters():
        p.requires_grad = False
    for p in vae.quant_conv.parameters():
        p.requires_grad = False

    # 3. Enable gradients ONLY on Decoder and Post-Quant-Conv
    vae.decoder.train()
    vae.post_quant_conv.train()
    for p in vae.decoder.parameters():
        p.requires_grad = True
    for p in vae.post_quant_conv.parameters():
        p.requires_grad = True

    trainable_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in vae.parameters() if not p.requires_grad)
    print(f">>> VAE Setup Complete:")
    print(f"  • Frozen Parameters (Encoder + QuantConv): {frozen_params:,} (100% frozen)")
    print(f"  • Trainable Parameters (Decoder):          {trainable_params:,}\n", flush=True)

    # 4. Load VisDrone Dataset
    from transformers import CLIPTokenizer
    from models.blip_override.blip import init_tokenizer
    from datasets.visdrone import StoryDataset

    cfg = OmegaConf.create({
        "dataset": "visdrone",
        "task": "continuation",
        "visdrone": {"hdf5_file": h5_path, "max_length": 77}
    })
    clip_tok = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
    blip_tok = init_tokenizer()

    print(f">>> Loading VisDrone training dataset from {h5_path}...", flush=True)
    dataset = StoryDataset("train", cfg, clip_tok, blip_tok)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device == "cuda"),
        drop_last=True
    )
    print(f">>> Dataset ready: {len(dataset)} stories ({len(dataloader)} steps per epoch).\n", flush=True)

    # 5. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        [p for p in vae.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(dataloader))
    sobel_loss_fn = SobelEdgeLoss(device)

    # 6. Training Loop with Mixed Precision and Gradient Accumulation
    os.makedirs(output_dir, exist_ok=True)
    scaling_factor = 0.18215  # SD 1.5 standard latent scaling
    use_amp = (device == "cuda")
    optimizer.zero_grad()

    end_epoch = start_epoch + epochs - 1
    print(f">>> Fine-tuning VAE Decoder from Epoch {start_epoch} to Epoch {end_epoch} ({epochs} epochs total)...\n", flush=True)

    for epoch in range(start_epoch, end_epoch + 1):
        epoch_start = time.time()
        running_l1 = 0.0
        running_edge = 0.0
        running_total = 0.0

        for step, batch in enumerate(dataloader):
            # batch[0]: (B, 4, 3, 512, 512) target frames normalized in [-1, 1]
            images = batch[0].to(device)
            B, S, C, H, W = images.shape
            images_flat = images.view(B * S, C, H, W)

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
                # Step A: Encode with frozen encoder
                with torch.no_grad():
                    latents = vae.encode(images_flat).latent_dist.sample() * scaling_factor

                # Step B: Decode with trainable decoder
                unscaled_latents = latents / scaling_factor
                recon_images = vae.decode(unscaled_latents).sample

                # Step C: Compute L1 pixel loss + Sobel edge loss
                l1_loss = F.l1_loss(recon_images.float(), images_flat.float())
                edge_loss = sobel_loss_fn(recon_images.float(), images_flat.float())
                total_loss = (l1_loss + edge_weight * edge_loss) / accumulate_grad_batches

            total_loss.backward()

            if (step + 1) % accumulate_grad_batches == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(vae.decoder.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_l1 += l1_loss.item()
            running_edge += edge_loss.item()
            running_total += (total_loss.item() * accumulate_grad_batches)

            if (step + 1) % 50 == 0 or (step + 1) == len(dataloader):
                avg_l1 = running_l1 / (step + 1)
                avg_edge = running_edge / (step + 1)
                avg_tot = running_total / (step + 1)
                print(
                    f"Epoch [{epoch:02d}/{end_epoch:02d}] Step [{step+1:04d}/{len(dataloader):04d}] "
                    f"| Total: {avg_tot:.4f} | L1: {avg_l1:.4f} | Edge: {avg_edge:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}",
                    flush=True
                )

        epoch_time = time.time() - epoch_start
        print(f"\n>>> Epoch {epoch} complete in {epoch_time:.1f}s. Average Total Loss: {running_total/len(dataloader):.4f}", flush=True)
        print(f">>> Checkpointing fine-tuned VAE to '{output_dir}'...", flush=True)
        vae.save_pretrained(output_dir)
        torch.save({"epoch": epoch}, meta_path)
        print(f"✅ VAE successfully checkpointed at Epoch {epoch}!\n", flush=True)

    # Final Save
    print(f">>> Final save of fine-tuned VAE to '{output_dir}'...", flush=True)
    vae.save_pretrained(output_dir)
    torch.save({"epoch": end_epoch}, meta_path)
    print("✅ VAE Decoder fine-tuning complete and saved successfully!\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path", type=str, default="visdrone.h5")
    parser.add_argument("--output_dir", type=str, default="ckpts/visdrone_vae")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--start_epoch", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accumulate_grad_batches", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--edge_weight", type=float, default=0.25)
    args = parser.parse_args()

    train_vae_decoder(
        h5_path=args.h5_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        start_epoch_arg=args.start_epoch,
        batch_size=args.batch_size,
        accumulate_grad_batches=args.accumulate_grad_batches,
        lr=args.lr,
        edge_weight=args.edge_weight
    )

