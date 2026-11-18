"""
modal_train.py
──────────────
Cloud orchestration pipeline for ARLDM on Modal.

Modes (via --mode flag):
  train   — Resume or start training on an A100 GPU
  sample  — Run inference and download generated images to local machine
  prepare — Run the dataset preprocessing script inside the container

Usage:
  modal run modal_train.py --mode train --epochs 40 --ckpt visdrone_run/last.ckpt
  modal run modal_train.py --mode sample --ckpt visdrone_run/last.ckpt --steps 150
  modal run modal_train.py --mode prepare
"""

import modal
import os
import subprocess
import sys

# ── App & Image ────────────────────────────────────────────────────────────────

app = modal.App("arldm-visdrone-pipeline")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "git")
    .pip_install(
        "torch", "torchvision", "pytorch-lightning", "transformers>=4.31.0",
        "diffusers>=0.20.0", "timm", "ftfy", "hydra-core", "opencv-python-headless",
        "h5py", "scipy", "tensorboard", "omegaconf", "accelerate", "typing_extensions"
    )
    .run_commands(
        "python -c 'from transformers import CLIPTokenizer; CLIPTokenizer.from_pretrained(\"runwayml/stable-diffusion-v1-5\", subfolder=\"tokenizer\")'",
        "python -c 'from transformers import CLIPTextModel; CLIPTextModel.from_pretrained(\"runwayml/stable-diffusion-v1-5\", subfolder=\"text_encoder\")'",
        "python -c 'from diffusers import AutoencoderKL, UNet2DConditionModel; AutoencoderKL.from_pretrained(\"runwayml/stable-diffusion-v1-5\", subfolder=\"vae\"); UNet2DConditionModel.from_pretrained(\"runwayml/stable-diffusion-v1-5\", subfolder=\"unet\")'"
    )
    .add_local_dir(".", remote_path="/root/ARLDM", ignore=[".git", "venv", "__pycache__", "ckpts", "samples", "results", "*.h5", "*.ckpt", "*.log", ".DS_Store"])
)

# ── Volumes ────────────────────────────────────────────────────────────────────

checkpoint_volume = modal.Volume.from_name("arldm-checkpoints", create_if_missing=True)
dataset_volume = modal.Volume.from_name("arldm-datasets", create_if_missing=True)

STORAGE_PATH = "/root/ARLDM/storage"   # checkpoint volume mount
DATA_PATH = "/root/ARLDM/data"         # dataset volume mount

# ── Helpers ────────────────────────────────────────────────────────────────────

def _setup_paths():
    """Symlink the dataset and checkpoint directories into the working tree."""
    dataset_path = os.path.join(STORAGE_PATH, "visdrone.h5")
    if not os.path.exists(dataset_path):
        # Fall back to the separate dataset volume
        dataset_path = os.path.join(DATA_PATH, "visdrone.h5")
    if not os.path.exists(dataset_path):
        print(f"ERROR: visdrone.h5 not found in {STORAGE_PATH} or {DATA_PATH}")
        print(">>> Contents of storage:")
        subprocess.run(["ls", "-R", STORAGE_PATH])
        return False
    if not os.path.exists("visdrone.h5"):
        os.symlink(dataset_path, "visdrone.h5")
        print(f">>> Symlinked dataset from {dataset_path}")
    if not os.path.exists("ckpts"):
        os.symlink(STORAGE_PATH, "ckpts")
    return True


def _resolve_ckpt(ckpt: str, mode: str) -> str | None:
    """Normalize a checkpoint path and verify it exists in the container."""
    if not ckpt:
        if mode == "sample":
            for candidate in [
                "ckpts/visdrone_dit_b_vit_init/last.ckpt",
                "ckpts/visdrone_dit_b_vit_init/last-v3.ckpt",
                "ckpts/visdrone_run/last.ckpt"
            ]:
                if os.path.exists(candidate):
                    return candidate
            return None
        return None
    if not ckpt.startswith("/") and not ckpt.startswith("ckpts/"):
        ckpt = f"ckpts/{ckpt}"
    if os.path.exists(ckpt):
        print(f">>> Checkpoint found: {os.path.abspath(ckpt)}")
        return ckpt
    print(f"WARNING: Checkpoint not found at {ckpt}. Starting from scratch.")
    return None

# ── Modal Functions ────────────────────────────────────────────────────────────

@app.function(
    image=image,
    volumes={
        STORAGE_PATH: checkpoint_volume,
        DATA_PATH: dataset_volume,
    },
    timeout=7200
)
def prepare_data():
    """Run the VisDrone HDF5 preprocessing script inside the Modal container."""
    print("Starting data preparation on Modal...")
    subprocess.run(["python", "data_script/visdrone_hdf5.py"], cwd="/root/ARLDM", check=True)
    checkpoint_volume.commit()
    print("Data preparation complete.")


@app.function(
    image=image,
    gpu="A100",
    timeout=57600,
    volumes={
        STORAGE_PATH: checkpoint_volume,
        DATA_PATH: dataset_volume,
    },
    env={"PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
)
def train_on_modal(epochs: int, mode: str = "train", ckpt: str = None, steps: int = None, run_name: str = "visdrone_dit_b_vit_init", guidance: float = None, scheduler: str = None):
    """
    Run training or sampling inside the Modal A100 container.

    Args:
        epochs:   Target max_epochs for the trainer.
        mode:     'train' or 'sample'.
        ckpt:     Relative path to a checkpoint file inside the volume.
        steps:    Number of DDIM inference steps (sampling only).
        run_name: Experiment run name for checkpoint directory.
        guidance: Classifier-free guidance scale (sampling only).
    """
    print(f"Starting ARLDM {mode} on Modal (A100)...")
    os.chdir("/root/ARLDM")

    if not _setup_paths():
        return False

    cmd = [
        "python", "main.py",
        f"mode={mode}",
        f"max_epochs={epochs}",
        "accelerator=gpu",
        "pin_memory=True",
        f"run_name={run_name}"
    ]

    if steps:
        cmd.append(f"num_inference_steps={steps}")

    if guidance is not None:
        cmd.append(f"guidance_scale={guidance}")

    if scheduler:
        cmd.append(f"scheduler={scheduler}")

    resolved = _resolve_ckpt(ckpt, mode)
    if resolved:
        key = "train_model_file" if mode == "train" else "test_model_file"
        cmd.append(f"{key}={resolved}")

    try:
        subprocess.run(cmd, check=True)
        return True
    except KeyboardInterrupt:
        print("Interrupted by user, saving progress...")
        return False
    except Exception as e:
        print(f"Task failed: {e}")
        return False
    finally:
        print("Committing volume changes to Modal storage...")
        checkpoint_volume.commit()

@app.function(
    image=image,
    volumes={
        STORAGE_PATH: checkpoint_volume,
    },
    timeout=60
)
def inspect_ckpt_on_modal(ckpt: str = "visdrone_dit_b_vit_init/last-v1.ckpt"):
    """Quickly read checkpoint metadata directly in the cloud without downloading."""
    import torch
    import os

    # Check potential paths
    candidates = [
        ckpt,
        os.path.join(STORAGE_PATH, ckpt),
        os.path.join(STORAGE_PATH, "visdrone_dit_b_vit_init", os.path.basename(ckpt)),
        os.path.join(STORAGE_PATH, "visdrone_run", os.path.basename(ckpt)),
    ]
    target = None
    for c in candidates:
        if os.path.exists(c) and not os.path.isdir(c):
            target = c
            break

    print("\n" + "=" * 50)
    print(">>> MODAL CLOUD STORAGE CONTENTS:")
    for root, dirs, files in os.walk(STORAGE_PATH):
        for f in files:
            p = os.path.join(root, f)
            rel = os.path.relpath(p, STORAGE_PATH)
            sz = os.path.getsize(p) / (1024 * 1024)
            print(f"  • {rel} ({sz:.1f} MB)")
    print("=" * 50)

    if not target:
        print(f"\n❌ Checkpoint '{ckpt}' not found in volume.\n")
        return

    print(f"\n>>> Reading metadata from: {target}")
    try:
        data = torch.load(target, map_location="cpu", weights_only=False)
        print("\n=== Checkpoint Metadata ===")
        print(f"Target file:  {os.path.basename(target)}")
        print(f"Last Epoch:   {data.get('epoch', 'N/A')}")
        print(f"Global Step:  {data.get('global_step', 'N/A')}")
        print(f"PL Version:   {data.get('pytorch-lightning_version', 'N/A')}")

        state_dict = data.get("state_dict", {})
        gates = [round(float(state_dict[f"unet.blocks.{i}.pos_gate"].squeeze()), 4) for i in range(12) if f"unet.blocks.{i}.pos_gate" in state_dict]
        if gates:
            print(f"Pos Gates (12 blocks): {gates}")
            print(f"Average Gate Magnitude: {sum(gates)/len(gates):.4f}")
        else:
            print("Pos Gates: Not found in state_dict")
        print("===========================\n")
    except Exception as e:
        print(f"Error loading checkpoint metadata: {e}\n")


@app.function(
    image=image,
    gpu="A100",
    timeout=18000,
    volumes={
        STORAGE_PATH: checkpoint_volume,
        DATA_PATH: dataset_volume,
    },
    env={"PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
)
def train_vae_on_modal(epochs: int = 2, lr: float = 1e-4, start_epoch: int = None):
    """Run VAE decoder fine-tuning directly on Modal A100."""
    print("Starting VisDrone VAE Decoder Fine-Tuning on Modal (A100)...")
    os.chdir("/root/ARLDM")
    if not _setup_paths():
        return False

    out_dir = os.path.join(STORAGE_PATH, "visdrone_vae")
    cmd = [
        "python", "train_vae_decoder.py",
        f"--epochs={epochs}",
        f"--lr={lr}",
        f"--output_dir={out_dir}"
    ]
    if start_epoch is not None:
        cmd.append(f"--start_epoch={start_epoch}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        print(f"VAE training failed: {e}")
        return False
    finally:
        if os.path.exists(out_dir):
            print("Committing fine-tuned VAE to volume...")
            checkpoint_volume.commit()
            print("✅ Fine-tuned VAE saved in storage/visdrone_vae!")


# ── Local Entrypoint ───────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(mode: str = "train", epochs: int = 10, local: bool = False, ckpt: str = None, steps: int = None, run_name: str = "visdrone_dit_b_vit_init", guidance: float = None, scheduler: str = None, start_epoch: int = None):
    """
    Orchestrate the pipeline from your local machine.

    Examples:
      modal run modal_train.py --mode train_vae --epochs 2 --start-epoch 3
      modal run modal_train.py --mode inspect --ckpt visdrone_dit_b_vit_init/last-v1.ckpt
      modal run modal_train.py --mode train --epochs 35 --ckpt visdrone_dit_b_vit_init/last-v1.ckpt
      modal run modal_train.py --mode sample --ckpt visdrone_dit_b_vit_init/last-v1.ckpt --steps 50 --guidance 2.2
      modal run modal_train.py --mode prepare
    """
    if mode in ["train_vae", "vae"]:
        print(f"--- Launching {epochs}-epoch VAE Decoder fine-tuning on Modal ---")
        train_vae_on_modal.remote(epochs=epochs, start_epoch=start_epoch)
        return

    if mode == "prepare":
        prepare_data.remote()
        return

    if mode == "inspect":
        target_ckpt = ckpt if ckpt else f"{run_name}/last-v1.ckpt"
        inspect_ckpt_on_modal.remote(ckpt=target_ckpt)
        return

    if mode == "train":
        print(f"--- Launching {epochs}-epoch training on Modal ---")
        success = train_on_modal.remote(epochs, "train", ckpt=ckpt, run_name=run_name)
        if success:
            print("--- Training finished. Downloading checkpoint to local machine ---")
            local_ckpt_path = f"./ckpts/{run_name}/"
            os.makedirs(local_ckpt_path, exist_ok=True)
            subprocess.run(["modal", "volume", "get", "arldm-checkpoints", f"{run_name}/last.ckpt", local_ckpt_path, "--force"])
            print(f"--- Checkpoint saved to {local_ckpt_path} ---")

    elif mode == "sample":
        if local:
            print("--- Running inference locally (MPS) ---")
            sample_ckpt = ckpt if ckpt else f"./ckpts/{run_name}/last.ckpt"
            local_cmd = [
                "python3", "main.py",
                "mode=sample",
                "accelerator=mps",
                "devices=1",
                f"test_model_file={sample_ckpt}"
            ]
            if guidance is not None:
                local_cmd.append(f"guidance_scale={guidance}")
            if steps is not None:
                local_cmd.append(f"num_inference_steps={steps}")
            subprocess.run(local_cmd)
        else:
            print(f"--- Running inference on Modal (A100), ckpt: {ckpt or 'last.ckpt'}, guidance: {guidance or 2.2}, scheduler: {scheduler or 'default'} ---")
            train_on_modal.remote(1, "sample", ckpt=ckpt, steps=steps, run_name=run_name, guidance=guidance, scheduler=scheduler)
            print("--- Downloading samples from Modal ---")
            local_samples_path = "./samples"
            os.makedirs(local_samples_path, exist_ok=True)
            subprocess.run(["modal", "volume", "get", "arldm-checkpoints", f"{run_name}/samples", local_samples_path, "--force"])
            print(f"--- Samples saved to {local_samples_path}/samples ---")
