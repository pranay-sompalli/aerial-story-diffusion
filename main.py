# Core imports for model definition and training utilities
import inspect
import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0" 

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, LMSDiscreteScheduler, PNDMScheduler, DDIMScheduler, DPMSolverMultistepScheduler
from diffusers.training_utils import compute_snr
from omegaconf import DictConfig
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel

from fid_utils import calculate_fid_given_features
from models.blip_override.blip import blip_feature_extractor, init_tokenizer
from models.diffusers_override.unet_2d_condition import UNet2DConditionModel
from models.inception import InceptionV3
from models.perceiver_dit import PerceiverIODiT


# DataModule for loading story datasets for training/validation/testing.
# It encapsulates data loading details, splitting, and dataloader setups.
class LightningDataset(pl.LightningDataModule):
    def __init__(self, args: DictConfig, clip_tokenizer=None, blip_tokenizer=None):
        super(LightningDataset, self).__init__()
        # Standard configuration options passed to DataLoader.
        # Set persistent_workers to prevent worker process recreation overhead between epochs.
        self.kwargs = {
            "num_workers": args.get("num_workers", 0),
            "persistent_workers": args.get("num_workers", 0) > 0,
            "pin_memory": args.get("pin_memory", False)
        }
        self.args = args
        self.clip_tokenizer = clip_tokenizer
        self.blip_tokenizer = blip_tokenizer

    def setup(self, stage="fit"):
        # Dynamically import the dataset module corresponding to the user config.
        # This keeps imports clean and avoids loading unnecessary datasets.
        if self.args.dataset == "pororo":
            import datasets.pororo as data
        elif self.args.dataset == 'flintstones':
            import datasets.flintstones as data
        elif self.args.dataset == 'vistsis':
            import datasets.vistsis as data
        elif self.args.dataset == 'vistdii':
            import datasets.vistdii as data
        elif self.args.dataset == 'visdrone':
            import datasets.visdrone as data
        else:
            raise ValueError("Unknown dataset: {}".format(self.args.dataset))
            
        # Instantiate dataset objects depending on the active execution phase.
        if stage == "fit":
            print(">>> Creating training and validation datasets...")
            self.train_data = data.StoryDataset("train", self.args, self.clip_tokenizer, self.blip_tokenizer)
            self.val_data = data.StoryDataset("val", self.args, self.clip_tokenizer, self.blip_tokenizer)
            print(">>> Datasets created.")
        if stage == "test":
            self.test_data = data.StoryDataset("test", self.args, self.clip_tokenizer, self.blip_tokenizer)


    def train_dataloader(self):
        if not hasattr(self, 'trainloader'):
            self.trainloader = DataLoader(self.train_data, batch_size=self.args.batch_size, shuffle=True, **self.kwargs)
        return self.trainloader

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.args.batch_size, shuffle=False, **self.kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.args.batch_size, shuffle=False, **self.kwargs)

    def predict_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.args.batch_size, shuffle=False, **self.kwargs)

    def get_length_of_train_dataloader(self):
        if not hasattr(self, 'trainloader'):
            self.trainloader = DataLoader(self.train_data, batch_size=self.args.batch_size, shuffle=True, **self.kwargs)
        return len(self.trainloader)


# Autoregressive Latent Diffusion Model (ARLDM) Lightning Module.
# This model uses CLIP text embeddings + BLIP multimodal representations of past frames
# to auto-regressively generate multi-frame stories.
class ARLDM(pl.LightningModule):
    def __init__(self, args: DictConfig, steps_per_epoch=1):
        super(ARLDM, self).__init__()
        if getattr(self, "global_rank", 0) == 0:
            print(f">>> Initializing ARLDM Model...", flush=True)
        self.args = args
        self.steps_per_epoch = steps_per_epoch
        self.task = args.task

        # Initialize sampling scheduler and evaluation networks (Inception for FID calculation)
        if args.mode == 'sample':
            if args.scheduler == "pndm":
                self.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                               skip_prk_steps=True)
            elif args.scheduler == "ddim":
                self.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                               clip_sample=False, set_alpha_to_one=True)
            elif args.scheduler in ["dpm", "dpmsolver", "dpm++"]:
                self.scheduler = DPMSolverMultistepScheduler(
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    algorithm_type="dpmsolver++",
                    use_karras_sigmas=True,
                )
            elif args.scheduler in ["sde-dpm", "sde-dpmsolver", "dpm-sde", "dpmsolver++-sde"]:
                self.scheduler = DPMSolverMultistepScheduler(
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    algorithm_type="sde-dpmsolver++",
                    use_karras_sigmas=True,
                )
            else:
                raise ValueError("Scheduler not supported")
            
            # Augmentation and model setup for computing FID statistics during evaluation
            self.fid_augment = transforms.Compose([
                transforms.Resize([299, 299]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
            self.inception = InceptionV3([block_idx])

        # Tokenizers and image processors setup for both text & visual conditioning
        self.clip_tokenizer = CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="tokenizer")
        self.blip_tokenizer = init_tokenizer()
        self.blip_image_processor = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])
        self.max_length = args.get(args.dataset).max_length

        # Set up null tokens (all zeros/empty text) used for Classifier-Free Guidance (CFG).
        # We pre-compute and register these as persistent buffers to avoid recreating them at every step.
        blip_image_null_token = self.blip_image_processor(
            Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))).unsqueeze(0).float()
        clip_text_null_token = self.clip_tokenizer([""], padding="max_length", max_length=self.max_length,
                                                   return_tensors="pt").input_ids
        blip_text_null_token = self.blip_tokenizer([""], padding="max_length", max_length=self.max_length,
                                                   return_tensors="pt").input_ids

        self.register_buffer('clip_text_null_token', clip_text_null_token)
        self.register_buffer('blip_text_null_token', blip_text_null_token)
        self.register_buffer('blip_image_null_token', blip_image_null_token)

        # Load CLIP text encoder and adjust position embeddings for custom max sequence length.
        self.text_encoder = CLIPTextModel.from_pretrained('runwayml/stable-diffusion-v1-5',
                                                          subfolder="text_encoder")
        self.text_encoder.resize_token_embeddings(args.get(args.dataset).clip_embedding_tokens)
        
        # Resize position embeddings if our maximum sequence length exceeds standard CLIP limits
        old_embeddings = self.text_encoder.embeddings.position_embedding
        new_embeddings = self.text_encoder._get_resized_embeddings(old_embeddings, self.max_length)
        self.text_encoder.embeddings.position_embedding = new_embeddings
        self.text_encoder.config.max_position_embeddings = self.max_length
        self.text_encoder.max_position_embeddings = self.max_length
        self.text_encoder.embeddings.position_ids = torch.arange(self.max_length).expand((1, -1))

        # Learnable embeddings to distinguish different input modalities (CLIP text vs BLIP multimodal representation)
        self.modal_type_embeddings = nn.Embedding(2, 768)
        # Learnable embeddings to encode relative sequence position (which frame of the story)
        self.time_embeddings = nn.Embedding(5, 768)
        
        # BLIP multimodal context encoder: extracts joint image-text features from past frames
        self.mm_encoder = blip_feature_extractor(
            pretrained='https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth',
            image_size=224, vit='base')
        self.mm_encoder.text_encoder.resize_token_embeddings(args.get(args.dataset).blip_embedding_tokens)

        # Load VAE (autoencoder). Checks first for a fine-tuned VisDrone VAE, then falls back to SD 1.5.
        vae_custom = args.get('vae_path', 'ckpts/visdrone_vae')
        if vae_custom and os.path.exists(vae_custom):
            if self.global_rank == 0:
                print(f">>> Found fine-tuned VisDrone VAE at '{vae_custom}'. Loading...", flush=True)
            self.vae = AutoencoderKL.from_pretrained(vae_custom, low_cpu_mem_usage=True)
        else:
            if self.global_rank == 0:
                print(">>> Loading default VAE from 'runwayml/stable-diffusion-v1-5'...", flush=True)
            self.vae = AutoencoderKL.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="vae", low_cpu_mem_usage=True)

        # Initialize Denoising Backbone: PerceiverIO-DiT or UNet
        backbone_type = args.get('backbone', 'perceiver_dit')
        if backbone_type == 'perceiver_dit':
            if self.global_rank == 0:
                print(">>> Initializing PerceiverIO-DiT fusion model...", flush=True)
            dit_cfg = args.get('dit', {})
            perceiver_cfg = args.get('perceiver', {})
            self.unet = PerceiverIODiT(
                sample_size=dit_cfg.get('sample_size', 48),
                patch_size=dit_cfg.get('patch_size', 2),
                in_channels=4,
                out_channels=4,
                hidden_size=dit_cfg.get('hidden_size', 768),
                depth=dit_cfg.get('depth', 12),
                num_heads=dit_cfg.get('num_heads', 12),
                mlp_ratio=dit_cfg.get('mlp_ratio', 4.0),
                dropout=dit_cfg.get('dropout', 0.0),
                use_conv_stem=dit_cfg.get('use_conv_stem', True),
                overlapping_patch=dit_cfg.get('overlapping_patch', True),
                perceiver_num_latents=perceiver_cfg.get('num_latents', 64),
                perceiver_latent_dim=perceiver_cfg.get('latent_dim', 768),
                perceiver_depth=perceiver_cfg.get('depth', 2),
                perceiver_num_heads=perceiver_cfg.get('num_heads', 8),
                context_dim=768,
            )
        else:
            if self.global_rank == 0:
                print(">>> Loading UNet from local cache...", flush=True)
            self.unet = UNet2DConditionModel.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="unet", low_cpu_mem_usage=True)

        # Optionally transfer pre-trained ViT-B weights into DiT-B transformer blocks.
        # This gives the model ImageNet visual priors from day 1 without adding parameters.
        # Only applicable when backbone is perceiver_dit and dit.init_from_vit is True.
        if backbone_type == 'perceiver_dit' and dit_cfg.get('init_from_vit', False):
            if self.global_rank == 0:
                vit_name = dit_cfg.get('vit_model_name', 'vit_base_patch16_224')
                print(f">>> ViT-B initialization enabled ('{vit_name}'). Transferring weights...", flush=True)
            self.unet.load_pretrained_vit_b(
                vit_model_name=dit_cfg.get('vit_model_name', 'vit_base_patch16_224')
            )

        if self.global_rank == 0:
            print(">>> Model components loaded.", flush=True)

        # Force garbage collection and empty CUDA/MPS caches to free GPU memory
        import gc
        gc.collect()
        if self.device.type == 'mps':
            torch.mps.empty_cache()
        elif self.device.type == 'cuda':
            torch.cuda.empty_cache()
            
        # The noise scheduler manages the forward diffusion process (adding noise) during training.
        self.noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                             num_train_timesteps=1000)
        # Trainable parameters (unet, embeddings) are kept in FP32 master precision;
        # PyTorch Lightning's precision='16-mixed' automatically handles FP16 autocast during forward passes.

        # Enable gradient checkpointing to keep activation memory strictly under control on 16GB Macs
        self.unet.enable_gradient_checkpointing()

        # Freeze components that are not being trained.
        # By default, VAE is frozen. Optionally freeze the backbone body or cross-attention maps.
        self.freeze_params(self.vae.parameters())
        if args.freeze_resnet:
            # Freeze backbone parameters, keeping cross-attention and Perceiver layers trainable.
            self.freeze_params(self.unet.parameters())
            for name, param in self.unet.named_parameters():
                if "attn2" in name or "cross_attn" in name or "perceiver" in name:
                    param.requires_grad = True

        if args.freeze_blip and hasattr(self, "mm_encoder"):
            self.freeze_params(self.mm_encoder.parameters())

        if args.freeze_clip and hasattr(self, "text_encoder"):
            self.freeze_params(self.text_encoder.parameters())

        # Apple Silicon Mac support: apply FP16 to frozen encoders too, keeping token embeddings in FP32
        if self.args.get('accelerator') == 'mps':
            self.vae.half()
            if args.freeze_blip and hasattr(self, "mm_encoder"):
                self.mm_encoder.half()
                self.mm_encoder.text_encoder.embeddings.word_embeddings.float()
            if args.freeze_clip and hasattr(self, "text_encoder"):
                self.text_encoder.half()
                if hasattr(self.text_encoder, "embeddings"):
                    self.text_encoder.embeddings.token_embedding.float()
                elif hasattr(self.text_encoder, "text_model"):
                    self.text_encoder.text_model.embeddings.token_embedding.float()
        print(">>> Model initialized.")

    @staticmethod
    def freeze_params(params):
        for param in params:
            param.requires_grad = False

    @staticmethod
    def unfreeze_params(params):
        for param in params:
            param.requires_grad = True

    def configure_optimizers(self):
        # Decoupled parameter groups with differential learning rates:
        # 1. Deep Positional Gates: ~1e-4 (rapid convergence to anchor coordinates across 12 layers)
        # 2. Pre/Post-Conv Stems & Heads: ~2.5e-5 (needs larger LR to learn sharp edge filters)
        # 3. Perceiver IO Resampler: ~3e-6 (stable multi-modal context adaptation)
        # 4. DiT Transformer Backbone: ~1e-6 (gentle LR to prevent contrast/canopy blowout)
        pos_gate_params = []
        conv_params = []
        perceiver_params = []
        backbone_params = []

        base_lr = float(self.args.init_lr)
        pos_gate_lr = float(self.args.get('pos_gate_lr', 1e-4))
        conv_lr = float(self.args.get('conv_lr', 2.5e-5))
        perceiver_lr = float(self.args.get('perceiver_lr', 3e-6))
        backbone_lr = float(self.args.get('backbone_lr', 1e-6))

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'pos_gate' in name:
                pos_gate_params.append(param)
            elif 'pre_conv' in name or 'post_conv' in name:
                conv_params.append(param)
            elif 'perceiver' in name:
                perceiver_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = [
            {'params': pos_gate_params, 'lr': pos_gate_lr, 'weight_decay': 0.0, 'name': 'pos_gate_anchors'},
            {'params': conv_params, 'lr': conv_lr, 'weight_decay': 1e-4, 'name': 'conv_stem_head'},
            {'params': perceiver_params, 'lr': perceiver_lr, 'weight_decay': 1e-4, 'name': 'perceiver_io'},
            {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': 1e-4, 'name': 'dit_backbone'},
        ]
        param_groups = [g for g in param_groups if len(g['params']) > 0]

        if self.global_rank == 0:
            print("\n" + "=" * 55)
            print(">>> DECOUPLED DIFFERENTIAL LEARNING RATES:")
            print(f"  • Pos Gate Anchors ({len(pos_gate_params)} tensors): lr = {pos_gate_lr:.2e}")
            print(f"  • Conv Stem & Head ({len(conv_params)} tensors): lr = {conv_lr:.2e}")
            print(f"  • Perceiver IO     ({len(perceiver_params)} tensors): lr = {perceiver_lr:.2e}")
            print(f"  • DiT Backbone     ({len(backbone_params)} tensors): lr = {backbone_lr:.2e}")
            print("=" * 55 + "\n", flush=True)

        if self.args.get('strategy') == 'deepspeed' and self.args.get('accelerator') == 'gpu':
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            optimizer = DeepSpeedCPUAdam(param_groups, weight_decay=1e-4)
        else:
            optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

        #CosineannealingLR scheduler - smoothly decays LR from init_lr toward 0 over all training steps.
        #scheduler = CosineAnnealingLR(optimizer, T_max=self.args.max_epochs * self.steps_per_epoch)

        # 2-epoch linear warmup to stabilize cross-attention projection maps
        warmup_epochs = 2
        warmup_steps = warmup_epochs * self.steps_per_epoch
        
        # Track the starting step at the moment of resumption
        self.start_step = None

        def lr_lambda(global_step):
            # Capture the very first step seen during this execution run
            if self.start_step is None:
                self.start_step = global_step
            
            # Calculate steps relative to when we hit "run"
            relative_step = global_step - self.start_step

            # 1. Active Warmup Phase relative to resumption
            if relative_step < warmup_steps:
                return float(relative_step) / float(max(1, warmup_steps))
            
            # 2. Constant Recovery Phase
            return 1.0

        from torch.optim.lr_scheduler import LambdaLR
        scheduler = LambdaLR(optimizer, lr_lambda)

        optim_dict = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }
        return optim_dict

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """
        Intercepts checkpoint loading to:
        1. Reset optimizer states if resuming an old 1-group checkpoint into our new
           3-group decoupled architecture (prevents ValueError: different number of parameter groups).
        2. Prevent KeyError caused by scheduler state changes mid-training.
        """
        # 1. Handle mismatched optimizer parameter groups or tensor counts
        if "optimizer_states" in checkpoint and checkpoint["optimizer_states"]:
            saved_param_groups = checkpoint["optimizer_states"][0].get("param_groups", [])
            saved_sizes = [len(g.get("params", [])) for g in saved_param_groups]

            # Count expected trainable parameters per decoupled group
            pos_gate_count = sum(1 for n, p in self.named_parameters() if p.requires_grad and 'pos_gate' in n)
            conv_count = sum(1 for n, p in self.named_parameters() if p.requires_grad and ('pre_conv' in n or 'post_conv' in n))
            perceiver_count = sum(1 for n, p in self.named_parameters() if p.requires_grad and 'perceiver' in n)
            backbone_count = sum(1 for n, p in self.named_parameters() if p.requires_grad and not ('pre_conv' in n or 'post_conv' in n or 'pos_gate' in n or 'perceiver' in n))
            current_sizes = [c for c in [pos_gate_count, conv_count, perceiver_count, backbone_count] if c > 0]

            if saved_sizes != current_sizes:
                print(f"\n>>> Checkpoint Intercept: Parameter group architecture updated since checkpoint was saved.")
                print(f"    Checkpoint group sizes: {saved_sizes} vs Current model group sizes: {current_sizes}")
                print(f">>> Resetting optimizer and scheduler states to initialize fresh decoupled learning rates (PosGate: {self.args.get('pos_gate_lr', 1e-4)}, Conv: {self.args.get('conv_lr', 2.5e-5)}, Perceiver: 3e-6, Backbone: 1e-6).\n", flush=True)
                checkpoint["optimizer_states"] = []
                checkpoint["lr_schedulers"] = []

        # 2. Resample patch_embed.proj.weight from 3x3 to 2x2 if switching to symmetric non-overlapping patches
        if "state_dict" in checkpoint and "unet.patch_embed.proj.weight" in checkpoint["state_dict"]:
            w = checkpoint["state_dict"]["unet.patch_embed.proj.weight"]
            if hasattr(self, 'unet') and hasattr(self.unet, 'patch_embed'):
                target_shape = self.unet.patch_embed.proj.weight.shape
                if w.shape != target_shape and w.shape[-2:] == (3, 3) and target_shape[-2:] == (2, 2):
                    print(f"\n>>> Checkpoint Intercept: Resampling patch_embed.proj.weight from 3x3 to 2x2 with variance preservation...", flush=True)
                    w_resampled = torch.nn.functional.interpolate(w, size=(2, 2), mode='bilinear', align_corners=False)
                    scale = (w.std() / (w_resampled.std() + 1e-8)).item()
                    checkpoint["state_dict"]["unet.patch_embed.proj.weight"] = w_resampled * scale
                    print(">>> Resetting optimizer states and pos_gate parameters for new 2x2 projection geometry.\n", flush=True)
                    checkpoint["optimizer_states"] = []
                    checkpoint["lr_schedulers"] = []
                    for i in range(12):
                        checkpoint["state_dict"][f"unet.blocks.{i}.pos_gate"] = torch.zeros(1)

        # 3. Patch missing or negative pos_gate parameters from older checkpoints so strict loading succeeds
        if "state_dict" in checkpoint:
            patched = 0
            for i in range(12):
                key = f"unet.blocks.{i}.pos_gate"
                if key not in checkpoint["state_dict"] or checkpoint["state_dict"][key].item() < 0.0:
                    checkpoint["state_dict"][key] = torch.zeros(1)
                    patched += 1
            if patched > 0:
                print(f">>> Checkpoint intercept: Re-anchored {patched} pos_gate parameters to exact 0.0 identity.", flush=True)

        # 4. Refresh optimizer and scheduler states when launching new fine-tuning stage
        if "optimizer_states" in checkpoint:
            checkpoint["optimizer_states"] = []
            checkpoint["lr_schedulers"] = []
            print(f">>> Checkpoint intercept: Initializing fresh optimizer momentums with updated learning rates (Backbone: {self.args.get('backbone_lr', 1e-6):.1e}, EdgeWeight: {self.args.get('edge_loss_weight', 0.25)}).\n", flush=True)

    def load_state_dict(self, state_dict, strict=True):
        """Auto-patch missing pos_gate parameters and resample 3x3 patch weights to allow seamless resumption."""
        for i in range(12):
            key = f"unet.blocks.{i}.pos_gate"
            if key not in state_dict:
                state_dict[key] = torch.zeros(1)
        if "unet.patch_embed.proj.weight" in state_dict:
            w = state_dict["unet.patch_embed.proj.weight"]
            if hasattr(self, 'unet') and hasattr(self.unet, 'patch_embed'):
                target_shape = self.unet.patch_embed.proj.weight.shape
                if w.shape != target_shape and w.shape[-2:] == (3, 3) and target_shape[-2:] == (2, 2):
                    print(f">>> Resampling patch_embed.proj.weight from 3x3 to 2x2 with variance preservation...", flush=True)
                    w_resampled = torch.nn.functional.interpolate(w, size=(2, 2), mode='bilinear', align_corners=False)
                    scale = (w.std() / (w_resampled.std() + 1e-8)).item()
                    state_dict["unet.patch_embed.proj.weight"] = w_resampled * scale
                    for i in range(12):
                        state_dict[f"unet.blocks.{i}.pos_gate"] = torch.zeros(1)
        return super().load_state_dict(state_dict, strict=False)


    def forward(self, batch):
        # Set frozen feature encoders to evaluation mode to disable dropout/norm updates
        if self.args.freeze_clip and hasattr(self, "text_encoder"):
            self.text_encoder.eval()
        if self.args.freeze_blip and hasattr(self, "mm_encoder"):
            self.mm_encoder.eval()
            
        # Unpack the PyTorch dataloader batch
        images, captions, attention_mask, source_images, source_caption, source_attention_mask = batch
        B, V, S = captions.shape
        src_V = V + 1 if self.task == 'continuation' else V
        
        # Flatten batch (B) and sequence (V / src_V) dimensions to process all frames as a single batch
        images = torch.flatten(images, 0, 1)
        captions = torch.flatten(captions, 0, 1)
        attention_mask = torch.flatten(attention_mask, 0, 1)
        source_images = torch.flatten(source_images, 0, 1)
        source_caption = torch.flatten(source_caption, 0, 1)
        source_attention_mask = torch.flatten(source_attention_mask, 0, 1)

        # Classifier-free guidance training: randomly drop conditioning at a 10% rate
        classifier_free_idx = np.random.rand(B * V) < 0.1

        # 1. Encode text via CLIP
        caption_embeddings = self.text_encoder(captions, attention_mask).last_hidden_state
        
        # 2. Encode source/context frame information via BLIP (multimodal)
        source_embeddings = self.mm_encoder(source_images, source_caption, source_attention_mask,
                                            mode='multimodal').reshape(B, src_V * S, -1)
        source_embeddings = source_embeddings.repeat_interleave(V, dim=0)
        
        # Replace dropped indices with null token embeddings (unconditional branch)
        caption_embeddings[classifier_free_idx] = \
            self.text_encoder(self.clip_text_null_token).last_hidden_state[0]
        source_embeddings[classifier_free_idx] = \
            self.mm_encoder(self.blip_image_null_token, self.blip_text_null_token, attention_mask=None,
                            mode='multimodal')[0].repeat(src_V, 1)
                            
        # Apply modal-type embeddings to help UNet differentiate CLIP text from BLIP multimodal tokens
        caption_embeddings += self.modal_type_embeddings(torch.tensor(0, device=self.device))
        source_embeddings += self.modal_type_embeddings(torch.tensor(1, device=self.device))
        
        # Add relative temporal (time) embeddings to the context features
        source_embeddings += self.time_embeddings(
            torch.arange(src_V, device=self.device).repeat_interleave(S, dim=0))
            
        # Concatenate text embeddings and past multimodal embeddings into a single sequence
        encoder_hidden_states = torch.cat([caption_embeddings, source_embeddings], dim=1)

        # Build attention masks for the concatenated sequence
        attention_mask = torch.cat(
            [attention_mask, source_attention_mask.reshape(B, src_V * S).repeat_interleave(V, dim=0)], dim=1)
        attention_mask = ~(attention_mask.bool())
        attention_mask[classifier_free_idx] = False

        # Apply causal/triangular masking so that the model doesn't look at future frames during generation.
        # Future frame states are masked out to maintain chronological causality.
        square_mask = torch.triu(torch.ones((V, V), device=self.device)).bool()
        square_mask = square_mask.unsqueeze(0).unsqueeze(-1).expand(B, V, V, S)
        square_mask = square_mask.reshape(B * V, V * S)
        attention_mask[:, -V * S:] = torch.logical_or(square_mask, attention_mask[:, -V * S:])

        # Reshape images tensor if batch dimension is missing or single
        if images.ndim == 4:
            images = images.unsqueeze(0)
            
        b, s, c, h, w = images.shape
        images_flat = images.reshape(-1, c, h, w)

        # =========================================================================
        # VAE ENCODING:
        # - Fast Batched Pass (Default for DiT): Encodes all frames simultaneously on GPU.
        # - Sequential Loop (Fallback for heavy UNet on low VRAM):
        #   If you switch back to UNet and encounter GPU OOM during VAE encoding,
        #   comment out the batched call and uncomment the sequential loop below:
        #
        #   latents_list = []
        #   for i in range(images_flat.shape[0]):
        #       img_single = images_flat[i:i+1]
        #       l = self.vae.encode(img_single).latent_dist.sample()
        #       latents_list.append(l)
        #   latents = torch.cat(latents_list, dim=0)
        # =========================================================================
        with torch.no_grad():
            latents = self.vae.encode(images_flat).latent_dist.sample()
            latents = latents.reshape(b, s, -1, h // 8, w // 8)
        # Apply Stable Diffusion scaling factor for latents
        latents = latents * 0.18215

        # Denoising Diffusion training step:
        # 1. Sample random Gaussian noise
        noise = torch.randn(latents.shape, device=self.device, dtype=latents.dtype)
        b, s, c, h, w = latents.shape
        latents_flat = latents.reshape(-1, c, h, w)
        noise_flat = noise.reshape(-1, c, h, w)
        
        # 2. Sample random timesteps for each image in the batch
        bsz_flat = latents_flat.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz_flat,), device=self.device).long()
        
        # 3. Add noise to the latents according to the schedule
        noisy_latents_flat = self.noise_scheduler.add_noise(latents_flat, noise_flat, timesteps)

        # 4. Predict the noise using the backbone and compute Min-SNR-gamma weighted MSE loss
        noise_pred = self.unet(noisy_latents_flat, timesteps, encoder_hidden_states, attention_mask).sample
        loss = F.mse_loss(noise_pred.float(), noise_flat.float(), reduction="none").mean([1, 2, 3])

        # Min-SNR-gamma weighting (Hang et al., ICCV 2023) to balance coarse layout vs fine edge sharpness
        snr = compute_snr(self.noise_scheduler, timesteps)
        snr_gamma = 5.0
        mse_loss_weights = torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
        loss = (loss * mse_loss_weights).mean()

        # 5. Differentiable Sobel Edge Loss on low-to-mid noise timesteps (t <= 600)
        # Directly penalizes wavy/sheared walls and forces straight architectural boundaries
        edge_mask = (timesteps <= 600)
        if edge_mask.any():
            sub_t = timesteps[edge_mask]
            sub_noisy = noisy_latents_flat[edge_mask]
            sub_pred_eps = noise_pred[edge_mask]
            sub_target_z0 = latents_flat[edge_mask]

            alphas = self.noise_scheduler.alphas_cumprod.to(self.device)
            sqrt_alpha = (alphas[sub_t] ** 0.5).view(-1, 1, 1, 1)
            sqrt_one_minus_alpha = ((1.0 - alphas[sub_t]) ** 0.5).view(-1, 1, 1, 1)
            pred_z0 = (sub_noisy - sqrt_one_minus_alpha * sub_pred_eps) / sqrt_alpha

            sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=self.device).view(1, 1, 3, 3).repeat(4, 1, 1, 1)
            sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=self.device).view(1, 1, 3, 3).repeat(4, 1, 1, 1)

            grad_pred_x = F.conv2d(pred_z0.float(), sobel_x, padding=1, groups=4)
            grad_pred_y = F.conv2d(pred_z0.float(), sobel_y, padding=1, groups=4)
            grad_tgt_x = F.conv2d(sub_target_z0.float(), sobel_x, padding=1, groups=4)
            grad_tgt_y = F.conv2d(sub_target_z0.float(), sobel_y, padding=1, groups=4)

            edge_loss = 0.5 * (F.l1_loss(grad_pred_x, grad_tgt_x) + F.l1_loss(grad_pred_y, grad_tgt_y))
            edge_weight = float(self.args.get('edge_loss_weight', 0.15))
            loss = loss + edge_weight * edge_loss

        return loss

    def sample(self, batch):
        """
        Runs auto-regressive inference on a batch of stories.
        Generates frames sequentially (one-by-one) using past frames decoded from
        the model as conditioning context for the next frame.
        """
        original_images, captions, attention_mask, source_images, source_caption, source_attention_mask = batch
        B, V, S = captions.shape
        src_V = V + 1 if self.task == 'continuation' else V
        original_images = torch.flatten(original_images, 0, 1)
        captions = torch.flatten(captions, 0, 1)
        attention_mask = torch.flatten(attention_mask, 0, 1)
        source_images = torch.flatten(source_images, 0, 1)
        source_caption = torch.flatten(source_caption, 0, 1)
        source_attention_mask = torch.flatten(source_attention_mask, 0, 1)

        # 1. Encode text via CLIP & source contexts via BLIP
        caption_embeddings = self.text_encoder(captions, attention_mask).last_hidden_state
        source_embeddings = self.mm_encoder(source_images, source_caption, source_attention_mask,
                                            mode='multimodal').reshape(B, src_V * S, -1)
        
        # 2. Add modality and temporal embeddings
        caption_embeddings += self.modal_type_embeddings(torch.tensor(0, device=self.device))
        source_embeddings += self.modal_type_embeddings(torch.tensor(1, device=self.device))
        source_embeddings += self.time_embeddings(
            torch.arange(src_V, device=self.device).repeat_interleave(S, dim=0))
        source_embeddings = source_embeddings.repeat_interleave(V, dim=0)
        
        # Concatenate CLIP + BLIP embeddings for conditional generation
        encoder_hidden_states = torch.cat([caption_embeddings, source_embeddings], dim=1)

        # 3. Create conditional attention mask with causal (triangular) masking
        attention_mask = torch.cat(
            [attention_mask, source_attention_mask.reshape(B, src_V * S).repeat_interleave(V, dim=0)], dim=1)
        attention_mask = ~(attention_mask.bool())
        square_mask = torch.triu(torch.ones((V, V), device=self.device)).bool()
        square_mask = square_mask.unsqueeze(0).unsqueeze(-1).expand(B, V, V, S)
        square_mask = square_mask.reshape(B * V, V * S)
        attention_mask[:, -V * S:] = torch.logical_or(square_mask, attention_mask[:, -V * S:])

        # 4. Create unconditional embeddings for Classifier-Free Guidance (CFG)
        uncond_caption_embeddings = self.text_encoder(self.clip_text_null_token).last_hidden_state
        uncond_source_embeddings = self.mm_encoder(self.blip_image_null_token, self.blip_text_null_token,
                                                   attention_mask=None, mode='multimodal').repeat(1, src_V, 1)
        uncond_caption_embeddings += self.modal_type_embeddings(torch.tensor(0, device=self.device))
        uncond_source_embeddings += self.modal_type_embeddings(torch.tensor(1, device=self.device))
        uncond_source_embeddings += self.time_embeddings(
            torch.arange(src_V, device=self.device).repeat_interleave(S, dim=0))
        uncond_embeddings = torch.cat([uncond_caption_embeddings, uncond_source_embeddings], dim=1)
        uncond_embeddings = uncond_embeddings.expand(B * V, -1, -1)

        # Concatenate unconditional and conditional representations to batch-process both in a single UNet pass
        encoder_hidden_states = torch.cat([uncond_embeddings, encoder_hidden_states])
        uncond_attention_mask = torch.zeros((B * V, (src_V + 1) * S), device=self.device).bool()
        uncond_attention_mask[:, -V * S:] = square_mask
        attention_mask = torch.cat([uncond_attention_mask, attention_mask], dim=0)

        # Reshape attention mask to separate unconditional and conditional branches
        attention_mask = attention_mask.reshape(2, B, V, (src_V + 1) * S)
        images = list()
        
        # Autoregressive generation loop: generate each frame sequential step V
        for i in range(V):
            encoder_hidden_states = encoder_hidden_states.reshape(2, B, V, (src_V + 1) * S, -1)
            decoded_caption = self.clip_tokenizer.decode(captions[i], skip_special_tokens=True)
            print(f"\n>>> [Sequence {i+1}] Generating image for prompt: \"{decoded_caption}\"", flush=True)

            # Denoise latents via diffusion loop
            sample_size = self.args.get('dit', {}).get('sample_size', 64) if self.args.get('backbone') == 'perceiver_dit' else 48
            img_size = sample_size * 8
            new_image = self.diffusion(encoder_hidden_states[:, :, i].reshape(2 * B, (src_V + 1) * S, -1),
                                       attention_mask[:, :, i].reshape(2 * B, (src_V + 1) * S),
                                       img_size, img_size, self.args.num_inference_steps, self.args.guidance_scale, 0.0)
            
            # Post-process for saved output only (does gamma correction/contrast stretching)
            images += self.post_process(new_image)

            # Auto-regressively feed the newly generated raw image back to BLIP
            # to serve as context for generating the subsequent frames.
            new_image_raw = new_image
            new_image = torch.stack([self.blip_image_processor(im) for im in new_image_raw]).to(self.device)
            new_embedding = self.mm_encoder(new_image, 
                                            source_caption.reshape(B, src_V, S)[:, i + src_V - V],
                                            source_attention_mask.reshape(B, src_V, S)[:, i + src_V - V],
                                            mode='multimodal')
            new_embedding = new_embedding.repeat_interleave(V, dim=0)
            new_embedding += self.modal_type_embeddings(torch.tensor(1, device=self.device))
            new_embedding += self.time_embeddings(torch.tensor(i + src_V - V, device=self.device))

            # Update the conditional branch embeddings with the new BLIP features
            encoder_hidden_states = encoder_hidden_states[1].reshape(B * V, (src_V + 1) * S, -1)
            encoder_hidden_states[:, (i + 1 + src_V - V) * S:(i + 2 + src_V - V) * S] = new_embedding
            encoder_hidden_states = torch.cat([uncond_embeddings, encoder_hidden_states])

        return original_images, images

    def training_step(self, batch, batch_idx):
        loss = self(batch)
        if self.global_rank == 0:
            print(f">>> [Epoch {self.current_epoch}][Batch {batch_idx}] Loss: {loss.item():.4f}", flush=True)
        self.log('loss/train_loss', loss.detach(), on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)
        # Clear inactive Metal allocations every 16 batches to prevent memory pressure spikes
        if (batch_idx + 1) % 16 == 0 and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('loss/val_loss', loss.detach(), on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

    def on_train_epoch_end(self):
        """Force garbage collection and empty MPS cache at the end of each training epoch."""
        import gc
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def on_validation_epoch_end(self):
        """Force garbage collection and empty MPS cache at the end of each validation epoch."""
        import gc
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if getattr(self, "global_rank", 0) == 0:
            print(f">>> [Sampling] Generating images for story {batch_idx}...", flush=True)
        original_images, images = self.sample(batch)
        if self.args.calculate_fid:
            original_images = original_images.cpu().numpy().astype('uint8')
            original_images = [Image.fromarray(im, 'RGB') for im in original_images]
            ori = self.inception_feature(original_images).cpu().numpy()
            gen = self.inception_feature(images).cpu().numpy()
        else:
            ori = None
            gen = None
        return images, ori, gen

    def diffusion(self, encoder_hidden_states, attention_mask, height, width, num_inference_steps, guidance_scale, eta):
        """
        Runs the iterative denoising diffusion loop (using DDIM or PNDM) to generate
        images from a starting Gaussian noise latent vector.
        """
        # 1. Initialize random latents (Gaussian noise)
        latents = torch.randn((encoder_hidden_states.shape[0] // 2, self.unet.in_channels, height // 8, width // 8),
                               device=self.device)

        # Handle different scheduler signatures for setting timesteps
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # LMSDiscreteScheduler requires scaling the initial noise by the standard deviation
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents * self.scheduler.sigmas[0]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # 2. Iteratively denoise latents
        for i, t in enumerate(self.scheduler.timesteps):
            # Duplicate latents for classifier-free guidance (uncond + cond)
            latent_model_input = torch.cat([latents] * 2)
            is_5d = latent_model_input.ndim == 5
            if is_5d:
                b_s_orig, c_orig, h_orig, w_orig = latent_model_input.shape[0], latent_model_input.shape[2], latent_model_input.shape[3], latent_model_input.shape[4]
                latent_model_input = latent_model_input.flatten(0, 1)

            # Predict the noise using UNet
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states, attention_mask).sample

            if is_5d:
                noise_pred = noise_pred.view(b_s_orig, -1, c_orig, h_orig, w_orig)

            # Perform classifier-free guidance interpolation
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # CFG Rescale (Lin et al. 2024 / SDXL standard) to prevent wobbly wall/edge distortion
            guidance_rescale = 0.7
            if guidance_rescale > 0.0 and guidance_scale > 1.0:
                std_text = noise_pred_text.std([1, 2, 3], keepdim=True)
                std_cfg = noise_pred.std([1, 2, 3], keepdim=True)
                noise_pred_rescaled = noise_pred * (std_text / (std_cfg + 1e-8))
                noise_pred = guidance_rescale * noise_pred_rescaled + (1.0 - guidance_rescale) * noise_pred
            
            # Step the scheduler to retrieve the cleaner latent
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        # 3. Decode clean latents back to RGB pixel space via VAE
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self.vae.decode(latents.to(self.vae.dtype)).sample.float()

        # Normalize pixel values back to [0, 1] range and transpose to NHWC channel order
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        # Return raw PIL images — post-processing is applied separately so
        # FID computation and BLIP auto-regressive conditioning use unmodified images.
        return self.numpy_to_pil(image)

    @staticmethod
    def post_process(pil_images):
        """
        Return the raw decoded VAE images directly.
        Percentile contrast stretching and gamma lifts are disabled to prevent
        amplifying dark nighttime shadow noise into fluorescent cyan/blue speckles.
        """
        return pil_images

    @staticmethod
    def numpy_to_pil(images):
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image, 'RGB') for image in images]
        return pil_images

    def inception_feature(self, images):
        images = torch.stack([self.fid_augment(image) for image in images])
        images = images.type(torch.FloatTensor).to(self.device)
        images = (images + 1) / 2
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        pred = self.inception(images)[0]
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
        return pred.reshape(-1, 2048)


def train(args: DictConfig) -> None:
    """
    Sets up the tokenizers, dataloaders, PyTorch Lightning callbacks, and trainer
    to perform fine-tuning/training of the ARLDM model.
    """
    clip_tokenizer = CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="tokenizer")
    from models.blip_override.blip import init_tokenizer
    blip_tokenizer = init_tokenizer()

    print(">>> Setting up Dataloader...")
    dataloader = LightningDataset(args, clip_tokenizer, blip_tokenizer)
    dataloader.setup('fit')
    print(">>> Dataloader ready.")

    print(">>> Initializing Lightning Module...")
    # Initialize the model with dynamic steps per epoch calculated from the dataloader length
    model = ARLDM(args, steps_per_epoch=dataloader.get_length_of_train_dataloader())
    print(">>> Lightning Module initialized.")

    # Configure TensorBoard logger and checkpoint settings
    logger = TensorBoardLogger(save_dir=os.path.join(args.ckpt_dir, args.run_name), name='log', default_hp_metric=False)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.ckpt_dir, args.run_name),
        every_n_epochs=args.get('save_every_n_epochs', 1),
        save_top_k=0,
        save_last=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callback_list = [lr_monitor, checkpoint_callback]

    print(">>> Setting up Lightning Strategy...")
    fsdp_strategy = 'auto'

    # Initialize PyTorch Lightning Trainer with mixed precision setting
    trainer = pl.Trainer(
        accelerator=args.get('accelerator', 'auto'),
        devices=args.get('devices', 1),
        strategy=fsdp_strategy,
        precision='bf16-mixed' if args.get('accelerator') == 'gpu' else '16-mixed',
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.get('accumulate_grad_batches', 4),
        limit_train_batches=args.get('limit_train_batches') or 1.0,
        limit_val_batches=args.get('limit_val_batches') or 1.0,
        benchmark=False,
        num_sanity_val_steps=0,
        logger=logger,
        log_every_n_steps=1,
        callbacks=callback_list
    )
    import sys
    sys.stderr.write(f"\n>>> TRAINER CKPT PATH: {args.train_model_file}\n")
    if args.train_model_file:
        sys.stderr.write(f">>> TRAINER CKPT EXISTS: {os.path.exists(args.train_model_file)}\n")
    sys.stderr.flush()

    # Wrap load_optimizer_state_dict with safety fallback to prevent crashes if parameter groups or sizes change
    orig_load_optimizer = trainer.strategy.load_optimizer_state_dict
    def safe_load_optimizer(checkpoint):
        try:
            orig_load_optimizer(checkpoint)
        except Exception as e:
            print(f"\n>>> [Checkpoint Connector] Optimizer state restoration safely bypassed ({e}).", flush=True)
            print(">>> Initializing fresh decoupled optimizer learning rates.\n", flush=True)
    trainer.strategy.load_optimizer_state_dict = safe_load_optimizer
    
    # Start the training process, optionally resuming from a checkpoint
    trainer.fit(model, dataloader, ckpt_path=args.train_model_file)


def sample(args: DictConfig) -> None:
    """
    Evaluates/samples generated stories using a trained checkpoint.
    Optionally computes Fréchet Inception Distance (FID) to measure generation quality.
    """
    assert args.gpu_ids == 1 or len(args.gpu_ids) == 1, "Only one GPU is supported in test mode"
    dataloader = LightningDataset(args)
    dataloader.setup('test')
    
    # Load model weights from a specific checkpoint or start from scratch if none specified
    if args.test_model_file:
        model = ARLDM.load_from_checkpoint(args.test_model_file, args=args, strict=False)
    else:
        model = ARLDM(args)

    # Ensure all modules (text_encoder, mm_encoder, vae, unet) have uniform float32 precision
    # to avoid Apple Silicon MPS assertion failures (MPSNDArrayMatrixMultiplication datatype mismatch).
    model.float()

    # Initialize trainer specifically for runtime predictions/inference
    predictor = pl.Trainer(
        accelerator=args.get('accelerator', 'auto'),
        devices=args.get('devices', 1),
        max_epochs=-1,
        limit_predict_batches=args.get('limit_predict_batches', 10),
        benchmark=True
    )
    predictions = predictor.predict(model, dataloader)
    print(f">>> Generation complete. Saving images to {args.sample_output_dir}...")
    
    if not os.path.exists(args.sample_output_dir):
        os.makedirs(args.sample_output_dir, exist_ok=True)

    # Save output images to the destination directory
    count = 0
    for batch_pred in predictions:
        story_images = batch_pred[0] 
        for img in story_images:
            if isinstance(img, np.ndarray):
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)
            
            img.save(os.path.join(args.sample_output_dir, f"sample_{count:04d}.png"))
            count += 1
    print(f">>> Successfully saved {count} images.")

    # Calculate and output FID metric comparing real images against generated ones
    if args.calculate_fid:
        ori = np.array([elem for sublist in predictions for elem in sublist[1]])
        gen = np.array([elem for sublist in predictions for elem in sublist[2]])
        fid = calculate_fid_given_features(ori, gen)
        print('FID: {}'.format(fid))


@hydra.main(config_path=".", config_name="config")
def main(args: DictConfig) -> None:
    pl.seed_everything(args.seed)
    if args.num_cpu_cores > 0:
        torch.set_num_threads(args.num_cpu_cores)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'sample':
        sample(args)


if __name__ == '__main__':
    main()
