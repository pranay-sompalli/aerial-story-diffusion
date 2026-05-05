import inspect
import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0" 

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, LMSDiscreteScheduler, PNDMScheduler, DDIMScheduler
from omegaconf import DictConfig
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel

from fid_utils import calculate_fid_given_features
from models.blip_override.blip import blip_feature_extractor, init_tokenizer
from models.diffusers_override.unet_2d_condition import UNet2DConditionModel
from models.inception import InceptionV3


class LightningDataset(pl.LightningDataModule):
    def __init__(self, args: DictConfig, clip_tokenizer=None, blip_tokenizer=None):
        super(LightningDataset, self).__init__()
        self.kwargs = {
            "num_workers": args.get("num_workers", 0),
            "persistent_workers": args.get("num_workers", 0) > 0,
            "pin_memory": args.get("pin_memory", False)
        }
        self.args = args
        self.clip_tokenizer = clip_tokenizer
        self.blip_tokenizer = blip_tokenizer

    def setup(self, stage="fit"):
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


class ARLDM(pl.LightningModule):
    def __init__(self, args: DictConfig, steps_per_epoch=1):
        super(ARLDM, self).__init__()
        if getattr(self, "global_rank", 0) == 0:
            print(f">>> Initializing ARLDM Model...", flush=True)
        self.args = args
        self.steps_per_epoch = steps_per_epoch
        self.task = args.task

        if args.mode == 'sample':
            if args.scheduler == "pndm":
                self.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                               skip_prk_steps=True)
            elif args.scheduler == "ddim":
                self.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                               clip_sample=False, set_alpha_to_one=True)
            else:
                raise ValueError("Scheduler not supported")
            self.fid_augment = transforms.Compose([
                transforms.Resize([64, 64]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
            self.inception = InceptionV3([block_idx])

        self.clip_tokenizer = CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="tokenizer")
        self.blip_tokenizer = init_tokenizer()
        self.blip_image_processor = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])
        self.max_length = args.get(args.dataset).max_length

        blip_image_null_token = self.blip_image_processor(
            Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))).unsqueeze(0).float()
        clip_text_null_token = self.clip_tokenizer([""], padding="max_length", max_length=self.max_length,
                                                   return_tensors="pt").input_ids
        blip_text_null_token = self.blip_tokenizer([""], padding="max_length", max_length=self.max_length,
                                                   return_tensors="pt").input_ids

        self.register_buffer('clip_text_null_token', clip_text_null_token)
        self.register_buffer('blip_text_null_token', blip_text_null_token)
        self.register_buffer('blip_image_null_token', blip_image_null_token)

        self.text_encoder = CLIPTextModel.from_pretrained('runwayml/stable-diffusion-v1-5',
                                                          subfolder="text_encoder")
        self.text_encoder.resize_token_embeddings(args.get(args.dataset).clip_embedding_tokens)
        
        old_embeddings = self.text_encoder.embeddings.position_embedding
        new_embeddings = self.text_encoder._get_resized_embeddings(old_embeddings, self.max_length)
        self.text_encoder.embeddings.position_embedding = new_embeddings
        self.text_encoder.config.max_position_embeddings = self.max_length
        self.text_encoder.max_position_embeddings = self.max_length
        self.text_encoder.embeddings.position_ids = torch.arange(self.max_length).expand((1, -1))

        self.modal_type_embeddings = nn.Embedding(2, 768)
        self.time_embeddings = nn.Embedding(5, 768)
        self.mm_encoder = blip_feature_extractor(
            pretrained='https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth',
            image_size=224, vit='base')
        self.mm_encoder.text_encoder.resize_token_embeddings(args.get(args.dataset).blip_embedding_tokens)

        if self.global_rank == 0:
            print(">>> Loading VAE and UNet from local cache...", flush=True)
        self.vae = AutoencoderKL.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="vae", low_cpu_mem_usage=True)
        self.unet = UNet2DConditionModel.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="unet", low_cpu_mem_usage=True)
        if self.global_rank == 0:
            print(">>> Model components loaded.", flush=True)

        import gc
        gc.collect()
        if self.device.type == 'mps':
            torch.mps.empty_cache()
        elif self.device.type == 'cuda':
            torch.cuda.empty_cache()
        self.noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                             num_train_timesteps=1000)
        
        if self.args.get('accelerator') == 'mps':
            self.unet.half()
            self.modal_type_embeddings.half()
            self.time_embeddings.half()

        self.unet.enable_gradient_checkpointing()

        self.freeze_params(self.vae.parameters())
        if args.freeze_resnet:
            self.freeze_params(self.unet.parameters())
            for name, param in self.unet.named_parameters():
                if "attn2" in name:
                    param.requires_grad = True

        if args.freeze_blip and hasattr(self, "mm_encoder"):
            self.freeze_params(self.mm_encoder.parameters())

        if args.freeze_clip and hasattr(self, "text_encoder"):
            self.freeze_params(self.text_encoder.parameters())

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
        if self.args.get('strategy') == 'deepspeed' and self.args.get('accelerator') == 'gpu':
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            optimizer = DeepSpeedCPUAdam(self.parameters(), lr=self.args.init_lr, weight_decay=1e-4)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.init_lr, weight_decay=1e-4)

        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.max_epochs * self.steps_per_epoch)
        optim_dict = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }
        return optim_dict

    def forward(self, batch):
        if self.args.freeze_clip and hasattr(self, "text_encoder"):
            self.text_encoder.eval()
        if self.args.freeze_blip and hasattr(self, "mm_encoder"):
            self.mm_encoder.eval()
        images, captions, attention_mask, source_images, source_caption, source_attention_mask = batch
        B, V, S = captions.shape
        src_V = V + 1 if self.task == 'continuation' else V
        images = torch.flatten(images, 0, 1)
        captions = torch.flatten(captions, 0, 1)
        attention_mask = torch.flatten(attention_mask, 0, 1)
        source_images = torch.flatten(source_images, 0, 1)
        source_caption = torch.flatten(source_caption, 0, 1)
        source_attention_mask = torch.flatten(source_attention_mask, 0, 1)

        classifier_free_idx = np.random.rand(B * V) < 0.1

        caption_embeddings = self.text_encoder(captions, attention_mask).last_hidden_state
        source_embeddings = self.mm_encoder(source_images, source_caption, source_attention_mask,
                                            mode='multimodal').reshape(B, src_V * S, -1)
        source_embeddings = source_embeddings.repeat_interleave(V, dim=0)
        caption_embeddings[classifier_free_idx] = \
            self.text_encoder(self.clip_text_null_token).last_hidden_state[0]
        source_embeddings[classifier_free_idx] = \
            self.mm_encoder(self.blip_image_null_token, self.blip_text_null_token, attention_mask=None,
                            mode='multimodal')[0].repeat(src_V, 1)
        caption_embeddings += self.modal_type_embeddings(torch.tensor(0, device=self.device))
        source_embeddings += self.modal_type_embeddings(torch.tensor(1, device=self.device))
        source_embeddings += self.time_embeddings(
            torch.arange(src_V, device=self.device).repeat_interleave(S, dim=0))
        encoder_hidden_states = torch.cat([caption_embeddings, source_embeddings], dim=1)

        attention_mask = torch.cat(
            [attention_mask, source_attention_mask.reshape(B, src_V * S).repeat_interleave(V, dim=0)], dim=1)
        attention_mask = ~(attention_mask.bool())
        attention_mask[classifier_free_idx] = False

        square_mask = torch.triu(torch.ones((V, V), device=self.device)).bool()
        square_mask = square_mask.unsqueeze(0).unsqueeze(-1).expand(B, V, V, S)
        square_mask = square_mask.reshape(B * V, V * S)
        attention_mask[:, -V * S:] = torch.logical_or(square_mask, attention_mask[:, -V * S:])

        if images.ndim == 4:
            images = images.unsqueeze(0)
            
        b, s, c, h, w = images.shape
        images_flat = images.reshape(-1, c, h, w)

        with torch.no_grad():
            latents_list = []
            for i in range(images_flat.shape[0]):
                img_single = images_flat[i:i+1]
                l = self.vae.encode(img_single).latent_dist.sample()
                latents_list.append(l)
            latents = torch.cat(latents_list, dim=0)
            latents = latents.reshape(b, s, -1, h // 8, w // 8)
        latents = latents * 0.18215

        noise = torch.randn(latents.shape, device=self.device, dtype=latents.dtype)
        b, s, c, h, w = latents.shape
        latents_flat = latents.reshape(-1, c, h, w)
        noise_flat = noise.reshape(-1, c, h, w)
        
        bsz_flat = latents_flat.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz_flat,), device=self.device).long()
        noisy_latents_flat = self.noise_scheduler.add_noise(latents_flat, noise_flat, timesteps)

        noise_pred = self.unet(noisy_latents_flat, timesteps, encoder_hidden_states, attention_mask).sample
        loss = F.mse_loss(noise_pred.float(), noise_flat.float(), reduction="none").mean([1, 2, 3]).mean()
        return loss

    def sample(self, batch):
        original_images, captions, attention_mask, source_images, source_caption, source_attention_mask = batch
        B, V, S = captions.shape
        src_V = V + 1 if self.task == 'continuation' else V
        original_images = torch.flatten(original_images, 0, 1)
        captions = torch.flatten(captions, 0, 1)
        attention_mask = torch.flatten(attention_mask, 0, 1)
        source_images = torch.flatten(source_images, 0, 1)
        source_caption = torch.flatten(source_caption, 0, 1)
        source_attention_mask = torch.flatten(source_attention_mask, 0, 1)

        caption_embeddings = self.text_encoder(captions, attention_mask).last_hidden_state
        source_embeddings = self.mm_encoder(source_images, source_caption, source_attention_mask,
                                            mode='multimodal').reshape(B, src_V * S, -1)
        caption_embeddings += self.modal_type_embeddings(torch.tensor(0, device=self.device))
        source_embeddings += self.modal_type_embeddings(torch.tensor(1, device=self.device))
        source_embeddings += self.time_embeddings(
            torch.arange(src_V, device=self.device).repeat_interleave(S, dim=0))
        source_embeddings = source_embeddings.repeat_interleave(V, dim=0)
        encoder_hidden_states = torch.cat([caption_embeddings, source_embeddings], dim=1)

        attention_mask = torch.cat(
            [attention_mask, source_attention_mask.reshape(B, src_V * S).repeat_interleave(V, dim=0)], dim=1)
        attention_mask = ~(attention_mask.bool())
        square_mask = torch.triu(torch.ones((V, V), device=self.device)).bool()
        square_mask = square_mask.unsqueeze(0).unsqueeze(-1).expand(B, V, V, S)
        square_mask = square_mask.reshape(B * V, V * S)
        attention_mask[:, -V * S:] = torch.logical_or(square_mask, attention_mask[:, -V * S:])

        uncond_caption_embeddings = self.text_encoder(self.clip_text_null_token).last_hidden_state
        uncond_source_embeddings = self.mm_encoder(self.blip_image_null_token, self.blip_text_null_token,
                                                   attention_mask=None, mode='multimodal').repeat(1, src_V, 1)
        uncond_caption_embeddings += self.modal_type_embeddings(torch.tensor(0, device=self.device))
        uncond_source_embeddings += self.modal_type_embeddings(torch.tensor(1, device=self.device))
        uncond_source_embeddings += self.time_embeddings(
            torch.arange(src_V, device=self.device).repeat_interleave(S, dim=0))
        uncond_embeddings = torch.cat([uncond_caption_embeddings, uncond_source_embeddings], dim=1)
        uncond_embeddings = uncond_embeddings.expand(B * V, -1, -1)

        encoder_hidden_states = torch.cat([uncond_embeddings, encoder_hidden_states])
        uncond_attention_mask = torch.zeros((B * V, (src_V + 1) * S), device=self.device).bool()
        uncond_attention_mask[:, -V * S:] = square_mask
        attention_mask = torch.cat([uncond_attention_mask, attention_mask], dim=0)

        attention_mask = attention_mask.reshape(2, B, V, (src_V + 1) * S)
        images = list()
        for i in range(V):
            encoder_hidden_states = encoder_hidden_states.reshape(2, B, V, (src_V + 1) * S, -1)
            decoded_caption = self.clip_tokenizer.decode(captions[0, i], skip_special_tokens=True)
            print(f"\n>>> [Sequence {i+1}] Generating image for prompt: \"{decoded_caption}\"", flush=True)

            new_image = self.diffusion(encoder_hidden_states[:, :, i].reshape(2 * B, (src_V + 1) * S, -1),
                                       attention_mask[:, :, i].reshape(2 * B, (src_V + 1) * S),
                                       384, 384, self.args.num_inference_steps, self.args.guidance_scale, 0.0)
            images += new_image

            new_image = torch.stack([self.blip_image_processor(im) for im in new_image]).to(self.device)
            new_embedding = self.mm_encoder(new_image, 
                                            source_caption.reshape(B, src_V, S)[:, i + src_V - V],
                                            source_attention_mask.reshape(B, src_V, S)[:, i + src_V - V],
                                            mode='multimodal')
            new_embedding = new_embedding.repeat_interleave(V, dim=0)
            new_embedding += self.modal_type_embeddings(torch.tensor(1, device=self.device))
            new_embedding += self.time_embeddings(torch.tensor(i + src_V - V, device=self.device))

            encoder_hidden_states = encoder_hidden_states[1].reshape(B * V, (src_V + 1) * S, -1)
            encoder_hidden_states[:, (i + 1 + src_V - V) * S:(i + 2 + src_V - V) * S] = new_embedding
            encoder_hidden_states = torch.cat([uncond_embeddings, encoder_hidden_states])

        return original_images, images

    def training_step(self, batch, batch_idx):
        loss = self(batch)
        if self.global_rank == 0:
            print(f">>> [Epoch {self.current_epoch}][Batch {batch_idx}] Loss: {loss.item():.4f}", flush=True)
        self.log('loss/train_loss', loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('loss/val_loss', loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

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
        latents = torch.randn((encoder_hidden_states.shape[0] // 2, self.unet.in_channels, height // 8, width // 8),
                              device=self.device)

        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents * self.scheduler.sigmas[0]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in enumerate(self.scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2)
            is_5d = latent_model_input.ndim == 5
            if is_5d:
                b_s_orig, c_orig, h_orig, w_orig = latent_model_input.shape[0], latent_model_input.shape[2], latent_model_input.shape[3], latent_model_input.shape[4]
                latent_model_input = latent_model_input.flatten(0, 1)

            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states, attention_mask).sample

            if is_5d:
                noise_pred = noise_pred.view(b_s_orig, -1, c_orig, h_orig, w_orig)

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self.vae.decode(latents.to(self.vae.dtype)).sample.float()

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        return self.numpy_to_pil(image)

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
    clip_tokenizer = CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="tokenizer")
    from models.blip_override.blip import init_tokenizer
    blip_tokenizer = init_tokenizer()

    print(">>> Setting up Dataloader...")
    dataloader = LightningDataset(args, clip_tokenizer, blip_tokenizer)
    dataloader.setup('fit')
    print(">>> Dataloader ready.")

    print(">>> Initializing Lightning Module...")
    model = ARLDM(args, steps_per_epoch=dataloader.get_length_of_train_dataloader())
    print(">>> Lightning Module initialized.")

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

    trainer = pl.Trainer(
        accelerator=args.get('accelerator', 'auto'),
        devices=args.get('devices', 1),
        strategy=fsdp_strategy,
        precision='bf16-mixed' if args.get('accelerator') == 'gpu' else '16-mixed',
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.get('accumulate_grad_batches', 4),
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
    
    trainer.fit(model, dataloader, ckpt_path=args.train_model_file)


def sample(args: DictConfig) -> None:
    assert args.gpu_ids == 1 or len(args.gpu_ids) == 1, "Only one GPU is supported in test mode"
    dataloader = LightningDataset(args)
    dataloader.setup('test')
    if args.test_model_file:
        model = ARLDM.load_from_checkpoint(args.test_model_file, args=args, strict=False)
    else:
        model = ARLDM(args)

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
