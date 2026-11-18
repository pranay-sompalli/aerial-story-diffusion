"""
datasets/visdrone.py
─────────────────────
PyTorch Dataset class for the VisDrone story-visualization task.

The dataset is stored as an HDF5 file with the following structure:
  /<split>/image0 ... image4   — JPEG-encoded bytes for each frame in the story
  /<split>/text                — pipe-separated captions for all 5 frames

Each story sequence consists of 5 frames and their associated captions.
In 'continuation' mode, the model is given the first frame as context and
asked to auto-regressively generate frames 1–4.
"""

import cv2
import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer

from models.blip_override.blip import init_tokenizer


class StoryDataset(Dataset):
    """
    VisDrone aerial image story dataset backed by an HDF5 file.

    Args:
        subset:         One of 'train', 'val', or 'test'.
        args:           Hydra/OmegaConf config object.
        clip_tokenizer: Optional pre-initialized CLIP tokenizer (shared across datasets).
        blip_tokenizer: Optional pre-initialized BLIP tokenizer (shared across datasets).
    """

    def __init__(self, subset: str, args, clip_tokenizer=None, blip_tokenizer=None):
        super(StoryDataset, self).__init__()
        self.args = args
        self.h5_file = args.get(args.dataset).hdf5_file
        self.subset = subset

        # Main image augmentation pipeline. Raw frames are resized to 512x512 for progressive fine-tuning.
        self.augment = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([512, 512]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.dataset = args.dataset
        self.max_length = args.get(args.dataset).max_length

        # Set up CLIP tokenizer: used for text conditioning during diffusion generation (Stable Diffusion).
        # We also register any custom/new tokens added specifically for our domain.
        if clip_tokenizer is None:
            self.clip_tokenizer = CLIPTokenizer.from_pretrained(
                'runwayml/stable-diffusion-v1-5', subfolder="tokenizer"
            )
            added = self.clip_tokenizer.add_tokens(list(args.get(args.dataset).new_tokens))
            print(f"CLIP tokenizer: {added} new tokens added")
        else:
            self.clip_tokenizer = clip_tokenizer

        # Set up BLIP tokenizer: used for contextual encoding of captions within the multimodal encoder.
        if blip_tokenizer is None:
            self.blip_tokenizer = init_tokenizer()
            added = self.blip_tokenizer.add_tokens(list(args.get(args.dataset).new_tokens))
            print(f"BLIP tokenizer: {added} new tokens added")
        else:
            self.blip_tokenizer = blip_tokenizer

        # BLIP image processor: prepares conditioning frames for the ViT encoder.
        # This requires 224x224 images normalized with ImageNet statistics.
        self.blip_image_processor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                 [0.26862954, 0.26130258, 0.27577711])
        ])

    def open_h5(self):
        """
        Lazily open the HDF5 file on the first access.
        This prevents file descriptors from being shared across multiprocessing
        workers, avoiding issues when PyTorch DataLoader uses num_workers > 0.
        """
        print(f"--- Opening HDF5 file: {self.h5_file} ---", flush=True)
        try:
            # Set rdcc_nbytes=1MB and rdcc_w0=0.0 to prevent HDF5 from caching gigabytes of raw chunks in RAM
            h5 = h5py.File(self.h5_file, "r", rdcc_nbytes=1024 * 1024, rdcc_w0=0.0)
            self.h5 = h5[self.subset]
            print(f"--- HDF5 opened successfully ---", flush=True)
        except Exception as e:
            print(f"--- ERROR opening HDF5: {e} ---", flush=True)
            raise

    def __getitem__(self, index: int):
        # Ensure the HDF5 file handle is opened for the current process/worker
        if not hasattr(self, 'h5'):
            self.open_h5()

        # Load and decode all 5 sequential frames of the story from compressed JPEG bytes
        images = []
        for i in range(5):
            raw = self.h5[f'image{i}'][index]
            buf = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # OpenCV uses BGR; convert to RGB
            images.append(img)

        # Prepare source conditioning images (all 5 frames) for the BLIP encoder at 224x224
        source_images: Tensor = torch.stack([self.blip_image_processor(im) for im in images])

        # Slice target images for generation. 
        # In 'continuation' mode, frame 0 is the starting context (given), and we only generate frames 1-4.
        # Otherwise, we generate all 5 frames.
        gen_images = images[1:] if self.args.task == 'continuation' else images
        if self.subset in ['train', 'val']:
            # Apply training augmentations (384x384 normalized tensors)
            images_tensor: Tensor = torch.stack([self.augment(im) for im in gen_images])
        else:
            # For testing/predicting, return raw numpy images for evaluation/visualization
            images_tensor = torch.from_numpy(np.array(gen_images))

        # Retrieve and parse the pipe-separated text captions for the 5 frames
        texts = self.h5['text'][index].decode('utf-8').split('|')

        # Tokenize target text prompts using the CLIP tokenizer.
        # In continuation mode, we discard the caption of the first (context) frame.
        clip_texts = texts[1:] if self.args.task == 'continuation' else texts
        clip_tok = self.clip_tokenizer(
            clip_texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        captions: Tensor = clip_tok['input_ids']
        attention_mask: Tensor = clip_tok['attention_mask']

        # Tokenize source/conditioning text prompts using the BLIP tokenizer.
        # Here we tokenize all 5 captions to serve as contextual history for the model.
        blip_tok = self.blip_tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        source_caption: Tensor = blip_tok['input_ids']
        source_attention_mask: Tensor = blip_tok['attention_mask']

        return images_tensor, captions, attention_mask, source_images, source_caption, source_attention_mask

    def __getstate__(self):
        """
        Exclude the open HDF5 file handle when pickling the dataset instance.
        This is necessary for PyTorch's multi-process DataLoader workers.
        """
        state = self.__dict__.copy()
        state.pop('h5', None)
        return state

    def __setstate__(self, state):
        # Restore state on workers; the file handle will be lazily re-opened
        self.__dict__.update(state)

    def __len__(self) -> int:
        # Cache the dataset length to avoid reading the file metadata repeatedly
        if not hasattr(self, '_len'):
            with h5py.File(self.h5_file, "r") as h5:
                self._len = len(h5[self.subset]['text'])
        return self._len

