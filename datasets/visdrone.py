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

        # Transform applied to story frames during training/validation
        self.augment = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([384, 384]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.dataset = args.dataset
        self.max_length = args.get(args.dataset).max_length

        # CLIP tokenizer
        if clip_tokenizer is None:
            self.clip_tokenizer = CLIPTokenizer.from_pretrained(
                'runwayml/stable-diffusion-v1-5', subfolder="tokenizer"
            )
            added = self.clip_tokenizer.add_tokens(list(args.get(args.dataset).new_tokens))
            print(f"CLIP tokenizer: {added} new tokens added")
        else:
            self.clip_tokenizer = clip_tokenizer

        # BLIP tokenizer
        if blip_tokenizer is None:
            self.blip_tokenizer = init_tokenizer()
            added = self.blip_tokenizer.add_tokens(list(args.get(args.dataset).new_tokens))
            print(f"BLIP tokenizer: {added} new tokens added")
        else:
            self.blip_tokenizer = blip_tokenizer

        # BLIP image processor — fixed at 224x224 (BLIP model requirement)
        self.blip_image_processor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                 [0.26862954, 0.26130258, 0.27577711])
        ])

    def open_h5(self):
        """Lazily open the HDF5 file on first access (avoids pickling issues with multiprocessing)."""
        print(f"--- Opening HDF5 file: {self.h5_file} ---", flush=True)
        try:
            h5 = h5py.File(self.h5_file, "r")
            self.h5 = h5[self.subset]
            print(f"--- HDF5 opened successfully ---", flush=True)
        except Exception as e:
            print(f"--- ERROR opening HDF5: {e} ---", flush=True)
            raise

    def __getitem__(self, index: int):
        if not hasattr(self, 'h5'):
            self.open_h5()

        # Load and decode all 5 frames from JPEG bytes
        images = []
        for i in range(5):
            raw = self.h5[f'image{i}'][index]
            buf = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

        # BLIP source images (all 5 frames at 224x224)
        source_images: Tensor = torch.stack([self.blip_image_processor(im) for im in images])

        # For continuation mode, drop the first frame from the generation targets
        gen_images = images[1:] if self.args.task == 'continuation' else images
        if self.subset in ['train', 'val']:
            images_tensor: Tensor = torch.stack([self.augment(im) for im in gen_images])
        else:
            images_tensor = torch.from_numpy(np.array(gen_images))

        # Parse pipe-separated captions
        texts = self.h5['text'][index].decode('utf-8').split('|')

        # CLIP tokenization (generation targets)
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

        # BLIP tokenization (all 5 frames for source context)
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
        """Exclude the open HDF5 handle from pickling (for DataLoader workers)."""
        state = self.__dict__.copy()
        state.pop('h5', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __len__(self) -> int:
        if not hasattr(self, '_len'):
            with h5py.File(self.h5_file, "r") as h5:
                self._len = len(h5[self.subset]['text'])
        return self._len
