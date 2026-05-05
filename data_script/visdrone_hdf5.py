import os
import cv2
import h5py
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

def main():
    # Use environment variables if on Modal, otherwise use local defaults
    data_base = "/root/ARLDM/data" if os.path.exists("/root/ARLDM/data") else "."
    
    save_path = "visdrone.h5"
    gemini_json = os.path.join(data_base, "gemini.json")
    images_dir = os.path.join(data_base, "raw_images")
    
    if not os.path.exists(gemini_json):
        # Fallback for local Mac if not in the root
        gemini_json = "/Users/pranaysompalli/Documents/AeroDiffusion/text_descriptions/gemini.json"
        images_dir = "/Users/pranaysompalli/Documents/AeroDiffusion/VisDrone2019-DET-train/images"
    
    with open(gemini_json, "r") as f:
        descriptions = json.load(f)
    
    # We will get all image filenames from the JSON, filter only those that exist
    valid_data = []
    for img_name, desc in descriptions.items():
        img_path = os.path.join(images_dir, img_name)
        if os.path.exists(img_path):
            valid_data.append((img_path, desc))
    
    print(f"Found {len(valid_data)} valid image-description pairs.")
    
    # Shuffle data (optional, but good for splits)
    np.random.seed(42)
    # We convert to a numpy array for shuffling? No, random.shuffle on list is better or just convert to array of tuples.
    # List of tuples is fine, but np.random.shuffle works on lists too.
    # Let's use Python's random instead of numpy to avoid 2D array string conversion issues.
    import random
    random.seed(42)
    random.shuffle(valid_data)
    
    # 80/10/10 split
    total = len(valid_data)
    train_end = int(0.8 * total)
    val_end = int(0.9 * total)
    
    splits = {
        'train': valid_data[:train_end],
        'val': valid_data[train_end:val_end],
        'test': valid_data[val_end:]
    }
    
    print(f"Splits: Train {len(splits['train'])}, Val {len(splits['val'])}, Test {len(splits['test'])}")
    
    f = h5py.File(save_path, "w")
    for subset, data in splits.items():
        length = len(data)
        group = f.create_group(subset)
        images_dsets = []
        for i in range(5):
            images_dsets.append(
                group.create_dataset(f'image{i}', (length,), dtype=h5py.vlen_dtype(np.dtype('uint8'))))
        text_dset = group.create_dataset('text', (length,), dtype=h5py.string_dtype(encoding='utf-8'))
        
        for i, (img_path, desc) in enumerate(tqdm(data, desc=f"Processing {subset}")):
            # Process Image
            img = Image.open(img_path).convert('RGB').resize((512, 512))
            img_arr = np.array(img).astype(np.uint8)
            img_bytes = cv2.imencode('.png', img_arr)[1].tobytes()
            img_buf = np.frombuffer(img_bytes, np.uint8)
            
            for j in range(5):
                images_dsets[j][i] = img_buf
            
            # Process Text
            # We only have 1 description per image, so we duplicate it 5 times separated by '|'
            clean_desc = desc.replace('\n', '').replace('\t', '').strip()
            txt_variations = [clean_desc for _ in range(5)]
            text_dset[i] = '|'.join(txt_variations)
            
    f.close()
    print(f"VisDrone dataset created at {save_path}")

if __name__ == '__main__':
    main()
