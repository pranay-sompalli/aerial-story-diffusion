import os
import cv2
import h5py
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

def main():
    # 1. Establish paths for raw image files and text descriptions (JSON)
    # Check if we are running in the cloud (Modal environment) or on a local Mac
    data_base = "/root/ARLDM/data" if os.path.exists("/root/ARLDM/data") else "."
    
    save_path = "visdrone.h5"
    gemini_json = os.path.join(data_base, "gemini.json")
    images_dir = os.path.join(data_base, "raw_images")
    
    if not os.path.exists(gemini_json):
        # Fallback local paths for development/testing
        gemini_json = "/Users/pranaysompalli/Documents/AeroDiffusion/text_descriptions/gemini.json"
        images_dir = "/Users/pranaysompalli/Documents/AeroDiffusion/VisDrone2019-DET-train/images"
    
    # Load all image description text
    with open(gemini_json, "r") as f:
        descriptions = json.load(f)
    
    # 2. Filter descriptions and verify that corresponding images exist locally
    valid_data = []
    for img_name, desc in descriptions.items():
        img_path = os.path.join(images_dir, img_name)
        if os.path.exists(img_path):
            valid_data.append((img_path, desc))
    
    print(f"Found {len(valid_data)} valid image-description pairs.")
    
    # 3. Seed and shuffle the data to ensure even distribution during splits
    import random
    random.seed(42)
    random.shuffle(valid_data)
    
    # 4. Standard 80/10/10 split for training, validation, and testing
    total = len(valid_data)
    train_end = int(0.8 * total)
    val_end = int(0.9 * total)
    
    splits = {
        'train': valid_data[:train_end],
        'val': valid_data[train_end:val_end],
        'test': valid_data[val_end:]
    }
    
    print(f"Splits: Train {len(splits['train'])}, Val {len(splits['val'])}, Test {len(splits['test'])}")
    
    # 5. Create HDF5 database to write processed splits
    f = h5py.File(save_path, "w")
    for subset, data in splits.items():
        length = len(data)
        group = f.create_group(subset)
        
        # We need to simulate a 5-frame sequence for the story model.
        # Initialize datasets for image0 ... image4 containing variable-length compressed byte arrays.
        images_dsets = []
        for i in range(5):
            images_dsets.append(
                group.create_dataset(f'image{i}', (length,), dtype=h5py.vlen_dtype(np.dtype('uint8'))))
        
        # Dataset to hold concatenated caption strings
        text_dset = group.create_dataset('text', (length,), dtype=h5py.string_dtype(encoding='utf-8'))
        
        # 6. Process and write each image-description pair to HDF5
        for i, (img_path, desc) in enumerate(tqdm(data, desc=f"Processing {subset}")):
            # Load the image, ensure it is in RGB, and resize to standard 512x512
            img = Image.open(img_path).convert('RGB').resize((512, 512))
            img_arr = np.array(img).astype(np.uint8)
            
            # Encode image as compressed PNG bytes to conserve HDF5 storage space
            img_bytes = cv2.imencode('.png', img_arr)[1].tobytes()
            img_buf = np.frombuffer(img_bytes, np.uint8)
            
            # Since VisDrone is originally a single-frame detection dataset, 
            # we duplicate the single image across all 5 frames to simulate a static sequence.
            for j in range(5):
                images_dsets[j][i] = img_buf
            
            # Clean and duplicate the text caption 5 times, joined by pipe characters '|'.
            clean_desc = desc.replace('\n', '').replace('\t', '').strip()
            txt_variations = [clean_desc for _ in range(5)]
            text_dset[i] = '|'.join(txt_variations)
            
    f.close()
    print(f"VisDrone dataset created at {save_path}")

if __name__ == '__main__':
    main()

