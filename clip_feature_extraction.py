import os
import torch
import clip
from PIL import Image
import numpy as np
from natsort import natsorted

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)


def extract_clip_features_from_images(image_folder, window_size=16, save_dir=None):

    # Get all the .jpg files in the folder and sort them by natural order
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    # Sort files naturally by frame number
    image_files = natsorted(image_files)

    num_frames = len(image_files)
    # print(f"Total number of frames in {image_folder}: {num_frames}")

    # Store frame features for aggregation
    frame_features = []
    aggregated_features = []

    # Process each frame
    for frame_file in image_files:
        # Load the image and preprocess it for CLIP
        image_path = os.path.join(image_folder, frame_file)
        pil_image = Image.open(image_path).convert('RGB')
        image = preprocess(pil_image).unsqueeze(0).to(device)

        # Extract CLIP features
        with torch.no_grad():
            features = model.encode_image(image)

        # Save the features for aggregation
        frame_features.append(features.squeeze().cpu().numpy())

        # Aggregate frames with window_size
        if len(frame_features) == window_size:
            avg_features = np.mean(frame_features, axis=0)
            aggregated_features.append(avg_features)
            frame_features = []

    # Average all the leftover features
    if frame_features:
        avg_features = np.mean(frame_features, axis=0)
        aggregated_features.append(avg_features)

    aggregated_features = np.array(aggregated_features)

    # Check if the number of extracted features matches the number of frames before averaging
    expected_num_segments = (num_frames // window_size) + \
        (1 if num_frames % window_size != 0 else 0)
    if len(aggregated_features) != expected_num_segments:
        # print(f"Warning: Mismatch in frame count! Expected {expected_num_segments}, but {len(aggregated_features)} feature windows extracted.")
        pass

    # save the features
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        video_name = os.path.basename(image_folder)
        save_path = os.path.join(save_dir, f"{video_name}.npy")
        np.save(save_path, aggregated_features)
        # print(f"Saved features to {save_path}")

    return aggregated_features


def process_all_folders(image_root_dir, save_dir, window_size=16):
    for folder in os.listdir(image_root_dir):
        folder_path = os.path.join(image_root_dir, folder)
        if os.path.isdir(folder_path):
            extract_clip_features_from_images(
                folder_path, window_size=window_size, save_dir=save_dir)


# Change the save dir as needed - Change the image_root_dir if you are not using gpu system 7
image_root_dir = '/home/data/CHARADES/Charades_v1_rgb'
save_dir = '/home/msoundar/clip-features-extract/clip-features-l14'

process_all_folders(image_root_dir, save_dir, window_size=16)
