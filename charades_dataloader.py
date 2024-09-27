import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate
import numpy as np
import json
import os
from tqdm import tqdm
import random
from utils import *


def make_dataset(split_file, split, i3d_root, clip_root, num_classes=157):
    gamma = 0.5
    tau = 4
    ku = 1
    dataset = []

    with open(split_file, 'r') as f:
        data = json.load(f)
    print('split!!!', split)
    i = 0

    for vid in tqdm(data.keys()):
        if data[vid]['subset'] != split:
            continue

        i3d_path = os.path.join(i3d_root, vid + '.npy')
        clip_path = os.path.join(clip_root, vid + '.npy')

        if not os.path.exists(i3d_path) or not os.path.exists(clip_path):
            continue

        if len(data[vid]['actions']) < 1:
            continue

        fts_i3d = np.load(i3d_path)
        fts_clip = np.load(clip_path)
        num_feat_i3d = fts_i3d.shape[0]
        num_feat_clip = fts_clip.shape[0]

        label_i3d = np.zeros((num_feat_i3d, num_classes), np.float32)
        label_clip = np.zeros((num_feat_clip, num_classes), np.float32)
        hmap_i3d = np.zeros((num_feat_i3d, num_classes), np.float32)
        hmap_clip = np.zeros((num_feat_clip, num_classes), np.float32)
        action_lengths_i3d = []
        center_loc_i3d = []
        num_action_i3d = 0
        action_lengths_clip = []
        center_loc_clip = []
        num_action_clip = 0

        fps_i3d = num_feat_i3d / data[vid]['duration']
        fps_clip = num_feat_clip / data[vid]['duration']

        for ann in data[vid]['actions']:
            if ann[2] < ann[1]:
                continue

            mid_point = (ann[2] + ann[1]) / 2

            # Loop for I3D frames
            for fr in range(0, num_feat_i3d, 1):
                if fr / fps_i3d > ann[1] and fr / fps_i3d < ann[2]:
                    label_i3d[fr, ann[0]] = 1
                if (fr+1) / fps_i3d > mid_point and fr / fps_i3d < mid_point:
                    center = fr + 1
                    class_ = ann[0]
                    action_duration = int((ann[2] - ann[1]) * fps_i3d)
                    radius = int(action_duration / gamma)
                    generate_gaussian(
                        hmap_i3d[:, class_], center, radius, tau, ku)
                    num_action_i3d += 1
                    center_loc_i3d.append([center, class_])
                    action_lengths_i3d.append([action_duration])

            # Loop for CLIP frames
            for fr in range(0, num_feat_clip, 1):
                if fr / fps_clip > ann[1] and fr / fps_clip < ann[2]:
                    label_clip[fr, ann[0]] = 1
                if (fr+1) / fps_clip > mid_point and fr / fps_clip < mid_point:
                    center = fr + 1
                    class_ = ann[0]
                    action_duration = int((ann[2] - ann[1]) * fps_clip)
                    radius = int(action_duration / gamma)
                    generate_gaussian(
                        hmap_clip[:, class_], center, radius, tau, ku)
                    num_action_clip += 1
                    center_loc_clip.append([center, class_])
                    action_lengths_clip.append([action_duration])

        dataset.append((vid, label_i3d, label_clip, data[vid]['duration'], [
                       hmap_i3d, hmap_clip, num_action_i3d, num_action_clip, np.asarray(center_loc_i3d), np.asarray(center_loc_clip),  np.asarray(action_lengths_i3d), np.asarray(action_lengths_clip)]))
        i += 1

    return dataset


class Charades(data_utl.Dataset):

    def __init__(self, split_file, split, i3d_root, clip_root, batch_size, classes, num_clips, skip):
        self.data = make_dataset(
            split_file, split, i3d_root, clip_root, classes)
        self.split = split
        self.batch_size = batch_size
        self.i3d_root = i3d_root
        self.clip_root = clip_root
        self.num_clips = num_clips
        self.skip = skip

    def __getitem__(self, index):
        entry = self.data[index]

        feat_i3d = np.load(os.path.join(self.i3d_root, entry[0] + '.npy'))
        feat_i3d = feat_i3d.reshape(
            (feat_i3d.shape[0], 1, 1, feat_i3d.shape[-1])).astype(np.float32)

        feat_clip = np.load(os.path.join(self.clip_root, entry[0] + '.npy'))
        feat_clip = feat_clip.reshape(
            (feat_clip.shape[0], 1, 1, feat_clip.shape[-1])).astype(np.float32)

        labels_i3d = entry[1]
        labels_clip = entry[2]
        hmap_i3d, hmap_clip, num_action_i3d, num_action_clip, center_loc_i3d, center_loc_clip, action_lengths_i3d, action_lengths_clip = entry[
            4]

        num_clips = self.num_clips

        if self.split in ["training", "testing"]:
            if len(feat_i3d) > num_clips:
                if self.split == "testing":
                    random_index = 0
                else:
                    random_index = random.choice(
                        range(0, len(feat_i3d) - num_clips))

                feat_i3d = feat_i3d[random_index: random_index + num_clips]
                labels_i3d = labels_i3d[random_index: random_index + num_clips]
                hmap_i3d = hmap_i3d[random_index: random_index + num_clips]

            if len(feat_clip) > num_clips:
                if self.split == "testing":
                    random_index = 0
                else:
                    random_index = random.choice(
                        range(0, len(feat_clip) - num_clips))

                feat_clip = feat_clip[random_index: random_index + num_clips]
                labels_clip = labels_clip[random_index: random_index + num_clips]
                hmap_clip = hmap_clip[random_index: random_index + num_clips]

        return feat_i3d, feat_clip, labels_i3d, labels_clip, hmap_i3d, hmap_clip, action_lengths_i3d, action_lengths_clip, [entry[0], entry[3], num_action_i3d], [entry[0], entry[3], num_action_clip]

    def __len__(self):
        return len(self.data)


class collate_fn_unisize():

    def __init__(self, num_clips):
        self.num_clips = num_clips

    def charades_collate_fn_unisize(self, batch):
        max_len = int(self.num_clips)
        new_batch = []

        for i, b in enumerate(batch):
            # Extract I3D and CLIP features, labels, heatmaps, and masks
            features_i3d, features_clip = b[0], b[1]
            labels_i3d, labels_clip = b[2], b[3]
            hmap_i3d, hmap_clip = b[4], b[5]

            f_i3d = np.zeros(
                (max_len, features_i3d.shape[1], features_i3d.shape[2], features_i3d.shape[3]), np.float32)
            l_i3d = np.zeros((max_len, labels_i3d.shape[1]), np.float32)
            h_i3d = np.zeros((max_len, hmap_i3d.shape[1]), np.float32)
            mask_i3d = np.zeros((max_len), np.float32)

            f_clip = np.zeros(
                (max_len, features_clip.shape[1], features_clip.shape[2], features_clip.shape[3]), np.float32)
            l_clip = np.zeros((max_len, labels_clip.shape[1]), np.float32)
            h_clip = np.zeros((max_len, hmap_clip.shape[1]), np.float32)
            mask_clip = np.zeros((max_len), np.float32)

            i3d_len = min(features_i3d.shape[0], max_len)
            f_i3d[:i3d_len] = features_i3d[:i3d_len]
            l_i3d[:i3d_len] = labels_i3d[:i3d_len]
            h_i3d[:i3d_len] = hmap_i3d[:i3d_len]
            mask_i3d[:i3d_len] = 1

            clip_len = min(features_clip.shape[0], max_len)
            f_clip[:clip_len] = features_clip[:clip_len]
            l_clip[:clip_len] = labels_clip[:clip_len]
            h_clip[:clip_len] = hmap_clip[:clip_len]
            mask_clip[:clip_len] = 1

            # Debugging print statements to check shapes
            # print(f"Batch {i} - Padded I3D feature shape: {f_i3d.shape}")
            # print(f"Batch {i} - Padded CLIP feature shape: {f_clip.shape}")
            # print(f"Batch {i} - Padded I3D label shape: {l_i3d.shape}")
            # print(f"Batch {i} - Padded CLIP label shape: {l_clip.shape}")
           # print(f"Batch {i} - Padded I3D heatmap shape: {h_i3d.shape}")
           # print(f"Batch {i} - Padded CLIP heatmap shape: {h_clip.shape}")
            # print(mask_clip.shape)
            # print(mask_i3d.shape)
            # print(b[8])
            # print(b[9])

            new_batch.append([
                video_to_tensor(f_i3d),
                video_to_tensor(f_clip),
                torch.from_numpy(l_i3d),
                torch.from_numpy(l_clip),
                torch.from_numpy(mask_i3d),
                torch.from_numpy(mask_clip),
                torch.from_numpy(h_i3d),
                torch.from_numpy(h_clip),
                b[8], b[9]
            ])

        # Print the size of the final batch
        # print(f"Final new_batch size: {len(new_batch)}")

        return default_collate(new_batch)
