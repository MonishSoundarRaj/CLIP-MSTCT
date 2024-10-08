#!/usr/bin/env bash
export PATH=/pytorch_env/bin:$PATH

python train.py \
-dataset charades \
-mode rgb \
-model MS_TCT \
-train False \
-train_mode "interleaved" \
-num_clips 256 \
-skip 0 \
-lr 0.0001 \
-comp_info False \
-epoch 50 \
-unisize True \
-alpha_l 1 \
-beta_l 0.05 \
-batch_size 32 \
-rgb_root "/home/data/msoundar/outputs/" \
-clip_root "/home/msoundar/clip-features-extract/clip_features_converted_l14/"
