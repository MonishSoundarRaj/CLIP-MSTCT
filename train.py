import time
import argparse
import csv
from torch.autograd import Variable
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from utils import *
from apmeter import APMeter
import os

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow (or joint for eval)')
parser.add_argument('-train', type=str2bool,
                    default='True', help='train or eval')
parser.add_argument('-comp_info', type=str)
parser.add_argument('-gpu', type=str, default='4')
parser.add_argument('-dataset', type=str, default='charades')
parser.add_argument('-rgb_root', type=str, default='no_root')
parser.add_argument('-clip_root', type=str, default='no_root')
parser.add_argument('-lr', type=str, default='0.1')
parser.add_argument('-epoch', type=str, default='50')
parser.add_argument('-model', type=str, default='')
parser.add_argument('-load_model', type=str, default='False')
parser.add_argument('-load_model_path', type=str,
                    default='./save_model/17.pth')
parser.add_argument('-batch_size', type=str, default='False')
parser.add_argument('-num_clips', type=str, default='False')
parser.add_argument('-skip', type=str, default='False')
parser.add_argument('-num_layer', type=str, default='False')
parser.add_argument('-unisize', type=str, default='False')
parser.add_argument('-alpha_l', type=float, default='1.0')
parser.add_argument('-beta_l', type=float, default='1.0')
parser.add_argument('-save_path', type=str, default='./save_three_logit')
parser.add_argument('-save_model_path', type=str, default='./save_model')

args = parser.parse_args()

# Set random seed
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print('Random_SEED:', SEED)

batch_size = int(args.batch_size)

if args.dataset == 'charades':
    from charades_dataloader import Charades as Dataset

    if str(args.unisize) == "True":
        print("uni-size padd all T to", args.num_clips)
        from charades_dataloader import collate_fn_unisize
        collate_fn_f = collate_fn_unisize(args.num_clips)
        collate_fn = collate_fn_f.charades_collate_fn_unisize
    else:
        from charades_dataloader import mt_collate_fn as collate_fn

    train_split = './data/charades.json'
    test_split = train_split
    rgb_root = args.rgb_root
    clip_root = args.clip_root
    classes = 157


def load_data(train_split, val_split, rgb_root, clip_root):
    print('load data', rgb_root, clip_root)

    if len(train_split) > 0:
        dataset = Dataset(train_split, 'training', rgb_root, clip_root,
                          batch_size, classes, int(args.num_clips), int(args.skip))
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn)
        dataloader.root = rgb_root
    else:
        dataset = None
        dataloader = None

    val_dataset = Dataset(val_split, 'testing', rgb_root, clip_root,
                          batch_size, classes, int(args.num_clips), int(args.skip))
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn)
    val_dataloader.root = rgb_root
    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}

    return dataloaders, datasets


def run(models, criterion, num_epochs=50):
    since = time.time()
    Best_val_map = 0.
    best_model_path = None

    for epoch in range(num_epochs):
        since1 = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for model, gpu, dataloader, optimizer, sched, model_file in models:
            _, _ = train_step(model, gpu, optimizer,
                              dataloader['train'], epoch)
            prob_val, val_loss, val_map = val_step(
                model, gpu, dataloader['val'], epoch)
            sched.step(val_loss)

            print("epoch", epoch, "Total_Time", time.time() -
                  since, "Epoch_time", time.time() - since1)

            if Best_val_map < val_map:
                Best_val_map = val_map
                print("epoch", epoch, "Best Val Map Update", Best_val_map)

                # Save the logits
                pickle.dump(prob_val, open('./save_logit/' +
                            str(epoch) + '.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
                print("logit_saved at:", "./save_logit/" + str(epoch) + ".pkl")

                # Save the model
                model_save_path = f'{args.save_model_path}/{epoch}.pth'
                torch.save(model.state_dict(), model_save_path)
                print("Model saved at:", model_save_path)
                best_model_path = model_save_path


def interleave_labels_heatmaps_masks(labels_i3d, labels_clip, hmap_i3d, hmap_clip, mask_i3d, mask_clip):

    assert labels_i3d.shape == labels_clip.shape, "Labels for I3D and CLIP must have the same shape"
    assert hmap_i3d.shape == hmap_clip.shape, "Heatmaps for I3D and CLIP must have the same shape"
    assert mask_i3d.shape == mask_clip.shape, "Masks for I3D and CLIP must have the same shape"

    batch_size, num_frames, num_classes = labels_i3d.shape

    interleaved_labels = torch.zeros(
        batch_size, num_frames * 2, num_classes).cuda()
    interleaved_heatmaps = torch.zeros(
        batch_size, num_frames * 2, num_classes).cuda()
    interleaved_masks = torch.zeros(batch_size, num_frames * 2).cuda()

    interleaved_labels[:, 0::2, :] = labels_i3d
    interleaved_labels[:, 1::2, :] = labels_clip
    interleaved_heatmaps[:, 0::2, :] = hmap_i3d
    interleaved_heatmaps[:, 1::2, :] = hmap_clip
    interleaved_masks[:, 0::2] = mask_i3d
    interleaved_masks[:, 1::2] = mask_clip

    return interleaved_labels, interleaved_heatmaps, interleaved_masks


def run_network(model, data, gpu, epoch=0):
    feat_i3d, feat_clip, labels_i3d, labels_clip, mask_i3d, mask_clip, hmap_i3d, hmap_clip, other_i3d, other_clip = data

    feat_i3d = Variable(feat_i3d.cuda(gpu))
    feat_clip = Variable(feat_clip.cuda(gpu))
    mask_i3d = Variable(mask_i3d.cuda(gpu))
    mask_clip = Variable(mask_clip.cuda(gpu))
    labels_i3d = Variable(labels_i3d.cuda(gpu))
    labels_clip = Variable(labels_clip.cuda(gpu))
    hmap_i3d = Variable(hmap_i3d.cuda(gpu))
    hmap_clip = Variable(hmap_clip.cuda(gpu))
    feat_i3d = feat_i3d.squeeze(3).squeeze(3)
    feat_clip = feat_clip.squeeze(3).squeeze(3)

    outputs_final, out_hm = model(feat_i3d, feat_clip, "combined")
    combined_labels, combined_hmaps, combined_mask = interleave_labels_heatmaps_masks(
        labels_i3d, labels_clip, hmap_i3d, hmap_clip, mask_i3d, mask_clip)
    # combined_mask = torch.cat((mask_i3d, mask_clip), dim=1)
    # combined_labels = torch.cat([labels_i3d, labels_clip], dim=1)
    # combined_hmaps = torch.cat([hmap_i3d, hmap_clip], dim=1)
    # print(combined_hmaps.shape)
    # print(combined_labels.shape)
    # print(combined_mask.shape)

    # Logits
    probs_f = F.sigmoid(outputs_final) * combined_mask.unsqueeze(2)

    loss_h = focal_loss(out_hm, combined_hmaps)
    loss_f = F.binary_cross_entropy_with_logits(
        outputs_final, combined_labels, size_average=False)

    loss_f = torch.sum(loss_f) / torch.sum(combined_hmaps)
    loss = args.alpha_l * loss_f + args.beta_l * loss_h

    corr = torch.sum(combined_mask)
    tot = torch.sum(combined_mask)

    return outputs_final, loss, probs_f, corr / tot


def train_step(model, gpu, optimizer, dataloader, epoch):
    model.train(True)
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.
    apm = APMeter()

    for data in dataloader:
        optimizer.zero_grad()
        num_iter += 1
        feat_i3d, feat_clip, labels_i3d, labels_clip, mask_i3d, mask_clip, hmap_i3d, hmap_clip, other_i3d, other_clip = data
        outputs, loss, probs, err = run_network(model, data, gpu, epoch)
        combined_labels = torch.cat([labels_i3d, labels_clip], dim=1)
        apm.add(probs.data.cpu().numpy()[0], combined_labels.cpu().numpy()[0])
        error += err.data
        tot_loss += loss.data
        loss.backward()
        optimizer.step()

    train_map = 100 * apm.value().mean()
    print('epoch', epoch, 'train-map:', train_map)
    apm.reset()

    epoch_loss = tot_loss / num_iter

    return train_map, epoch_loss


def val_step(model, gpu, dataloader, epoch):
    model.train(False)
    apm = APMeter()
    sampled_apm = APMeter()
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.
    full_probs = {}

    for data in dataloader:
        num_iter += 1
        feat_i3d, feat_clip, labels_i3d, labels_clip, mask_i3d, mask_clip, hmap_i3d, hmap_clip, other_i3d, other_clip = data
        combined_labels, combined_hmaps, combined_mask = interleave_labels_heatmaps_masks(
            labels_i3d, labels_clip, hmap_i3d, hmap_clip, mask_i3d, mask_clip)
        # combined_mask = torch.cat([mask_i3d, mask_clip], dim=1)

        outputs, loss, probs, err = run_network(model, data, gpu, epoch)

        # combined_labels = torch.cat([labels_i3d, labels_clip], dim=1)

        if sum(combined_mask.cpu().numpy()[0]) > 25:
            p1, l1 = sampled_25(probs.data.cpu().numpy()[0], combined_labels.cpu().numpy()[
                                0], combined_mask.cpu().numpy()[0])
            sampled_apm.add(p1, l1)

        apm.add(probs.data.cpu().numpy()[0], combined_labels.cpu().numpy()[0])
        error += err.data
        tot_loss += loss.data
        probs_1 = mask_probs(probs.data.cpu().numpy()[
                             0], combined_mask.cpu().numpy()[0]).squeeze()
        # print(other)
        full_probs[other_i3d[0][0]] = probs_1.T

    epoch_loss = tot_loss / num_iter
    val_map = torch.sum(100 * apm.value()) / \
        torch.nonzero(100 * apm.value()).size()[0]
    sample_val_map = torch.sum(
        100 * sampled_apm.value()) / torch.nonzero(100 * sampled_apm.value()).size()[0]

    print('epoch', epoch, 'Full-val-map:', val_map)
    print('epoch', epoch, 'sampled-val-map:', sample_val_map)
    print(100 * sampled_apm.value())

    # Reset APMeters
    apm.reset()
    sampled_apm.reset()

    return full_probs, epoch_loss, val_map


def run_network_i3d(model, inputs_i3d, mask_i3d, gpu):
    inputs_i3d = Variable(inputs_i3d.cuda(gpu))
    mask_i3d = Variable(mask_i3d.cuda(gpu))

    inputs_i3d = inputs_i3d.squeeze(3).squeeze(3)

    outputs_final, out_hm = model(inputs_i3d=inputs_i3d, mode='i3d')

    probs = F.sigmoid(outputs_final)

    probs_f = mask_probs(probs.detach().cpu().numpy()[
                         0], mask_i3d.detach().cpu().numpy()[0]).squeeze()

    return outputs_final, probs_f


def run_network_clip(model, inputs_clip, mask_clip, gpu):
    inputs_clip = Variable(inputs_clip.cuda(gpu))
    mask_clip = Variable(mask_clip.cuda(gpu))

    inputs_clip = inputs_clip.squeeze(3).squeeze(3)

    outputs_final, out_hm = model(
        inputs_i3d=None, inputs_clip=inputs_clip,  mode='clip')

    probs = F.sigmoid(outputs_final)
    probs_f = mask_probs(probs.detach().cpu().numpy()[
                         0], mask_clip.detach().cpu().numpy()[0]).squeeze()

    return outputs_final, probs_f


# def run_network_combined(model, inputs_i3d, inputs_clip, mask_i3d, mask_clip, gpu):
#     inputs_i3d = Variable(inputs_i3d.cuda(gpu))
#     inputs_clip = Variable(inputs_clip.cuda(gpu))
#     mask_i3d = Variable(mask_i3d.cuda(gpu))
#     mask_clip = Variable(mask_clip.cuda(gpu))

#     inputs_i3d = inputs_i3d.squeeze(3).squeeze(3)
#     inputs_clip = inputs_clip.squeeze(3).squeeze(3)

#     # Forward pass for combined I3D + CLIP
#     outputs_final, out_hm = model(
#         inputs_i3d=inputs_i3d, inputs_clip=inputs_clip, mode='combined')
#     combined_labels, combined_hmaps, combined_mask = interleave_labels_heatmaps_masks()
#     # combined_mask = torch.cat([mask_i3d, mask_clip], dim=1)
#     probs = F.sigmoid(outputs_final)
#     probs_f = mask_probs(probs.detach().cpu().numpy()[
#                          0], combined_mask.detach().cpu().numpy()[0]).squeeze()

#     return outputs_final, probs_f


def eval_model(model, dataloader, save_path):
    eval_step(model, 0, dataloader, save_path)


def load_model_for_eval(model, model_path):
    print(f"Loading model weights from: {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def eval_step(model, gpu, dataloader, save_path):
    model.train(False)
    full_probs_i3d = {}
    full_probs_clip = {}
    full_probs_combined = {}
    frame_metadata = {}

    for data in dataloader:
        feat_i3d, feat_clip, labels_i3d, labels_clip, mask_i3d, mask_clip, hmap_i3d, hmap_clip, other_i3d, other_clip = data

        # Run I3D-only inference
        outputs_i3d, probs_i3d = run_network_i3d(
            model, feat_i3d, mask_i3d, gpu)
        full_probs_i3d[other_i3d[0][0]
                       ] = probs_i3d.T

        # Run CLIP-only inference
        outputs_clip, probs_clip = run_network_clip(
            model, feat_clip, mask_clip, gpu)
        full_probs_clip[other_clip[0][0]
                        ] = probs_clip.T

        # # Run combined inference
        # outputs_combined, probs_combined = run_network_combined(
        #     model, feat_i3d, feat_clip, mask_i3d, mask_clip, gpu)
        # full_probs_combined[other_i3d[0][0]
        #                     ] = probs_combined.T

        frame_metadata[other_i3d[0][0]] = {
            'i3d_frames': other_i3d[3],
            'clip_frames': other_clip[3]
        }

    pickle.dump(full_probs_i3d, open(
        f'{save_path}/logits_i3d.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(full_probs_clip, open(
        f'{save_path}/logits_clip.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    # pickle.dump(full_probs_combined, open(
    #     f'{save_path}/logits_combined.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

    # pickle.dump(frame_metadata, open(
    #     f'{save_path}/frame_metadata.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    if not args.train:
        dataloaders, datasets = load_data(
            train_split, test_split, rgb_root, clip_root)
        print("Running evaluation mode...")
        model_path = args.load_model_path
        if not os.path.exists('./save_three_logit'):
            os.makedirs('./save_three_logit')
        from MSTCT.MSTCT_Model import MSTCT
        num_clips = int(args.num_clips)
        num_classes = classes
        inter_channels = [256, 384, 576, 864]
        num_block = 3
        head = 8
        mlp_ratio = 8
        in_feat_dim = 1024
        final_embedding_dim = 512

        input_size = 768

        model = MSTCT(inter_channels, num_block, head, mlp_ratio,
                      in_feat_dim, final_embedding_dim, num_classes, input_size)
        model = load_model_for_eval(model, model_path)
        model = model.cuda()

        eval_model(model, dataloaders['val'], args.save_path)

    else:
        from MSTCT.MSTCT_Model import MSTCT
        if args.mode == 'flow':
            print('flow mode', rgb_root)
            dataloaders, datasets = load_data(
                train_split, test_split, rgb_root, clip_root)
        elif args.mode == 'rgb':
            print('RGB mode', rgb_root)
            dataloaders, datasets = load_data(
                train_split, test_split, rgb_root, clip_root)

        if not os.path.exists('./save_logit'):
            os.makedirs('./save_logit')
        if not os.path.exists('./save_model'):
            os.makedirs('./save_model')
        if args.train:
            if args.model == "MS_TCT":
                print("MS_TCT")
                from MSTCT.MSTCT_Model import MSTCT
                num_clips = int(args.num_clips)
                num_classes = classes
                inter_channels = [256, 384, 576, 864]
                num_block = 3
                head = 8
                mlp_ratio = 8
                in_feat_dim = 1024
                final_embedding_dim = 512
                input_size = 768

                rgb_model = MSTCT(inter_channels, num_block, head, mlp_ratio,
                                  in_feat_dim, final_embedding_dim, num_classes, input_size)
                print("loaded", args.load_model)

                rgb_model.cuda()

                criterion = nn.NLLLoss(reduce=False)
                lr = float(args.lr)
                optimizer = optim.Adam(rgb_model.parameters(), lr=lr)
                lr_sched = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, factor=0.5, patience=8, verbose=True)
                run([(rgb_model, 0, dataloaders, optimizer, lr_sched,
                    args.comp_info)], criterion, num_epochs=int(args.epoch))
