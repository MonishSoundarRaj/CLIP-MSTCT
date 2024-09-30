import json
import pickle5 as pickle
from apmeter import APMeter
import numpy as np
from utils import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def make_gt_split(gt_file, logits, num_classes=157, num_frames=None):
    gt_new = {}
    vid_length = {}
    fps_seg = {}
    with open(gt_file, 'r') as f:
        gt = json.load(f)

    for vid in gt.keys():
        if gt[vid]['subset'] != "testing":
            continue

        if vid not in logits.keys():
            continue

        if num_frames is not None:
            num_pred = num_frames
        else:
            num_pred = logits[vid].shape[1]

        label = np.zeros((num_pred, num_classes), np.float32)

        fps = float(num_pred / float(gt[vid]['duration']))
        for ann in gt[vid]['actions']:
            for fr in range(0, num_pred, 1):
                if fr / fps > ann[1] and fr / fps < ann[2]:
                    label[fr, ann[0]] = 1
        gt_new[vid] = label
        vid_length[vid] = gt[vid]['duration']
        fps_seg[vid] = fps
    return gt_new, vid_length, fps_seg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-pkl_path', type=str)
    parser.add_argument('-metadata_path', type=str)
    args = parser.parse_args()

    pkl_path = args.pkl_path
    metadata_path = args.metadata_path

    gt_file = './data/charades.json'
    classes = 157

    with open(pkl_path, 'rb') as pkl:
        logits = pickle.load(pkl)

    with open(metadata_path, 'rb') as meta_pkl:
        frame_metadata = pickle.load(meta_pkl)

    apm_i3d = APMeter()
    apm_clip = APMeter()
    sampled_apm_i3d = APMeter()
    sampled_apm_clip = APMeter()

    pred_probs_i3d = []
    pred_probs_clip = []
    gt_labels_i3d = []
    gt_labels_clip = []

    # Here the frame_metadata has the info on the video name,
    # and the frames that comes from I3D and CLIP individual, so we can split things correct for ground truth calculation
    # Video ID: 08LOY
    # I3D Frames: tensor([84])
    # CLIP Frames: tensor([43])

    # Video ID: MLWB5
    # I3D Frames: tensor([92])
    # CLIP Frames: tensor([47])

    for vid in frame_metadata.keys():
        logit = np.transpose(logits[vid], (1, 0))

        i3d_frames = int(frame_metadata[vid]['i3d_frames'])
        clip_frames = int(frame_metadata[vid]['clip_frames'])

        i3d_logits = logit[:i3d_frames, :]
        clip_logits = logit[i3d_frames:i3d_frames + clip_frames, :]

        i3d_gt, _, _ = make_gt_split(
            gt_file, logits, num_classes=classes, num_frames=i3d_frames)
        print(len(i3d_gt))
        clip_gt, _, _ = make_gt_split(
            gt_file, logits, num_classes=classes, num_frames=clip_frames)

        apm_i3d.add(i3d_logits, i3d_gt[vid])
        sampled_25_inference(i3d_logits, i3d_gt[vid], sampled_apm_i3d)

        apm_clip.add(clip_logits, clip_gt[vid])
        sampled_25_inference(clip_logits, clip_gt[vid], sampled_apm_clip)

        pred_probs_i3d.append(i3d_logits)
        gt_labels_i3d.append(i3d_gt[vid])

        pred_probs_clip.append(clip_logits)
        gt_labels_clip.append(clip_gt[vid])

    i3d_val_map = 100 * apm_i3d.value().mean()
    clip_val_map = 100 * apm_clip.value().mean()

    sampled_map_i3d = 100 * sampled_apm_i3d.value().mean()
    sampled_map_clip = 100 * sampled_apm_clip.value().mean()

    final_val_map = (i3d_val_map + clip_val_map) / 2
    final_sampled_map = (sampled_map_i3d + sampled_map_clip) / 2

    # print("I3D Frame-based map:", i3d_val_map)
    # print("CLIP Frame-based map:", clip_val_map)
    print("Combined Averaged Frame-based map:", final_val_map)

    # print("I3D Sampled Frame-based map:", sampled_map_i3d)
    # print("CLIP Sampled Frame-based map:", sampled_map_clip)
    print("25 Combined Averaged Sampled Frame-based map:", final_sampled_map)

    # Run action-conditional metrics for both I3D and CLIP separately, and average
    prec0_i3d, re0_i3d, ns0_i3d, map0_i3d = conditional_metric(
        pred_probs_i3d, gt_labels_i3d, t=0, avg=True)
    prec0_clip, re0_clip, ns0_clip, map0_clip = conditional_metric(
        pred_probs_clip, gt_labels_clip, t=0, avg=True)

    fs0_i3d = get_f1(prec0_i3d, re0_i3d)
    fs0_clip = get_f1(prec0_clip, re0_clip)

    final_prec0 = (prec0_i3d + prec0_clip) / 2
    final_re0 = (re0_i3d + re0_clip) / 2
    final_fs0 = (fs0_i3d + fs0_clip) / 2
    final_map0 = (map0_i3d + map0_clip) / 2

    print('Precision(c_i|c_j,0)=', final_prec0)
    print('Recall(c_i|c_j,0)=', final_re0)
    print('F1Score(c_i|c_j,0)=', final_fs0)
    print('mAP(c_i|c_j,0)=', final_map0)

    # t=20
    prec20_i3d, re20_i3d, ns20_i3d, map20_i3d = conditional_metric(
        pred_probs_i3d, gt_labels_i3d, t=20, avg=True)
    prec20_clip, re20_clip, ns20_clip, map20_clip = conditional_metric(
        pred_probs_clip, gt_labels_clip, t=20, avg=True)

    fs20_i3d = get_f1(prec20_i3d, re20_i3d)
    fs20_clip = get_f1(prec20_clip, re20_clip)

    # Average the metrics
    final_prec20 = (prec20_i3d + prec20_clip) / 2
    final_re20 = (re20_i3d + re20_clip) / 2
    final_fs20 = (fs20_i3d + fs20_clip) / 2
    final_map20 = (map20_i3d + map20_clip) / 2

    # Print action-conditional results for t=20
    print('Precision(c_i|c_j,20)=', final_prec20)
    print('Recall(c_i|c_j,20)=', final_re20)
    print('F1Score(c_i|c_j,20)=', final_fs20)
    print('mAP(c_i|c_j,20)=', final_map20)

    # t=40
    prec40_i3d, re40_i3d, ns40_i3d, map40_i3d = conditional_metric(
        pred_probs_i3d, gt_labels_i3d, t=40, avg=True)
    prec40_clip, re40_clip, ns40_clip, map40_clip = conditional_metric(
        pred_probs_clip, gt_labels_clip, t=40, avg=True)

    fs40_i3d = get_f1(prec40_i3d, re40_i3d)
    fs40_clip = get_f1(prec40_clip, re40_clip)

    # Average the metrics
    final_prec40 = (prec40_i3d + prec40_clip) / 2
    final_re40 = (re40_i3d + re40_clip) / 2
    final_fs40 = (fs40_i3d + fs40_clip) / 2
    final_map40 = (map40_i3d + map40_clip) / 2

    # Print action-conditional results for t=40
    print('Precision(c_i|c_j,40)=', final_prec40)
    print('Recall(c_i|c_j,40)=', final_re40)
    print('F1Score(c_i|c_j,40)=', final_fs40)
    print('mAP(c_i|c_j,40)=', final_map40)
