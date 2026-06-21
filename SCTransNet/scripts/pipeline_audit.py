#!/usr/bin/env python3
"""P0: DenseSIRST / SCTransNet 实验管线可信性检查。"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage import measure
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dataset import TestSetLoader, TrainSetLoader, _load_image_and_mask, _read_split, _resolve_dataset_layout
from metrics import PD_FA, mIoU
from model.SCTransNet import SCTransNet
import model.Config as config
from utils import get_img_norm_cfg


def _image_hw(size):
    h, w = size[0], size[1]
    if isinstance(h, torch.Tensor):
        h = int(h.item())
    if isinstance(w, torch.Tensor):
        w = int(w.item())
    return int(h), int(w)


def _pixel_area(size):
    h, w = _image_hw(size)
    return h * w


def check_correspondence(layout, sample_ids, out_dir):
    missing_img, missing_mask, shape_mismatch = [], [], []
    fg_stats = []
    mask_value_sets = []

    for sid in sample_ids:
        img_dir = layout['image_dir']
        mask_dir = layout['mask_dir']
        img_path = mask_path = None
        for ext in ['.png', '.bmp', '.jpg']:
            p = os.path.join(img_dir, sid + ext)
            if os.path.exists(p):
                img_path = p
                break
        for suf in ['_pixels0.png', '.png', '_pixels0.bmp', '.bmp']:
            p = os.path.join(mask_dir, sid + suf)
            if os.path.exists(p):
                mask_path = p
                break
        if img_path is None:
            missing_img.append(sid)
            continue
        if mask_path is None:
            missing_mask.append(sid)
            continue
        img = np.array(Image.open(img_path).convert('I'))
        mask = np.array(Image.open(mask_path))
        if img.shape[:2] != mask.shape[:2]:
            shape_mismatch.append((sid, img.shape[:2], mask.shape[:2]))
        uniq = np.unique(mask)
        mask_value_sets.append(set(uniq.tolist()))
        bin_mask = (mask > 127).astype(np.float32)
        if bin_mask.ndim == 3:
            bin_mask = bin_mask[:, :, 0]
        fg_stats.append(bin_mask.mean())

    report = {
        'checked': len(sample_ids),
        'missing_img': len(missing_img),
        'missing_mask': len(missing_mask),
        'shape_mismatch': len(shape_mismatch),
        'fg_ratio_min': float(np.min(fg_stats)) if fg_stats else None,
        'fg_ratio_mean': float(np.mean(fg_stats)) if fg_stats else None,
        'fg_ratio_max': float(np.max(fg_stats)) if fg_stats else None,
        'fg_ratio_median': float(np.median(fg_stats)) if fg_stats else None,
        'mask_unique_values_union': sorted(set().union(*mask_value_sets)) if mask_value_sets else [],
    }
    if missing_img[:5]:
        report['missing_img_examples'] = missing_img[:5]
    if missing_mask[:5]:
        report['missing_mask_examples'] = missing_mask[:5]
    if shape_mismatch[:3]:
        report['shape_mismatch_examples'] = shape_mismatch[:3]
    return report


def check_splits(layout, dataset_name):
    splits = {}
    for name in ['train_v2', 'val_v2', 'test_v2']:
        phase = name.split('_')[0]
        ids = _read_split(layout, dataset_name, phase, name)
        splits[name] = {'count': len(ids), 'first3': ids[:3], 'last3': ids[-3:]}
    train_ids = set(_read_split(layout, dataset_name, 'train', 'train_v2'))
    val_ids = set(_read_split(layout, dataset_name, 'val', 'val_v2'))
    test_ids = set(_read_split(layout, dataset_name, 'test', 'test_v2'))
    splits['overlap'] = {
        'train_val': len(train_ids & val_ids),
        'train_test': len(train_ids & test_ids),
        'val_test': len(val_ids & test_ids),
    }
    splits['union_count'] = len(train_ids | val_ids | test_ids)
    return splits


def save_train_visualizations(dataset_dir, dataset_name, out_dir, n=6):
    os.makedirs(out_dir, exist_ok=True)
    train_set = TrainSetLoader(dataset_dir, dataset_name, patch_size=256, split_name='train_v2')
    indices = np.linspace(0, len(train_set) - 1, n, dtype=int)
    for i, idx in enumerate(indices):
        img_t, mask_t = train_set[idx]
        img = img_t[0].numpy()
        mask = mask_t[0].numpy()
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title(f'train norm img #{idx}')
        axes[1].imshow(mask, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f'mask fg={mask.mean():.5f}')
        overlay = np.stack([img] * 3, axis=-1)
        overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min() + 1e-6)
        overlay[mask > 0.5, 0] = 1.0
        axes[2].imshow(overlay)
        axes[2].set_title('overlay')
        for ax in axes:
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'train_sample_{i:02d}.png'), dpi=120)
        plt.close(fig)


def threshold_sweep(net, loader, thresholds, device):
    rows = []
    for thr in thresholds:
        miou = mIoU()
        pdfa = PD_FA()
        pred_fg_total, gt_fg_total, n_cc_pred, n_cc_gt = 0.0, 0.0, 0, 0
        with torch.no_grad():
            for img, gt_mask, size, _ in loader:
                h, w = _image_hw(size)
                img = img.to(device)
                pred = net(img)
                if isinstance(pred, (tuple, list)):
                    pred = pred[-1]
                pred = pred[:, :, :h, :w]
                gt_mask = gt_mask[:, :, :h, :w]
                bin_pred = (pred > thr).cpu()
                miou.update(bin_pred, gt_mask.cpu())
                pdfa.update(bin_pred[0, 0], gt_mask[0, 0], size)
                pred_fg_total += (pred > thr).float().mean().item()
                gt_fg_total += (gt_mask > 0.5).float().mean().item()
                n_cc_pred += measure.label((pred[0, 0] > thr).cpu().numpy().astype(np.uint8), connectivity=2).max()
                n_cc_gt += measure.label((gt_mask[0, 0] > 0.5).cpu().numpy().astype(np.uint8), connectivity=2).max()
        pixacc, miou_v = miou.get()
        pd_v, fa_v = pdfa.get()
        n = max(len(loader), 1)
        rows.append({
            'threshold': thr,
            'pixAcc': float(pixacc),
            'mIoU': float(miou_v),
            'PD': float(pd_v),
            'FA': float(fa_v),
            'pred_fg_ratio_mean': pred_fg_total / n,
            'gt_fg_ratio_mean': gt_fg_total / n,
            'pred_cc_total': int(n_cc_pred),
            'gt_cc_total': int(n_cc_gt),
        })
    return rows


def save_pred_visualizations(net, loader, out_dir, device, n=6, threshold=0.5):
    os.makedirs(out_dir, exist_ok=True)
    net.eval()
    saved = 0
    with torch.no_grad():
        for img, gt_mask, size, sid in loader:
            if saved >= n:
                break
            h, w = _image_hw(size)
            img = img.to(device)
            pred = net(img)
            if isinstance(pred, (tuple, list)):
                pred = pred[-1]
            pred = pred[:, :, :h, :w]
            gt_mask = gt_mask[:, :, :h, :w]
            p = pred[0, 0].cpu().numpy()
            g = gt_mask[0, 0].numpy()
            raw_img = img[0, 0].cpu().numpy()
            bin_p = (p > threshold).astype(np.float32)

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(raw_img, cmap='gray')
            axes[0].set_title('input')
            axes[1].imshow(g, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title(f'GT fg={g.mean():.5f}')
            axes[2].imshow(p, cmap='hot', vmin=0, vmax=1)
            axes[2].set_title(f'pred max={p.max():.3f} mean={p.mean():.5f}')
            axes[3].imshow(bin_p, cmap='gray', vmin=0, vmax=1)
            axes[3].set_title(f'bin@{threshold} fg={bin_p.mean():.5f}')
            sid_str = sid[0] if isinstance(sid, (list, tuple)) else sid
            fig.suptitle(str(sid_str))
            for ax in axes:
                ax.axis('off')
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f'val_pred_{saved:02d}_{sid_str}.png'), dpi=120)
            plt.close(fig)
            saved += 1


def probe_untrained_stats(net, loader, device, threshold=0.5):
    pred_means, pred_maxs, fg_ratios = [], [], []
    with torch.no_grad():
        for img, gt_mask, size, _ in loader:
            h, w = _image_hw(size)
            pred = net(img.to(device))
            if isinstance(pred, (tuple, list)):
                pred = pred[-1]
            pred = pred[:, :, :h, :w]
            pred_means.append(pred.mean().item())
            pred_maxs.append(pred.max().item())
            fg_ratios.append((pred > threshold).float().mean().item())
    return {
        'pred_mean_avg': float(np.mean(pred_means)),
        'pred_max_avg': float(np.mean(pred_maxs)),
        'pred_fg_ratio_at_threshold': float(np.mean(fg_ratios)),
        'n_images': len(pred_means),
    }


def main():
    parser = argparse.ArgumentParser(description='P0 pipeline audit for DenseSIRST')
    parser.add_argument('--dataset_dir', default='./datasets')
    parser.add_argument('--dataset_name', default='DenseSIRST')
    parser.add_argument('--gpu', default='4')
    parser.add_argument('--out_dir', default='./log/audit')
    parser.add_argument('--checkpoint', default='', help='optional checkpoint for pred viz / sweep')
    parser.add_argument('--max_val_samples', type=int, default=0, help='0=all val')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(args.out_dir, f'{args.dataset_name}_{stamp}')
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, 'audit.log')

    def log(msg):
        print(msg, flush=True)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

    layout = _resolve_dataset_layout(args.dataset_dir, args.dataset_name)
    log(f'=== P0 Pipeline Audit {stamp} ===')
    log(f'dataset_root: {layout["root"]} type={layout["type"]}')

    all_ids = []
    for split in ['train_v2', 'val_v2', 'test_v2']:
        phase = split.split('_')[0]
        all_ids.extend(_read_split(layout, args.dataset_name, phase, split))
    unique_ids = sorted(set(all_ids))
    log(f'total unique ids in v2 splits: {len(unique_ids)}')

    corr = check_correspondence(layout, unique_ids, out_dir)
    log(f'[correspondence] {json.dumps(corr, ensure_ascii=False, indent=2)}')

    splits = check_splits(layout, args.dataset_name)
    log(f'[splits] {json.dumps(splits, ensure_ascii=False, indent=2)}')

    norm_cfg = get_img_norm_cfg(args.dataset_name, args.dataset_dir, 'train_v2')
    log(f'[normalization] {norm_cfg}')

    save_train_visualizations(args.dataset_dir, args.dataset_name,
                              os.path.join(out_dir, 'train_samples'))

    val_set = TestSetLoader(args.dataset_dir, args.dataset_name, args.dataset_name,
                            split_name='val_v2', phase='val')
    if args.max_val_samples > 0:
        val_set.test_list = val_set.test_list[:args.max_val_samples]
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

    cfg = config.get_SCTrans_config()
    net = SCTransNet(cfg, mode='test', deepsuper=True).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt.get('state_dict', ckpt)
        if any(k.startswith('model.') for k in state):
            from train import Net
            wrapper = Net('SCTransNet', mode='test').to(device)
            wrapper.load_state_dict(state, strict=False)
            net = wrapper.model
        else:
            net.load_state_dict({k.replace('module.', ''): v for k, v in state.items()}, strict=False)
        log(f'loaded checkpoint: {args.checkpoint}')
    else:
        log('using untrained model for probe')

    untrained = probe_untrained_stats(net, val_loader, device)
    log(f'[untrained/probe pred stats] {json.dumps(untrained, indent=2)}')

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    sweep = threshold_sweep(net, val_loader, thresholds, device)
    sweep_path = os.path.join(out_dir, 'threshold_sweep.csv')
    with open(sweep_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
        writer.writeheader()
        writer.writerows(sweep)
    log('[threshold sweep]')
    for row in sweep:
        log(f"  thr={row['threshold']:.1f} mIoU={row['mIoU']:.6f} PD={row['PD']:.4f} "
            f"FA={row['FA']:.6f} pred_fg={row['pred_fg_ratio_mean']:.6f}")

    save_pred_visualizations(net, val_loader, os.path.join(out_dir, 'val_predictions'),
                             device, n=8, threshold=0.5)

    summary = {
        'timestamp': stamp,
        'out_dir': out_dir,
        'correspondence': corr,
        'splits': splits,
        'normalization': norm_cfg,
        'probe': untrained,
        'threshold_sweep': sweep,
    }
    with open(os.path.join(out_dir, 'audit_summary.json'), 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f'=== audit done -> {out_dir} ===')


if __name__ == '__main__':
    main()
