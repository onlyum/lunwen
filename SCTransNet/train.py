import argparse
import csv
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import measure
from torch.nn import init
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from dataset import TrainSetLoader, TrainSetLoader02, TrainSetLoader03, TrainSetLoader04, TestSetLoader
from metrics import mIoU, PD_FA
from model.SCTransNet import SCTransNet
import model.Config as config
from utils import get_optimizer, seed_pytorch


parser = argparse.ArgumentParser(description="PyTorch SCTransNet train")
parser.add_argument("--model_names", default='SCTransNet', type=str, help="'SCTransNet'")
parser.add_argument("--dataset_names", default='DenseSIRST', type=str)
parser.add_argument("--experiment_name", default=None, type=str)
parser.add_argument("--optimizer_name", default='Adam', type=str, help="optimizer name: AdamW, Adam, Adagrad, SGD")
parser.add_argument("--epochs", default=1000, type=int)
parser.add_argument("--begin_val", default=500, type=int,
                    help='First epoch to run validation')
parser.add_argument("--val_interval", default=10, type=int,
                    help='Validate every N epochs')
parser.add_argument("--every_test", default=None, type=int, help="Deprecated alias for --val_interval")
parser.add_argument("--test_interval", default=50, type=int,
                    help='Run full test split every N epochs (0=off)')
parser.add_argument("--val_max_samples", default=0, type=int,
                    help='Quick val subset size; 0 means always use full split')
parser.add_argument("--full_val_interval", default=1, type=int,
                    help='Run full val and update best checkpoint every N epochs')
parser.add_argument("--every_save_pth", default=1000, type=int)
parser.add_argument("--every_print", default=1, type=int)
parser.add_argument("--dataset_dir", default=r'./datasets')
parser.add_argument("--gpu", default='4', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument("--lr", default=None, type=float, help='Override default optimizer lr')
parser.add_argument("--train_split", default='train_v2', type=str)
parser.add_argument("--val_split", default='val_v2', type=str)
parser.add_argument("--test_split", default='test_v2', type=str)
parser.add_argument("--batchSize", type=int, default=16, help="Training batch size")
parser.add_argument("--patchSize", type=int, default=256, help="Training patch size")
parser.add_argument("--save", default=r'./log', type=str, help="Save path of checkpoints")
parser.add_argument("--log_dir", type=str, default="./otherlogs/SCTransNet", help='path of tensorboard log files')
parser.add_argument("--record_dir", type=str, default="./records", help='path of csv result records')
parser.add_argument("--img_norm_cfg", default=None, type=dict)
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use")
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", default=False, type=list, help="Resume from existing checkpoints")

parser.add_argument("--loss_mode", default='bce',
                    choices=['bce', 'weighted_bce', 'bce_softiou', 'bce_dice', 'tversky', 'focal_tversky'])
parser.add_argument("--bce_weight", default=1.0, type=float)
parser.add_argument("--softiou_weight", default=0.0, type=float)
parser.add_argument("--dice_weight", default=0.0, type=float)
parser.add_argument("--pos_weight", default=50.0, type=float,
                    help='Foreground weight for weighted_bce (auto if <=0)')
parser.add_argument("--tversky_alpha", default=0.7, type=float)
parser.add_argument("--tversky_beta", default=0.3, type=float)
parser.add_argument("--focal_gamma", default=0.75, type=float)
parser.add_argument("--viz_interval", default=0, type=int,
                    help='Save val viz every N full-val epochs (0=off)')
parser.add_argument("--threshold_sweep_interval", default=0, type=int,
                    help='Run threshold sweep every N epochs on full val only (0=off)')
parser.add_argument("--progress", action='store_true', default=True, help='Show tqdm progress bars')
parser.add_argument("--use_prior", action='store_true')
parser.add_argument("--prior_gamma_init", default=0.0, type=float)
parser.add_argument("--use_aux", action='store_true')
parser.add_argument("--center_weight", default=0.0, type=float)
parser.add_argument("--edge_weight", default=0.0, type=float)
parser.add_argument("--center_mode", default='avg', choices=['avg', 'gaussian'])
parser.add_argument("--center_dirac_area", default=2, type=int)
parser.add_argument("--center_min_sigma", default=0.5, type=float)
parser.add_argument("--center_max_sigma", default=1.5, type=float)
parser.add_argument("--fa_weight", default=0.0, type=float)
parser.add_argument("--valley_weight", default=0.0, type=float)
parser.add_argument("--valley_max_distance", default=24.0, type=float)
parser.add_argument("--valley_line_width", default=1, type=int)
parser.add_argument("--repulsion_sigma", default=5.0, type=float)
parser.add_argument("--repulsion_w0", default=1.0, type=float)
parser.add_argument("--train_loader_variant", default='normal', choices=['normal', 'noise', 'gamma', 'noise_gamma'])
parser.add_argument("--early_stop_patience", default=0, type=int)
parser.add_argument("--min_epochs", default=0, type=int)

global opt
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
if opt.every_test is not None:
    opt.val_interval = opt.every_test


def _parse_name_list(value):
    if isinstance(value, list):
        return value
    return [item.strip() for item in str(value).split(',') if item.strip()]


opt.model_names = _parse_name_list(opt.model_names)
opt.dataset_names = _parse_name_list(opt.dataset_names)

seed_pytorch(opt.seed)

config_vit = config.get_SCTrans_config()

if not torch.cuda.is_available():
    raise RuntimeError('CUDA is required for training; CPU loading/training is not supported.')
print('Using CUDA device for training:', torch.cuda.get_device_name(0))


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        if getattr(m, 'bias', None) is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        if getattr(m, 'bias', None) is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def _train_loader_class():
    loaders = {
        'normal': TrainSetLoader,
        'noise': TrainSetLoader02,
        'gamma': TrainSetLoader03,
        'noise_gamma': TrainSetLoader04,
    }
    return loaders[opt.train_loader_variant]


def _final_pred(preds):
    if isinstance(preds, dict):
        preds = preds['seg']
    if isinstance(preds, (tuple, list)):
        return preds[-1]
    return preds


def _as_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        return float(value.item())
    return float(value)


def _image_hw(size):
    h, w = size[0], size[1]
    if isinstance(h, torch.Tensor):
        h = int(h.item())
    if isinstance(w, torch.Tensor):
        w = int(w.item())
    return int(h), int(w)


def _init_record_file():
    os.makedirs(opt.run_record_dir, exist_ok=True)
    with open(opt.run_record_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'time', 'experiment', 'dataset', 'model', 'epoch', 'split',
            'loss', 'pixAcc', 'mIoU', 'PD', 'FA', 'lr',
            'pred_fg_ratio', 'gt_fg_ratio', 'pred_cc', 'gt_cc',
        ])


def _init_epoch_metrics_file():
    path = os.path.join(opt.run_save_dir, 'epoch_metrics.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'train_loss', 'lr', 'val_mode', 'val_samples', 'val_loss', 'val_mIoU', 'val_PD', 'val_FA',
            'val_pred_fg_ratio', 'val_gt_fg_ratio', 'val_pred_cc', 'val_gt_cc', 'best_updated',
        ])
    return path


def _append_epoch_metrics(path, row):
    with open(path, 'a', newline='') as f:
        csv.writer(f).writerow(row)


def _count_cc(bin_map):
    labeled = measure.label(bin_map.astype(np.uint8), connectivity=2)
    return max(int(labeled.max()), 0)


def _collect_batch_stats(pred, gt_mask, threshold):
    pred_np = pred.detach().cpu().numpy()
    gt_np = gt_mask.detach().cpu().numpy()
    bin_pred = (pred_np > threshold).astype(np.uint8)
    bin_gt = (gt_np > 0.5).astype(np.uint8)
    stats = {
        'pred_fg_ratio': float(bin_pred.mean()),
        'gt_fg_ratio': float(bin_gt.mean()),
        'pred_cc': _count_cc(bin_pred[0, 0]),
        'gt_cc': _count_cc(bin_gt[0, 0]),
        'pred_mean': float(pred_np.mean()),
        'pred_max': float(pred_np.max()),
    }
    return stats


def _run_threshold_sweep(net, eval_loader, thresholds, tag, epoch):
    rows = []
    was_training = net.training
    net.eval()
    with torch.no_grad():
        for thr in thresholds:
            eval_mIoU = mIoU()
            eval_PD_FA = PD_FA()
            pred_fg, gt_fg = [], []
            for img, gt_mask, size, _ in eval_loader:
                h, w = _image_hw(size)
                pred = _final_pred(net.forward(img.cuda()))
                pred = pred[:, :, :h, :w]
                gt_mask = gt_mask[:, :, :h, :w]
                eval_mIoU.update((pred > thr).cpu(), gt_mask.cpu())
                eval_PD_FA.update((pred[0, 0, :, :] > thr).cpu(), gt_mask[0, 0, :, :], size)
                pred_fg.append((pred > thr).float().mean().item())
                gt_fg.append((gt_mask > 0.5).float().mean().item())
            r1 = eval_mIoU.get()
            r2 = eval_PD_FA.get()
            rows.append({
                'epoch': epoch, 'split': tag, 'threshold': thr,
                'pixAcc': _as_float(r1[0]), 'mIoU': _as_float(r1[1]),
                'PD': _as_float(r2[0]), 'FA': _as_float(r2[1]),
                'pred_fg_ratio': float(np.mean(pred_fg)), 'gt_fg_ratio': float(np.mean(gt_fg)),
            })
    if was_training:
        net.train()
    sweep_path = os.path.join(opt.run_save_dir, f'threshold_sweep_{tag}_epoch{epoch}.csv')
    with open(sweep_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log = f'THRESHOLD_SWEEP epoch={epoch} split={tag} -> {sweep_path}'
    print(log, flush=True)
    opt.f.write(log + '\n')
    for row in rows:
        line = (f"  thr={row['threshold']:.2f} mIoU={row['mIoU']:.6f} PD={row['PD']:.4f} "
                f"FA={row['FA']:.6f} pred_fg={row['pred_fg_ratio']:.6f}")
        print(line, flush=True)
        opt.f.write(line + '\n')
    opt.f.flush()
    return rows


def _save_val_visualizations(net, eval_loader, epoch, n=4):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    viz_dir = os.path.join(opt.run_save_dir, 'viz', f'epoch_{epoch:04d}')
    os.makedirs(viz_dir, exist_ok=True)
    was_training = net.training
    net.eval()
    saved = 0
    with torch.no_grad():
        for img, gt_mask, size, sid in eval_loader:
            if saved >= n:
                break
            h, w = _image_hw(size)
            pred = _final_pred(net.forward(img.cuda()))
            pred = pred[:, :, :h, :w]
            gt_mask = gt_mask[:, :, :h, :w]
            p = pred[0, 0].cpu().numpy()
            g = gt_mask[0, 0].numpy()
            inp = img[0, 0].numpy()
            bin_p = (p > opt.threshold).astype(np.float32)
            sid_str = sid[0] if isinstance(sid, (list, tuple)) else sid
            fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
            axes[0].imshow(inp, cmap='gray'); axes[0].set_title('input')
            axes[1].imshow(g, cmap='gray', vmin=0, vmax=1); axes[1].set_title(f'GT fg={g.mean():.5f}')
            axes[2].imshow(p, cmap='hot', vmin=0, vmax=1); axes[2].set_title(f'pred mean={p.mean():.4f}')
            axes[3].imshow(bin_p, cmap='gray'); axes[3].set_title(f'bin@{opt.threshold}')
            fig.suptitle(str(sid_str))
            for ax in axes:
                ax.axis('off')
            fig.tight_layout()
            fig.savefig(os.path.join(viz_dir, f'{saved:02d}_{sid_str}.png'), dpi=120)
            plt.close(fig)
            saved += 1
    if was_training:
        net.train()
    return viz_dir


def _append_record(epoch, split, loss_value, results1, results2, lr=None, extra=None):
    extra = extra or {}
    with open(opt.run_record_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            time.strftime('%Y-%m-%d %H:%M:%S'),
            opt.run_name,
            opt.dataset_name,
            opt.model_name,
            epoch,
            split,
            '' if loss_value is None else _as_float(loss_value),
            '' if results1 is None else _as_float(results1[0]),
            '' if results1 is None else _as_float(results1[1]),
            '' if results2 is None else _as_float(results2[0]),
            '' if results2 is None else _as_float(results2[1]),
            '' if lr is None else _as_float(lr),
            extra.get('pred_fg_ratio', ''),
            extra.get('gt_fg_ratio', ''),
            extra.get('pred_cc', ''),
            extra.get('gt_cc', ''),
        ])


def _is_full_validation(epoch):
    if opt.val_max_samples <= 0:
        return True
    if opt.full_val_interval <= 0:
        return False
    return epoch % opt.full_val_interval == 0


def _should_validate(epoch):
    return epoch >= opt.begin_val and epoch % opt.val_interval == 0


def _should_run_threshold_sweep(epoch, is_full_val):
    return (
        is_full_val and
        opt.threshold_sweep_interval > 0 and
        epoch % opt.threshold_sweep_interval == 0
    )


def _make_eval_loader(split_name, phase, max_samples=0):
    eval_set = TestSetLoader(opt.dataset_dir, opt.dataset_name, opt.dataset_name,
                             img_norm_cfg=opt.img_norm_cfg,
                             split_name=split_name, phase=phase)
    total = len(eval_set.test_list)
    if max_samples > 0:
        eval_set.test_list = eval_set.test_list[:max_samples]
    return DataLoader(dataset=eval_set, num_workers=0, batch_size=1, shuffle=False), len(eval_set), total


def _evaluate_split(net, split_name, phase, epoch, writer, tag, lr=None, save_viz=False, max_samples=0):
    eval_loader, n_eval, n_total = _make_eval_loader(split_name, phase, max_samples=max_samples)
    was_training = net.training
    net.eval()
    pred_fg_list, gt_fg_list, pred_cc_list, gt_cc_list = [], [], [], []
    with torch.no_grad():
        eval_mIoU = mIoU()
        eval_PD_FA = PD_FA()
        losses = []
        iterator = eval_loader
        if opt.progress:
            iterator = tqdm(eval_loader, desc=f'{tag} e{epoch}', leave=False, dynamic_ncols=True)
        for img, gt_mask, size, _ in iterator:
            h, w = _image_hw(size)
            img = img.cuda()
            pred = _final_pred(net.forward(img))
            pred = pred[:, :, :h, :w]
            gt_mask = gt_mask[:, :, :h, :w]
            loss = net.loss(pred, gt_mask.cuda())
            losses.append(loss.detach().cpu())
            eval_mIoU.update((pred > opt.threshold).cpu(), gt_mask.cpu())
            eval_PD_FA.update((pred[0, 0, :, :] > opt.threshold).cpu(), gt_mask[0, 0, :, :], size)
            batch_stats = _collect_batch_stats(pred, gt_mask, opt.threshold)
            pred_fg_list.append(batch_stats['pred_fg_ratio'])
            gt_fg_list.append(batch_stats['gt_fg_ratio'])
            pred_cc_list.append(batch_stats['pred_cc'])
            gt_cc_list.append(batch_stats['gt_cc'])

        loss_value = float(np.array(losses).mean()) if losses else 0.0
        results1 = eval_mIoU.get()
        results2 = eval_PD_FA.get()

    extra = {
        'pred_fg_ratio': float(np.mean(pred_fg_list)) if pred_fg_list else 0.0,
        'gt_fg_ratio': float(np.mean(gt_fg_list)) if gt_fg_list else 0.0,
        'pred_cc': float(np.mean(pred_cc_list)) if pred_cc_list else 0.0,
        'gt_cc': float(np.mean(gt_cc_list)) if gt_cc_list else 0.0,
    }

    if writer is not None:
        writer.add_scalar(f'{tag}_mIoU', results1[-1], epoch)
        writer.add_scalar(f'{tag}_loss', loss_value, epoch)
        writer.add_scalar(f'{tag}_PD', results2[0], epoch)
        writer.add_scalar(f'{tag}_FA', results2[1], epoch)
        writer.add_scalar(f'{tag}_pred_fg_ratio', extra['pred_fg_ratio'], epoch)
        writer.add_scalar(f'{tag}_gt_fg_ratio', extra['gt_fg_ratio'], epoch)
        writer.add_scalar(f'{tag}_pred_cc', extra['pred_cc'], epoch)
        writer.add_scalar(f'{tag}_gt_cc', extra['gt_cc'], epoch)

    log = (
        f'{tag.upper()}_RECORD Epoch---{epoch}, samples---{n_eval}/{n_total}, {tag}_loss---{loss_value:.6f}, '
        f'pixAcc---{_as_float(results1[0]):.6f}, mIoU---{_as_float(results1[1]):.6f}, '
        f'PD---{_as_float(results2[0]):.6f}, FA---{_as_float(results2[1]):.6f}, '
        f'pred_fg---{extra["pred_fg_ratio"]:.6f}, gt_fg---{extra["gt_fg_ratio"]:.6f}, '
        f'pred_cc---{extra["pred_cc"]:.1f}, gt_cc---{extra["gt_cc"]:.1f}'
    )
    print(log, flush=True)
    opt.f.write(log + '\n')
    opt.f.flush()
    _append_record(epoch, tag, loss_value, results1, results2, lr, extra)

    if save_viz:
        viz_dir = _save_val_visualizations(net, eval_loader, epoch)
        opt.f.write(f'VIZ saved -> {viz_dir}\n')
        opt.f.flush()

    if was_training:
        net.train()
    extra['n_eval'] = n_eval
    extra['n_total'] = n_total
    return results1, results2, loss_value, extra


def train():
    train_set = _train_loader_class()(dataset_dir=opt.dataset_dir, dataset_name=opt.dataset_name,
                                      patch_size=opt.patchSize, img_norm_cfg=opt.img_norm_cfg,
                                      split_name=opt.train_split)
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads,
                              batch_size=opt.batchSize, shuffle=True)
    net = Net(model_name=opt.model_name, mode='train').cuda()
    net.apply(weights_init_kaiming)
    net.train()

    epoch_state = 0
    total_loss_list = []
    total_loss_epoch = []

    os.makedirs(opt.run_log_dir, exist_ok=True)
    writer = SummaryWriter(opt.run_log_dir)
    epoch_metrics_path = _init_epoch_metrics_file()

    if opt.resume:
        ckpt = torch.load(opt.resume)
        net.load_state_dict(ckpt['state_dict'])
        epoch_state = ckpt['epoch']
        total_loss_list = ckpt['total_loss']

    if opt.optimizer_name == 'Adam':
        opt.optimizer_settings = {'lr': opt.lr if opt.lr is not None else 0.001}
        opt.scheduler_name = 'CosineAnnealingLR'
        opt.scheduler_settings = {'epochs': opt.epochs, 'eta_min': 1e-5, 'last_epoch': -1}
    elif opt.optimizer_name == 'Adagrad':
        opt.optimizer_settings = {'lr': opt.lr if opt.lr is not None else 0.05}
        opt.scheduler_name = 'CosineAnnealingLR'
        opt.scheduler_settings = {'epochs': opt.epochs, 'eta_min': 1e-5}
    elif opt.optimizer_name == 'AdamW':
        opt.optimizer_settings = {'lr': opt.lr if opt.lr is not None else 0.001, 'betas': (0.9, 0.999), "eps": 1e-8,
                                  "weight_decay": 1e-2, "amsgrad": False}
        opt.scheduler_name = 'CosineAnnealingLR'
        opt.scheduler_settings = {'epochs': opt.epochs, 'T_max': 50, 'eta_min': 1e-5, 'last_epoch': -1}
    else:
        raise ValueError(f'Unsupported optimizer: {opt.optimizer_name}')

    opt.f.write(f'training config: loss_mode={opt.loss_mode} lr={opt.optimizer_settings["lr"]} '
                f'pos_weight={opt.pos_weight} threshold={opt.threshold}\n')
    opt.f.flush()

    opt.nEpochs = opt.scheduler_settings['epochs']
    optimizer, scheduler = get_optimizer(net, opt.optimizer_name, opt.scheduler_name,
                                         opt.optimizer_settings, opt.scheduler_settings)

    best_mIOU = (0.0, -1.0)
    best_Pd = (0.0, 0.0)
    best_epoch = 0
    epochs_without_improve = 0
    full_val_loader_cache = None

    opt.f.write(
        f'val schedule: begin={opt.begin_val}, interval={opt.val_interval}, '
        f'quick_samples={opt.val_max_samples}, full_every={opt.full_val_interval}, '
        f'test_every={opt.test_interval}, sweep_every={opt.threshold_sweep_interval}\n'
    )
    opt.f.flush()

    epoch_bar = tqdm(range(epoch_state, opt.nEpochs), desc='epochs', dynamic_ncols=True) if opt.progress else range(epoch_state, opt.nEpochs)
    for idx_epoch in epoch_bar:
        epoch = idx_epoch + 1
        net.train()
        train_loss = None
        batch_iter = train_loader
        if opt.progress:
            batch_iter = tqdm(train_loader, desc=f'train e{epoch}', leave=False, dynamic_ncols=True)
        for img, gt_mask in batch_iter:
            img, gt_mask = img.cuda(), gt_mask.cuda()
            if img.shape[0] == 1:
                continue
            preds = net.forward(img)
            loss = net.loss(preds, gt_mask)
            total_loss_epoch.append(loss.detach().cpu())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        if epoch % opt.every_print == 0:
            train_loss = float(np.array(total_loss_epoch).mean()) if total_loss_epoch else 0.0
            total_loss_list.append(train_loss)
            train_log = time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f, lr---%f' % (epoch, train_loss, lr)
            print(train_log, flush=True)
            opt.f.write(train_log + '\n')
            opt.f.flush()
            _append_record(epoch, 'train', train_loss, None, None, lr)
            total_loss_epoch = []
            writer.add_scalar('loss', train_loss, epoch)
            writer.add_scalar('lr', lr, epoch)

        should_val = _should_validate(epoch)
        val_extra = {}
        if should_val:
            is_full_val = _is_full_validation(epoch)
            max_samples = 0 if is_full_val else opt.val_max_samples
            val_tag = 'val' if is_full_val else 'val_quick'
            save_viz = is_full_val and opt.viz_interval > 0 and epoch % opt.viz_interval == 0
            results1, results2, val_loss, val_extra = _evaluate_split(
                net, opt.val_split, 'val', epoch, writer, val_tag, lr,
                save_viz=save_viz, max_samples=max_samples)
            best_updated = False
            if is_full_val and _as_float(results1[1]) > _as_float(best_mIOU[1]):
                best_mIOU = results1
                best_Pd = results2
                best_epoch = epoch
                epochs_without_improve = 0
                best_updated = True
                print('------save the best model epoch', opt.model_name, '_%d ------' % epoch, flush=True)
                opt.f.write("the best model epoch \t" + str(epoch) + '\n')
                print("pixAcc, mIoU:\t" + str(best_mIOU), flush=True)
                print("valloss:\t" + str(val_loss), flush=True)
                print("PD, FA:\t" + str(best_Pd), flush=True)
                opt.f.write("pixAcc, mIoU:\t" + str(best_mIOU) + '\n')
                opt.f.write("PD, FA:\t" + str(best_Pd) + '\n')
                opt.f.flush()
                save_pth = os.path.join(opt.run_save_dir, opt.model_name + '_' + str(epoch) + '_best.pth.tar')
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': net.state_dict(),
                    'total_loss': total_loss_list,
                    'experiment_name': opt.run_name,
                    'loss_mode': opt.loss_mode,
                    'threshold': opt.threshold,
                }, save_pth)
            elif is_full_val:
                epochs_without_improve += 1

            if _should_run_threshold_sweep(epoch, is_full_val):
                if full_val_loader_cache is None:
                    full_val_loader_cache, _, _ = _make_eval_loader(opt.val_split, 'val', max_samples=0)
                _run_threshold_sweep(
                    net, full_val_loader_cache,
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    'val', epoch)

            _append_epoch_metrics(epoch_metrics_path, [
                epoch,
                train_loss if epoch % opt.every_print == 0 else '',
                lr,
                'full' if is_full_val else 'quick',
                val_extra.get('n_eval', ''),
                val_loss,
                _as_float(results1[1]),
                _as_float(results2[0]),
                _as_float(results2[1]),
                val_extra.get('pred_fg_ratio', ''),
                val_extra.get('gt_fg_ratio', ''),
                val_extra.get('pred_cc', ''),
                val_extra.get('gt_cc', ''),
                int(best_updated),
            ])
            if opt.progress and hasattr(epoch_bar, 'set_postfix'):
                epoch_bar.set_postfix(
                    loss=f'{train_loss:.3f}' if train_loss is not None else '-',
                    val=val_tag,
                    mIoU=f'{_as_float(results1[1]):.4f}',
                    PD=f'{_as_float(results2[0]):.3f}',
                    pred_fg=f'{val_extra.get("pred_fg_ratio", 0):.5f}',
                    refresh=False,
                )

        if opt.test_interval > 0 and epoch % opt.test_interval == 0:
            _evaluate_split(net, opt.test_split, 'test', epoch, writer, 'test', lr, max_samples=0)

        if epoch % opt.every_save_pth == 0:
            save_pth = os.path.join(opt.run_save_dir, opt.model_name + '_' + str(epoch) + '.pth.tar')
            save_checkpoint({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'total_loss': total_loss_list,
                'experiment_name': opt.run_name,
            }, save_pth)

        if (should_val and _is_full_validation(epoch) and opt.early_stop_patience > 0
                and epoch >= opt.min_epochs and epochs_without_improve >= opt.early_stop_patience):
            stop_log = (
                f'EARLY_STOP Epoch---{epoch}, best_epoch---{best_epoch}, '
                f'best_mIoU---{_as_float(best_mIOU[1]):.6f}, patience---{opt.early_stop_patience}'
            )
            print(stop_log, flush=True)
            opt.f.write(stop_log + '\n')
            opt.f.flush()
            break

    final_epoch = epoch if 'epoch' in locals() else epoch_state
    save_pth = os.path.join(opt.run_save_dir, opt.model_name + '_' + str(final_epoch) + '_last.pth.tar')
    save_checkpoint({
        'epoch': final_epoch,
        'state_dict': net.state_dict(),
        'total_loss': total_loss_list,
        'experiment_name': opt.run_name,
    }, save_pth)
    done_log = (
        f'EXPERIMENT_DONE {opt.run_name}, final_epoch---{final_epoch}, '
        f'best_epoch---{best_epoch}, best_mIoU---{_as_float(best_mIOU[1]):.6f}'
    )
    print(done_log, flush=True)
    opt.f.write(done_log + '\n')
    opt.f.flush()
    writer.close()


def test(save_pth):
    net = Net(model_name=opt.model_name, mode='test').cuda()
    ckpt = torch.load(save_pth)
    net.load_state_dict(ckpt['state_dict'])
    return _evaluate_split(net, opt.test_split, 'test', ckpt.get('epoch', 0), None, 'test_checkpoint')


def save_checkpoint(state, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state, save_path)
    return save_path


class Net(nn.Module):
    def __init__(self, model_name, mode):
        super(Net, self).__init__()
        self.model_name = model_name
        self.cal_loss = nn.BCELoss(reduction='mean')
        if model_name == 'SCTransNet':
            self.model = SCTransNet(
                config_vit,
                mode=mode,
                deepsuper=True,
                use_prior=opt.use_prior,
                prior_gamma_init=opt.prior_gamma_init,
                use_aux=opt.use_aux,
            )

    def forward(self, img):
        return self.model(img)

    def _soft_iou_loss(self, pred, gt_masks):
        inter = (pred * gt_masks).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + gt_masks.sum(dim=(1, 2, 3)) - inter
        return (1.0 - (inter + 1.0) / (union + 1.0)).mean()

    def _dice_loss(self, pred, gt_masks):
        inter = (pred * gt_masks).sum(dim=(1, 2, 3))
        denom = pred.sum(dim=(1, 2, 3)) + gt_masks.sum(dim=(1, 2, 3))
        return (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()

    def _weighted_bce(self, pred, gt_masks):
        pos_w = opt.pos_weight
        if pos_w <= 0:
            fg = gt_masks.sum()
            bg = gt_masks.numel() - fg
            pos_w = float(bg / (fg + 1.0))
            pos_w = min(max(pos_w, 1.0), 500.0)
        weight = torch.ones_like(gt_masks)
        weight = torch.where(gt_masks > 0.5, weight * pos_w, weight)
        pred = pred.clamp(1e-6, 1.0 - 1e-6)
        bce = F.binary_cross_entropy(pred, gt_masks, weight=weight, reduction='mean')
        return bce

    def _tversky_loss(self, pred, gt_masks, alpha=None, beta=None):
        alpha = opt.tversky_alpha if alpha is None else alpha
        beta = opt.tversky_beta if beta is None else beta
        tp = (pred * gt_masks).sum(dim=(1, 2, 3))
        fp = (pred * (1.0 - gt_masks)).sum(dim=(1, 2, 3))
        fn = ((1.0 - pred) * gt_masks).sum(dim=(1, 2, 3))
        tversky = (tp + 1.0) / (tp + alpha * fp + beta * fn + 1.0)
        return (1.0 - tversky).mean()

    def _focal_tversky_loss(self, pred, gt_masks):
        tv = self._tversky_loss(pred, gt_masks)
        return torch.pow(tv, opt.focal_gamma)

    def _seg_loss_single(self, pred, gt_masks):
        if opt.loss_mode == 'weighted_bce':
            return self._weighted_bce(pred, gt_masks)
        if opt.loss_mode == 'tversky':
            return self._tversky_loss(pred, gt_masks)
        if opt.loss_mode == 'focal_tversky':
            return self._focal_tversky_loss(pred, gt_masks)
        bce = self.cal_loss(pred, gt_masks)
        if opt.loss_mode == 'bce':
            return bce
        if opt.loss_mode == 'bce_softiou':
            return opt.bce_weight * bce + opt.softiou_weight * self._soft_iou_loss(pred, gt_masks)
        if opt.loss_mode == 'bce_dice':
            return opt.bce_weight * bce + opt.dice_weight * self._dice_loss(pred, gt_masks)
        raise ValueError(f'Unsupported loss mode: {opt.loss_mode}')

    def _seg_loss(self, seg_preds, gt_masks):
        if isinstance(seg_preds, (tuple, list)):
            loss_total = 0
            for pred in seg_preds:
                loss_total = loss_total + self._seg_loss_single(pred, gt_masks)
            return loss_total
        return self._seg_loss_single(seg_preds, gt_masks)

    def _center_heatmap(self, gt_masks):
        if opt.center_mode == 'gaussian':
            return self._gaussian_center_heatmap(gt_masks)
        return F.avg_pool2d(gt_masks, kernel_size=5, stride=1, padding=2).clamp(0.0, 1.0)

    def _gaussian_center_heatmap(self, gt_masks):
        masks = gt_masks.detach().cpu().numpy()
        heatmaps = []
        for sample in masks:
            mask = (sample[0] > 0.5).astype(np.uint8)
            heat = np.zeros_like(mask, dtype=np.float32)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
            for label_idx in range(1, num_labels):
                area = stats[label_idx, cv2.CC_STAT_AREA]
                if area <= 0:
                    continue
                cx, cy = centroids[label_idx]
                bbox_w = stats[label_idx, cv2.CC_STAT_WIDTH]
                bbox_h = stats[label_idx, cv2.CC_STAT_HEIGHT]
                cx_i = int(round(cx))
                cy_i = int(round(cy))
                if area <= opt.center_dirac_area or max(bbox_w, bbox_h) <= 2:
                    if 0 <= cy_i < mask.shape[0] and 0 <= cx_i < mask.shape[1]:
                        heat[cy_i, cx_i] = 1.0
                    continue
                sigma = min(
                    max(min(bbox_w, bbox_h) / 3.0, opt.center_min_sigma),
                    opt.center_max_sigma,
                )
                radius = int(np.ceil(3.0 * sigma))
                x0 = max(0, cx_i - radius)
                x1 = min(mask.shape[1], cx_i + radius + 1)
                y0 = max(0, cy_i - radius)
                y1 = min(mask.shape[0], cy_i + radius + 1)
                yy, xx = np.mgrid[y0:y1, x0:x1]
                gaussian = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
                heat[y0:y1, x0:x1] = np.maximum(heat[y0:y1, x0:x1], gaussian.astype(np.float32))
            heatmaps.append(heat)
        heatmaps = np.stack(heatmaps, axis=0)[:, np.newaxis, :, :]
        return torch.from_numpy(heatmaps).to(device=gt_masks.device, dtype=gt_masks.dtype)

    def _edge_map(self, gt_masks):
        dilated = F.max_pool2d(gt_masks, kernel_size=3, stride=1, padding=1)
        eroded = 1.0 - F.max_pool2d(1.0 - gt_masks, kernel_size=3, stride=1, padding=1)
        return (dilated - eroded).clamp(0.0, 1.0)

    def _hard_false_alarm_loss(self, pred, gt_masks):
        background = gt_masks < 0.5
        if background.sum() == 0:
            return pred.sum() * 0.0
        bg_pred = pred[background]
        k = max(1, int(bg_pred.numel() * 0.01))
        return torch.topk(bg_pred, k).values.mean()

    def _distance_repulsion_loss(self, pred, gt_masks):
        masks = gt_masks.detach().cpu().numpy()
        weight_maps = []
        for sample in masks:
            mask = (sample[0] > 0.5).astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(mask, 8)
            if num_labels <= 2:
                weight_maps.append(np.zeros_like(mask, dtype=np.float32))
                continue

            distances = []
            for label_idx in range(1, num_labels):
                component = (labels == label_idx).astype(np.uint8)
                distances.append(cv2.distanceTransform(1 - component, cv2.DIST_L2, 3))
            nearest = np.partition(np.stack(distances, axis=0), kth=1, axis=0)[:2]
            d1, d2 = nearest[0], nearest[1]
            dist_sum = d1 + d2
            weight = opt.repulsion_w0 * np.exp(-(dist_sum ** 2) / (2.0 * opt.repulsion_sigma ** 2))
            if opt.valley_max_distance > 0:
                weight[dist_sum > opt.valley_max_distance] = 0.0
            weight[mask > 0] = 0.0
            weight_maps.append(weight.astype(np.float32))

        weight_maps = np.stack(weight_maps, axis=0)[:, np.newaxis, :, :]
        weights = torch.from_numpy(weight_maps).to(device=pred.device, dtype=pred.dtype)
        if weights.sum() < 1:
            return pred.sum() * 0.0
        pred = pred.clamp(1e-6, 1.0 - 1e-6)
        bce = F.binary_cross_entropy(pred, gt_masks, reduction='none')
        return (bce * weights).sum() / weights.sum()

    def loss(self, preds, gt_masks):
        if isinstance(preds, dict):
            loss_total = self._seg_loss(preds['seg'], gt_masks)
            final_pred = _final_pred(preds)
            if opt.center_weight > 0:
                loss_total = loss_total + opt.center_weight * self.cal_loss(preds['center'], self._center_heatmap(gt_masks))
            if opt.edge_weight > 0:
                loss_total = loss_total + opt.edge_weight * self.cal_loss(preds['edge'], self._edge_map(gt_masks))
            if opt.fa_weight > 0:
                loss_total = loss_total + opt.fa_weight * self._hard_false_alarm_loss(final_pred, gt_masks)
            if opt.valley_weight > 0:
                loss_total = loss_total + opt.valley_weight * self._distance_repulsion_loss(final_pred, gt_masks)
            return loss_total

        loss_total = self._seg_loss(preds, gt_masks)
        final_pred = _final_pred(preds)
        if opt.fa_weight > 0:
            loss_total = loss_total + opt.fa_weight * self._hard_false_alarm_loss(final_pred, gt_masks)
        if opt.valley_weight > 0:
            loss_total = loss_total + opt.valley_weight * self._distance_repulsion_loss(final_pred, gt_masks)
        return loss_total


if __name__ == '__main__':
    for dataset_name in opt.dataset_names:
        opt.dataset_name = dataset_name
        for model_name in opt.model_names:
            opt.model_name = model_name
            stamp = time.strftime('%Y%m%d_%H%M%S')
            opt.run_name = opt.experiment_name or opt.model_name
            opt.run_save_dir = os.path.join(opt.save, opt.dataset_name, opt.run_name)
            opt.run_log_dir = os.path.join(opt.log_dir, opt.run_name + '_' + stamp)
            opt.run_record_dir = os.path.join(opt.record_dir, opt.dataset_name)
            opt.run_record_path = os.path.join(opt.run_record_dir, opt.run_name + '_' + stamp + '.csv')
            os.makedirs(opt.save, exist_ok=True)
            os.makedirs(opt.run_save_dir, exist_ok=True)
            _init_record_file()
            log_path = os.path.join(opt.save, opt.dataset_name + '_' + opt.run_name + '_' + stamp + '.txt')
            opt.f = open(log_path, 'w')
            print(opt.dataset_name + '\t' + opt.model_name + '\t' + opt.run_name, flush=True)
            opt.f.write('options:\t' + str(vars(opt)) + '\n')
            opt.f.flush()
            train()
            print('\n', flush=True)
            opt.f.close()
