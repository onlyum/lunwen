import argparse
import csv
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
parser.add_argument("--begin_val", default=1, type=int)
parser.add_argument("--val_interval", default=1, type=int)
parser.add_argument("--every_test", default=None, type=int, help="Deprecated alias for --val_interval")
parser.add_argument("--test_interval", default=5, type=int)
parser.add_argument("--every_save_pth", default=1000, type=int)
parser.add_argument("--every_print", default=1, type=int)
parser.add_argument("--dataset_dir", default=r'D:/lunwen/dataset')
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

parser.add_argument("--loss_mode", default='bce', choices=['bce', 'bce_softiou', 'bce_dice'])
parser.add_argument("--bce_weight", default=1.0, type=float)
parser.add_argument("--softiou_weight", default=0.0, type=float)
parser.add_argument("--dice_weight", default=0.0, type=float)
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
        ])


def _append_record(epoch, split, loss_value, results1, results2, lr=None):
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
        ])


def _evaluate_split(net, split_name, phase, epoch, writer, tag, lr=None):
    eval_set = TestSetLoader(opt.dataset_dir, opt.dataset_name, opt.dataset_name,
                             img_norm_cfg=opt.img_norm_cfg,
                             split_name=split_name, phase=phase)
    eval_loader = DataLoader(dataset=eval_set, num_workers=0, batch_size=1, shuffle=False)
    was_training = net.training
    net.eval()
    with torch.no_grad():
        eval_mIoU = mIoU()
        eval_PD_FA = PD_FA()
        losses = []
        for img, gt_mask, size, _ in eval_loader:
            h, w = _image_hw(size)
            img = img.cuda()
            pred = _final_pred(net.forward(img))
            pred = pred[:, :, :h, :w]
            gt_mask = gt_mask[:, :, :h, :w]
            loss = net.loss(pred, gt_mask.cuda())
            losses.append(loss.detach().cpu())
            eval_mIoU.update((pred > opt.threshold).cpu(), gt_mask.cpu())
            eval_PD_FA.update((pred[0, 0, :, :] > opt.threshold).cpu(), gt_mask[0, 0, :, :], size)

        loss_value = float(np.array(losses).mean()) if losses else 0.0
        results1 = eval_mIoU.get()
        results2 = eval_PD_FA.get()

    if writer is not None:
        writer.add_scalar(f'{tag}_mIoU', results1[-1], epoch)
        writer.add_scalar(f'{tag}_loss', loss_value, epoch)
        writer.add_scalar(f'{tag}_PD', results2[0], epoch)
        writer.add_scalar(f'{tag}_FA', results2[1], epoch)

    log = (
        f'{tag.upper()}_RECORD Epoch---{epoch}, {tag}_loss---{loss_value:.6f}, '
        f'pixAcc---{_as_float(results1[0]):.6f}, mIoU---{_as_float(results1[1]):.6f}, '
        f'PD---{_as_float(results2[0]):.6f}, FA---{_as_float(results2[1]):.6f}'
    )
    print(log, flush=True)
    opt.f.write(log + '\n')
    opt.f.flush()
    _append_record(epoch, tag, loss_value, results1, results2, lr)

    if was_training:
        net.train()
    return results1, results2, loss_value


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

    if opt.resume:
        ckpt = torch.load(opt.resume)
        net.load_state_dict(ckpt['state_dict'])
        epoch_state = ckpt['epoch']
        total_loss_list = ckpt['total_loss']

    if opt.optimizer_name == 'Adam':
        opt.optimizer_settings = {'lr': 0.001}
        opt.scheduler_name = 'CosineAnnealingLR'
        opt.scheduler_settings = {'epochs': opt.epochs, 'eta_min': 1e-5, 'last_epoch': -1}
    elif opt.optimizer_name == 'Adagrad':
        opt.optimizer_settings = {'lr': 0.05}
        opt.scheduler_name = 'CosineAnnealingLR'
        opt.scheduler_settings = {'epochs': opt.epochs, 'eta_min': 1e-5}
    elif opt.optimizer_name == 'AdamW':
        opt.optimizer_settings = {'lr': 0.001, 'betas': (0.9, 0.999), "eps": 1e-8,
                                  "weight_decay": 1e-2, "amsgrad": False}
        opt.scheduler_name = 'CosineAnnealingLR'
        opt.scheduler_settings = {'epochs': opt.epochs, 'T_max': 50, 'eta_min': 1e-5, 'last_epoch': -1}
    else:
        raise ValueError(f'Unsupported optimizer: {opt.optimizer_name}')

    opt.nEpochs = opt.scheduler_settings['epochs']
    optimizer, scheduler = get_optimizer(net, opt.optimizer_name, opt.scheduler_name,
                                         opt.optimizer_settings, opt.scheduler_settings)

    best_mIOU = (0.0, -1.0)
    best_Pd = (0.0, 0.0)
    best_epoch = 0
    epochs_without_improve = 0

    for idx_epoch in range(epoch_state, opt.nEpochs):
        epoch = idx_epoch + 1
        net.train()
        for img, gt_mask in train_loader:
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

        should_val = epoch >= opt.begin_val and epoch % opt.val_interval == 0
        if should_val:
            results1, results2, val_loss = _evaluate_split(net, opt.val_split, 'val', epoch, writer, 'val', lr)
            if _as_float(results1[1]) > _as_float(best_mIOU[1]):
                best_mIOU = results1
                best_Pd = results2
                best_epoch = epoch
                epochs_without_improve = 0
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
                }, save_pth)
            else:
                epochs_without_improve += 1

        if opt.test_interval > 0 and epoch % opt.test_interval == 0:
            _evaluate_split(net, opt.test_split, 'test', epoch, writer, 'test', lr)

        if epoch % opt.every_save_pth == 0:
            save_pth = os.path.join(opt.run_save_dir, opt.model_name + '_' + str(epoch) + '.pth.tar')
            save_checkpoint({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'total_loss': total_loss_list,
                'experiment_name': opt.run_name,
            }, save_pth)

        if (should_val and opt.early_stop_patience > 0 and epoch >= opt.min_epochs
                and epochs_without_improve >= opt.early_stop_patience):
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

    def _seg_loss_single(self, pred, gt_masks):
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
