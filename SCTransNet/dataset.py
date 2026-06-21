from utils import *
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def _is_dense_sirst_root(root):
    return (
        os.path.isdir(os.path.join(root, 'PNGImages')) and
        os.path.isdir(os.path.join(root, 'SIRST', 'BinaryMask')) and
        os.path.isdir(os.path.join(root, 'Splits'))
    )


def _unique_paths(paths):
    seen = set()
    result = []
    for path in paths:
        norm_path = os.path.normcase(os.path.abspath(path))
        if norm_path not in seen:
            seen.add(norm_path)
            result.append(path)
    return result


def _resolve_dataset_layout(dataset_dir, dataset_name):
    dense_candidates = _unique_paths([
        os.path.join(dataset_dir, dataset_name, 'SIRSTdevkit'),
        os.path.join(dataset_dir, dataset_name),
        os.path.join(dataset_dir, 'DenseSIRST', 'SIRSTdevkit'),
        os.path.join(dataset_dir, 'DenseSIRST'),
        dataset_dir,
    ])
    for root in dense_candidates:
        if _is_dense_sirst_root(root):
            return {
                'type': 'dense_sirst',
                'root': root,
                'image_dir': os.path.join(root, 'PNGImages'),
                'mask_dir': os.path.join(root, 'SIRST', 'BinaryMask'),
                'split_dir': os.path.join(root, 'Splits'),
            }

    root = os.path.join(dataset_dir, dataset_name)
    return {
        'type': 'legacy',
        'root': root,
        'image_dir': os.path.join(root, 'images'),
        'mask_dir': os.path.join(root, 'masks'),
        'split_dir': os.path.join(root, 'img_idx'),
    }


def _default_split_name(layout, dataset_name, phase):
    if layout['type'] == 'dense_sirst':
        if phase == 'train':
            return 'train_v2'
        if phase == 'val':
            return 'val_v2'
        return 'test_v2'
    return f'{phase}_{dataset_name}'


def _split_path(layout, dataset_name, phase, split_name=None):
    file_name = split_name or _default_split_name(layout, dataset_name, phase)
    if not file_name.endswith('.txt'):
        file_name = file_name + '.txt'
    return os.path.join(layout['split_dir'], file_name)


def _read_split(layout, dataset_name, phase, split_name=None):
    path = _split_path(layout, dataset_name, phase, split_name)
    with open(path, 'r') as f:
        return [line.strip() for line in f.read().splitlines() if line.strip()]


def _open_with_candidates(base_dir, sample_id, suffixes):
    for suffix in suffixes:
        path = os.path.join(base_dir, sample_id + suffix)
        if os.path.exists(path):
            return Image.open(path)
    raise FileNotFoundError(f'Cannot find sample {sample_id} in {base_dir}')


def _load_image_and_mask(layout, sample_id):
    img = _open_with_candidates(layout['image_dir'], sample_id, ['.png', '.bmp', '.jpg']).convert('I')
    if layout['type'] == 'dense_sirst':
        mask = _open_with_candidates(layout['mask_dir'], sample_id, ['_pixels0.png', '.png', '_pixels0.bmp', '.bmp'])
    else:
        mask = _open_with_candidates(layout['mask_dir'], sample_id, ['.png', '.bmp', '.jpg'])
    return img, mask


class _TrainSetLoaderBase(Dataset):
    add_noise = False
    add_gamma = False

    def __init__(self, dataset_dir, dataset_name, patch_size, img_norm_cfg=None, split_name=None):
        super().__init__()
        self.dataset_name = dataset_name
        self.layout = _resolve_dataset_layout(dataset_dir, dataset_name)
        self.dataset_dir = self.layout['root']
        self.patch_size = patch_size
        self.train_list = _read_split(self.layout, dataset_name, 'train', split_name)
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir, split_name or 'train_v2')
        else:
            self.img_norm_cfg = img_norm_cfg
        self.tranform = augumentation()

    def __getitem__(self, idx):
        img, mask = _load_image_and_mask(self.layout, self.train_list[idx])
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        if self.add_noise:
            img += np.random.normal(0, 0.03)

        if self.add_gamma:
            minm = img.min()
            rng = img.max() - minm
            if rng > 0:
                gamma = np.random.uniform(0.5, 1.6)
                img = np.power((img - minm) / rng, gamma) * rng + minm

        img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5)
        img_patch, mask_patch = self.tranform(img_patch, mask_patch)
        img_patch, mask_patch = img_patch[np.newaxis, :], mask_patch[np.newaxis, :]
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))
        return img_patch, mask_patch

    def __len__(self):
        return len(self.train_list)


class TrainSetLoader(_TrainSetLoaderBase):
    pass


class TrainSetLoader02(_TrainSetLoaderBase):
    add_noise = True


class TrainSetLoader03(_TrainSetLoaderBase):
    add_gamma = True


class TrainSetLoader04(_TrainSetLoaderBase):
    add_noise = True
    add_gamma = True


class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, img_norm_cfg=None,
                 split_name=None, phase='test'):
        super().__init__()
        self.layout = _resolve_dataset_layout(dataset_dir, test_dataset_name)
        self.dataset_dir = self.layout['root']
        self.test_list = _read_split(self.layout, test_dataset_name, phase, split_name)
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg

    def __getitem__(self, idx):
        img, mask = _load_image_and_mask(self.layout, self.test_list[idx])

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        h, w = img.shape

        img = PadImg(img)
        mask = PadImg(mask)

        img, mask = img[np.newaxis, :], mask[np.newaxis, :]

        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        if img.size() != mask.size():
            print('111')
        return img, mask, [h, w], self.test_list[idx]

    def __len__(self):
        return len(self.test_list)


class EvalSetLoader(Dataset):
    def __init__(self, dataset_dir, mask_pred_dir, test_dataset_name, model_name,
                 split_name=None, phase='test'):
        super().__init__()
        self.layout = _resolve_dataset_layout(dataset_dir, test_dataset_name)
        self.dataset_dir = self.layout['root']
        self.mask_pred_dir = mask_pred_dir
        self.test_dataset_name = test_dataset_name
        self.model_name = model_name
        self.test_list = _read_split(self.layout, test_dataset_name, phase, split_name)

    def __getitem__(self, idx):
        mask_pred = Image.open(os.path.join(
            self.mask_pred_dir, self.test_dataset_name, self.model_name, self.test_list[idx] + '.png'))
        _, mask_gt = _load_image_and_mask(self.layout, self.test_list[idx])

        mask_pred = np.array(mask_pred, dtype=np.float32) / 255.0
        mask_gt = np.array(mask_gt, dtype=np.float32) / 255.0

        if len(mask_pred.shape) == 3:
            mask_pred = mask_pred[:, :, 0]
        if len(mask_gt.shape) == 3:
            mask_gt = mask_gt[:, :, 0]

        h, w = mask_pred.shape

        mask_pred, mask_gt = mask_pred[np.newaxis, :], mask_gt[np.newaxis, :]

        mask_pred = torch.from_numpy(np.ascontiguousarray(mask_pred))
        mask_gt = torch.from_numpy(np.ascontiguousarray(mask_gt))
        return mask_pred, mask_gt, [h, w]

    def __len__(self):
        return len(self.test_list)


class augumentation(object):
    def __call__(self, input, target):
        if random.random() < 0.5:
            input = input[::-1, :]
            target = target[::-1, :]
        if random.random() < 0.5:
            input = input[:, ::-1]
            target = target[:, ::-1]
        if random.random() < 0.5:
            input = input.transpose(1, 0)
            target = target.transpose(1, 0)
        return input, target
