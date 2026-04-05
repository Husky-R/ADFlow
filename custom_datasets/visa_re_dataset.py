import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms as T
import cv2
from torch.utils.data import Dataset

# VISA类别列表
VISA_CLASS_NAMES = [
    'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
    'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'
]


class VisADataset(Dataset):

    def __init__(self, c, is_train=False):

        assert c.class_name in VISA_CLASS_NAMES, \
            f'class_name: {c.class_name}, 应属于VISA类别列表: {VISA_CLASS_NAMES}'

        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.cropsize = c.crp_size
        self.x, self.y, self.mask = self.load_dataset_folder()

        assert len(self.x) == len(self.y) == len(self.mask), \
            f"列表长度不一致: x={len(self.x)}, y={len(self.y)}, mask={len(self.mask)}"

        print(f"[VisADataset] 初始化完成 - 类别: {self.class_name}, 样本数: {len(self.x)}, 训练集: {self.is_train}")

        if is_train:
            self.transform_x = T.Compose([
                T.Resize(c.img_size, Image.LANCZOS),
                T.CenterCrop(c.crp_size),
                T.ToTensor()
            ])
        else:
            self.transform_x = T.Compose([
                T.Resize(c.img_size, Image.LANCZOS),
                T.CenterCrop(c.crp_size),
                T.ToTensor()
            ])

        self.transform_mask = T.Compose([
            T.Resize(c.img_size, Image.NEAREST),
            T.CenterCrop(c.crp_size),
            T.ToTensor()
        ])

        self.normalize = T.Compose([
            T.Normalize(c.norm_mean, c.norm_std)
        ])

    def __getitem__(self, idx):

        if idx < 0 or idx >= len(self.x):
            raise IndexError(f"索引 {idx} 超出范围 (总样本数: {len(self.x)})")

        x_path, y_label, mask_path = self.x[idx], self.y[idx], self.mask[idx]

        try:
            x = Image.open(x_path)
        except Exception as e:
            x = Image.new('RGB', self.cropsize)
            y_label = 0
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
            return self.normalize(self.transform_x(x)), y_label, mask

        x = self.normalize(self.transform_x(x))


        if y_label == 0:

            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:

            try:
                mask = Image.open(mask_path)
                mask = self.transform_mask(mask)
            except Exception as e:

                mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])

        return x, y_label, mask

    def __len__(self):

        return len(self.x)

    def load_dataset_folder(self):

        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        if not os.path.exists(img_dir):

            return [], [], []

        img_types = sorted(os.listdir(img_dir))
        print(f"[VisADataset] find {len(img_types)} image type: {img_types}")

        for img_type in img_types:
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue

            img_fpath_list = sorted([
                os.path.join(img_type_dir, f)
                for f in os.listdir(img_type_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ])

            print(f"[VisADataset]  '{img_type}' find {len(img_fpath_list)} images")

            if len(img_fpath_list) > 0:
                print(f"[VisADataset] 示例图像: {img_fpath_list[0]}")

            if img_type == 'good':
                y_extend = [0] * len(img_fpath_list)
                mask_extend = [None] * len(img_fpath_list)
            else:
                y_extend = [1] * len(img_fpath_list)

                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [
                    os.path.join(gt_type_dir, img_fname + '.png')
                    for img_fname in img_fname_list
                ]

                # 验证掩码文件是否存在
                missing_masks = [p for p in gt_fpath_list if not os.path.exists(p)]
                if missing_masks:
                    print(f"'{img_type}' lack {len(missing_masks)} files")
                    print(f": {missing_masks[0]}")

                mask_extend = gt_fpath_list

            assert len(y_extend) == len(img_fpath_list), f"{img_type} "
            assert len(mask_extend) == len(img_fpath_list), f"{img_type} "

            x.extend(img_fpath_list)
            y.extend(y_extend)
            mask.extend(mask_extend)

        # 最终验证
        assert len(x) == len(y) == len(mask), \
            f"总长度不匹配: x={len(x)}, y={len(y)}, mask={len(mask)}"

        return list(x), list(y), list(mask)