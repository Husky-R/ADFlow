import os
from PIL import Image
import numpy as np
import torch
from torchvision.io import read_video, write_jpeg
from torch.utils.data import Dataset
from torchvision import transforms as T
import cv2
import numpy as np
import imgaug.augmenters as iaa
from .perlin import rand_perlin_2d_np
from .NSA import patch_ex
import random
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import CIFAR10

# from datasets.datasets_train import get_nomral_dataset
# from datasets.utils import sparse_to_coarse, BaseDataset


__all__ = ('MVTecDataset',  'MVTecPseudoDataset', 'StcDataset', 'CIFAR10Dataset', 'VisADataset')

# URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
MVTEC_CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper', 'cookie', 'splicing_connectors']

class MVTecDataset(Dataset):
    def __init__(self, c, is_train=False):
        assert c.class_name in MVTEC_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name, MVTEC_CLASS_NAMES)
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.cropsize = c.crp_size
        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()
        # set transforms
        if is_train:
            self.transform_x = T.Compose([
                T.Resize(c.img_size, Image.LANCZOS),
                T.CenterCrop(c.crp_size),
                T.ToTensor()])
        # test:
        else:
            self.transform_x = T.Compose([
                T.Resize(c.img_size, Image.LANCZOS),
                T.CenterCrop(c.crp_size),
                T.ToTensor()])
        # mask
        self.transform_mask = T.Compose([
            T.Resize(c.img_size, Image.NEAREST),
            T.CenterCrop(c.crp_size),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(c.norm_mean, c.norm_std)])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        x = Image.open(x)
        if self.class_name in ['zipper', 'screw', 'grid']:  # handle greyscale classes
            x = np.expand_dims(np.array(x), axis=2)
            x = np.concatenate([x, x, x], axis=2)

            x = Image.fromarray(x.astype('uint8')).convert('RGB')
        #
        x = self.normalize(self.transform_x(x))
        #
        if y == 0:
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')
        # print(img_dir)

        img_types = sorted(os.listdir(img_dir)) #['ground_truth', 'license.txt', 'readme.txt', 'test', 'train']
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)


class MVTecPseudoDataset(Dataset):
    def __init__(self, c, is_train=True):
        """
        Mvtec train dataset with anomaly samples.
        Anomaly samples: Pseudo anomaly samples.
        """
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.cropsize = c.crp_size
        self.anomaly_nums = c.num_anomalies
        self.repeat_num = 10
        self.reuse_times = 5
        self.ano_type = 'nsa'

        # load dataset
        self.n_imgs, self.n_labels, self.n_masks, self.a_imgs, self.a_labels, self.a_masks = self.load_dataset_folder()
        self.a_imgs = self.a_imgs * self.repeat_num
        self.a_labels = self.a_labels * self.repeat_num
        self.a_masks = self.a_masks * self.repeat_num

        self.labels = np.array(self.n_labels + self.a_labels)
        self.normal_idx = np.argwhere(self.labels == 0).flatten()
        self.anomaly_idx = np.argwhere(self.labels == 1).flatten()

        # set transforms
        if is_train:
            self.transform_img = T.Compose([
                T.Resize(c.img_size, Image.LANCZOS),
                # T.RandomRotation(5),
                T.CenterCrop(c.crp_size),
                T.ToTensor()])
        # test:
        else:
            self.transform_img = T.Compose([
                T.Resize(c.img_size, Image.LANCZOS),
                T.CenterCrop(c.crp_size),
                T.ToTensor()])
        # mask
        self.transform_mask = T.Compose([
            T.Resize(c.img_size, Image.NEAREST),
            T.CenterCrop(c.crp_size),
            T.ToTensor()])

        #self.anomaly_source_paths = sorted(glob.glob(c.anomaly_source_path + "/*/*.jpg"))

        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        # keep same with the MVTecCopyPasteDataset
        self.transform_img_np = T.Compose([
            T.Resize(c.img_size, Image.LANCZOS),
            # T.RandomRotation(5),
            T.CenterCrop(c.crp_size)])
        self.normalize = T.Compose([T.Normalize(c.norm_mean, c.norm_std)])
        self.normalize_np = T.Compose([T.ToTensor(), T.Normalize(c.norm_mean, c.norm_std)])

    def __len__(self):
        return len(self.n_imgs) + len(self.a_imgs)

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]])
        # aug = iaa.Sequential([self.augmenters[4], self.augmenters[3], self.augmenters[5]])
        return aug

    def augment_image(self, image, anomaly_source_path):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        # anomaly_source_path = '/disk/yxc/datasets/dtd/images/blotchy/blotchy_0069.jpg'
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.cropsize[1], self.cropsize[0]))

        # temp_img = Image.open(anomaly_source_path)
        # temp_img.save('texture_img.jpg')
        anomaly_img_augmented = aug(image=anomaly_source_img)
        # temp_img = cv2.cvtColor(anomaly_img_augmented, cv2.COLOR_BGR2RGB)
        # temp_img = Image.fromarray(temp_img)
        # temp_img.save('texture_img_aug.jpg')
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.cropsize[0], self.cropsize[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)
        # temp_img = augmented_image.astype(np.float32)
        # temp_img = cv2.cvtColor(temp_img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('aug_img.jpg', temp_img, )
        # temp_img = augmented_image.astype(np.uint8)
        # temp_img = Image.fromarray(temp_img)
        # temp_img.save('aug_img.jpg')

        augmented_image = augmented_image.astype(np.float32)
        msk = (perlin_thr).astype(np.float32)
        augmented_image = msk * augmented_image + (1 - msk) * image
        has_anomaly = 1
        if np.sum(msk) == 0:
            has_anomaly = 0
        return augmented_image, msk, has_anomaly

    def transform_image(self, image_path, anomaly_source_path):
        image = Image.open(image_path)
        # image.save('ori_img.jpg')
        if self.class_name in ['zipper', 'screw', 'grid']:  # handle greyscale classes
            image = np.expand_dims(np.asarray(image), axis=2)
            image = np.concatenate([image, image, image], axis=2)
            image = Image.fromarray(image.astype('uint8')).convert('RGB')
        #
        image = self.transform_img_np(image)
        image = np.asarray(image)  # (256, 256, 3)

        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        if do_aug_orig:
            image = self.rot(image=image)

        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
        augmented_image = augmented_image.astype(np.uint8)
        augmented_image = Image.fromarray(augmented_image)
        # augmented_image.save('aug_img.jpg')
        augmented_image = self.normalize_np(augmented_image)
        anomaly_mask = torch.from_numpy(np.transpose(anomaly_mask, (2, 0, 1)))

        return augmented_image, anomaly_mask, has_anomaly

    def __getitem__(self, idx):
        if idx >= len(self.n_imgs):  # anomaly samples
            if self.ano_type == 'perlin':
                n_idx = np.random.randint(len(self.n_imgs))  # get a random normal sample
                anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
                image, mask, label = self.transform_image(self.n_imgs[n_idx],
                                                          self.anomaly_source_paths[anomaly_source_idx])
            if self.ano_type == 'nsa':
                n_idx1 = np.random.randint(len(self.n_imgs))  # get a random normal sample
                n_idx2 = np.random.randint(len(self.n_imgs))
                dst_img = Image.open(self.n_imgs[n_idx1])
                src_img = Image.open(self.n_imgs[n_idx2])
                if self.class_name in ['zipper', 'screw', 'grid']:  # handle greyscale classes
                    dst_img = np.expand_dims(np.asarray(dst_img), axis=2)
                    dst_img = np.concatenate([dst_img, dst_img, dst_img], axis=2)
                    src_img = np.expand_dims(np.asarray(src_img), axis=2)
                    src_img = np.concatenate([src_img, src_img, src_img], axis=2)
                else:
                    dst_img = np.array(dst_img)
                    src_img = np.array(src_img)

                image, mask = patch_ex(dst_img, src_img)
                mask = cv2.resize(mask, dsize=(self.cropsize[1], self.cropsize[0]), interpolation=cv2.INTER_NEAREST)
                image = Image.fromarray(image).convert('RGB')
                image = self.normalize(self.transform_img(image))
                mask = torch.from_numpy(mask).unsqueeze(0).to(torch.float32)  # (1, 256, 256), float32, [0, 1]
                label = 1

            return image, label, mask
        else:
            img, label, mask = self.n_imgs[idx], self.n_labels[idx], self.n_masks[idx]
        img = Image.open(img)
        if self.class_name in ['zipper', 'screw', 'grid']:  # handle greyscale classes
            img = np.expand_dims(np.asarray(img), axis=2)
            img = np.concatenate([img, img, img], axis=2)
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
        #
        img = self.normalize(self.transform_img(img))
        #
        if label == 0:
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return img, label, mask

    def load_dataset_folder(self):
        n_img_paths, n_labels, n_mask_paths = [], [], []  # normal
        a_img_paths, a_labels, a_mask_paths = [], [], []  # abnormal

        img_dir = os.path.join(self.dataset_path, self.class_name, 'test')
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        ano_types = sorted(os.listdir(img_dir))  # anomaly types

        num_ano_types = len(ano_types) - 1
        anomaly_nums_per_type = self.anomaly_nums // num_ano_types
        extra_nums = self.anomaly_nums % num_ano_types
        extra_ano_img_list, extra_ano_gt_list = [], []
        for type_ in ano_types:
            # load images
            img_type_dir = os.path.join(img_dir, type_)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])

            if type_ == 'good':  # normal images
                continue
            else:  # anomaly images
                # randomly choose some anomaly images
                random.shuffle(img_fpath_list)
                a_img_paths.extend(img_fpath_list[:anomaly_nums_per_type])
                a_labels.extend([1] * anomaly_nums_per_type)

                extra_ano_img_list.extend(img_fpath_list[anomaly_nums_per_type:])

                gt_type_dir = os.path.join(gt_dir, type_)
                ano_img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in
                                      img_fpath_list[:anomaly_nums_per_type]]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in ano_img_fname_list]
                a_mask_paths.extend(gt_fpath_list)

                extra_img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in
                                        img_fpath_list[anomaly_nums_per_type:]]
                extra_gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                       for img_fname in extra_img_fname_list]
                extra_ano_gt_list.extend(extra_gt_fpath_list)
        if extra_nums > 0:
            assert len(extra_ano_img_list) == len(extra_ano_gt_list)
            inds = list(range(len(extra_ano_img_list)))
            random.shuffle(inds)
            select_ind = inds[:extra_nums]
            extra_a_img_paths = [extra_ano_img_list[ind] for ind in select_ind]
            extra_a_labels = [1] * extra_nums
            extra_a_mask_paths = [extra_ano_gt_list[ind] for ind in select_ind]
            a_img_paths.extend(extra_a_img_paths)
            a_labels.extend(extra_a_labels)
            a_mask_paths.extend(extra_a_mask_paths)

        # append normal images in train set
        img_dir = os.path.join(self.dataset_path, self.class_name, 'train', 'good')
        img_fpath_list = sorted([os.path.join(img_dir, f)
                                 for f in os.listdir(img_dir)
                                 if f.endswith('.png')])
        n_img_paths.extend(img_fpath_list)
        n_labels.extend([0] * len(img_fpath_list))
        n_mask_paths.extend([None] * len(img_fpath_list))

        return n_img_paths, n_labels, n_mask_paths, a_img_paths, a_labels, a_mask_paths
