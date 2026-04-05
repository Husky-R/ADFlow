import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torch
from torchvision import transforms
import cv2
import glob
class Padding2Resize():
    def __init__(self, pad_l, pad_t, pad_r, pad_b):
        self.pad_l = pad_l
        self.pad_t = pad_t
        self.pad_r = pad_r
        self.pad_b = pad_b

    def __call__(self, image, target_size, mode='nearest'):
        shape = len(image.shape)
        if shape == 3:
            image = image[None, :, :, :]
        elif shape == 2:
            image = image[None, None, :, :]
        # B,C,H,W
        if self.pad_b == 0:
            image = image[:, :, self.pad_t:]
        else:
            image = image[:, :, self.pad_t:-self.pad_b]
        if self.pad_r == 0:
            image = image[:, :, :, self.pad_l:]
        else:
            image = image[:, :, :, self.pad_l:-self.pad_r]

        if isinstance(image, np.ndarray):
            image = cv2.resize(image, (target_size, target_size),
                               interpolation=cv2.INTER_NEAREST if mode == 'nearest' else cv2.INTER_LINEAR)
        elif isinstance(image, torch.Tensor):
            image = torch.nn.functional.interpolate(image, size=(target_size, target_size), mode=mode)

        if shape == 3:
            return image[0]
        elif shape == 2:
            return image[0, 0]
        return image
def get_padding_functions(orig_size,target_size=256,resize_target_size=None,mode='nearest',fill=0):
    """
        padding_func, inverse_padding_func = get_padding_functions(image.size,target_size=256)
        image2 = padding_func(image) # image2.size = (256,256) with padding
        image2.show()
        image3 = inverse_padding_func(image2) # image3.size = (256,256) without padding
        image3.show()
    """
    resize_target_size = target_size if resize_target_size is None else resize_target_size
    imsize = orig_size
    long_size = max(imsize)
    scale = target_size / long_size
    new_h = int(imsize[1] * scale + 0.5)
    new_w = int(imsize[0] * scale + 0.5)

    if (target_size - new_w) % 2 == 0:
        pad_l = pad_r = (target_size - new_w) // 2
    else:
        pad_l,pad_r = (target_size - new_w) // 2,(target_size - new_w) // 2 + 1
    if (target_size - new_h) % 2 == 0:
        pad_t = pad_b = (target_size - new_h) // 2
    else:
        pad_t,pad_b = (target_size - new_h) // 2,(target_size - new_h) // 2 + 1
    inter =  Image.NEAREST if mode == 'nearest' else Image.BILINEAR

    padding_func = transforms.Compose([
        transforms.Resize((new_h,new_w),interpolation=inter),
        transforms.Pad((pad_l, pad_t, pad_r, pad_b), fill=fill, padding_mode='constant')
    ])
    return padding_func, Padding2Resize(pad_l,pad_t,pad_r,pad_b)
class MVTecLOCODataset(Dataset):

    def __init__(self, root, image_size, phase, category, use_pad=False, to_gpu=True, config=None):
        self.phase = phase
        self.category = category
        self.image_size = image_size

        self.use_pad = use_pad
        self.build_transform()

        if phase == 'train':
            print(f"Loading MVTec LOCO {self.category} (train)")
            self.img_path = os.path.join(root, category, 'train')
        elif phase == 'eval':
            print(f"Loading MVTec LOCO {self.category} (validation)")
            self.img_path = os.path.join(root, category, 'validation')
        else:
            print(f"Loading MVTec LOCO {self.category} (test)")
            self.img_path = os.path.join(root, category, 'test')
            self.gt_path = os.path.join(root, category, 'ground_truth')
        assert os.path.isdir(os.path.join(root, category)), 'Error MVTecLOCODataset category:{}'.format(category)

        self.img_paths, self.gt_paths, self.labels, self.types = self.load_paths()  # self.labels => good : 0, anomaly : 1

        # load dataset
        #self.load_images(to_gpu=to_gpu)

    def build_transform(self):
        self.norm_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.resize_norm_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.aug_tranform = transforms.RandomChoice([
            transforms.ColorJitter(brightness=0.2),
            transforms.ColorJitter(contrast=0.2),
            transforms.ColorJitter(saturation=0.2),
        ])
        self.transform_gt = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize(self.image_size, Image.NEAREST),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor()])
    # def load_paths(self):
    #
    #     img_tot_paths = []
    #     gt_tot_paths = []
    #     tot_labels = []
    #     tot_types = []
    #
    #     defect_types = os.listdir(self.img_path)
    #
    #     for defect_type in defect_types:
    #         if defect_type == 'good':
    #             img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
    #             img_tot_paths.extend(img_paths)
    #             gt_tot_paths.extend([0] * len(img_paths))
    #             tot_labels.extend([0] * len(img_paths))
    #             tot_types.extend(['good'] * len(img_paths))
    #         else:
    #             img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
    #             gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*")
    #             gt_paths = [g for g in gt_paths if os.path.isdir(g)]
    #             img_paths.sort()
    #             gt_paths.sort()
    #             img_tot_paths.extend(img_paths)
    #             if len(gt_paths) == 0:
    #                 gt_paths = [0] * len(img_paths)
    #
    #             gt_tot_paths.extend(gt_paths)
    #             tot_labels.extend([1] * len(img_paths))
    #             tot_types.extend([defect_type] * len(img_paths))
    #
    #     assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"
    #
    #     return img_tot_paths, gt_tot_paths, tot_labels, tot_types
    def load_paths(self):
        img_tot_paths = []
        gt_tot_paths = []  # 修改为存储 ground truth 文件夹路径
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([None] * len(img_paths))  # 对于 'good' 类，没有 ground truth，设置为 None
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_folders = glob.glob(os.path.join(self.gt_path, defect_type) + "/*")
                gt_folders = [g for g in gt_folders if os.path.isdir(g)]  # 获取所有有效的 ground truth 文件夹路径
                img_paths.sort()
                gt_folders.sort()
                img_tot_paths.extend(img_paths)
                if len(gt_folders) == 0:
                    gt_tot_paths.extend([None] * len(img_paths))  # 如果没有对应的 ground truth 文件夹，填充 None
                else:
                    gt_tot_paths.extend(gt_folders[:len(img_paths)])  # 假设每个图像都有对应的 ground truth 文件夹

                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def load_images(self, to_gpu=False):

        self.pad_func, self.pad2resize = get_padding_functions(
            Image.open(self.img_paths[0]).size,
            target_size=self.image_size,
            mode='bilinear')

        self.samples = list()
        self.images = list()
        for i in range(len(self.img_paths)):
            img_path, gt, label, img_type = self.img_paths[i], self.gt_paths[i], self.labels[i], self.types[i]
            img = Image.open(img_path).convert('RGB')

            self.images.append(img.copy())

            resize_img = self.resize_norm_transform(img)
            pad_img = self.norm_transform(self.pad_func(img))

            if to_gpu:
                resize_img = resize_img.cuda()
                pad_img = pad_img.cuda()
            self.samples.append({
                'image': resize_img,
                'pad_image': pad_img,
                'label': label,
                'name': os.path.basename(img_path[:-4]),
                'type': img_type,
                'path': img_path,
            })

    # def __getitem__(self, idx):
    #     if self.phase == 'train':
    #         image_path, label = self.img_paths[idx], self.labels[idx]
    #         mask = torch.zeros([1, 256, 256])
    #         image = Image.open(image_path).convert('RGB')
    #         image = self.resize_norm_transform(image)
    #     else:
    #         image_path, label, gt_folders = self.img_paths[idx], self.labels[idx], self.gt_paths[idx]
    #         if label == '0':
    #             mask = torch.zeros([1, 256, 256])
    #         else:
    #             mask_paths = glob.glob(os.path.join(gt_folders, "*.png"))
    #             # 确保 mask_paths 是一个列表
    #             if isinstance(mask_paths, list):
    #                 masks = []
    #                 for mask_path in mask_paths:  # 假设 gt_mask 是一个包含多个 mask 路径的列表
    #                     mask = Image.open(mask_path).convert('1')  # 转为二值图
    #                     masks.append(np.array(mask))
    #                 # 合并多个 masks
    #                 combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
    #                 for mask in masks:
    #                     combined_mask = np.logical_or(combined_mask, mask)
    #                 mask = combined_mask.astype(np.uint8)
    #                 mask = self.transform_mask(mask)
    #             else:
    #                 # 如果只有一个 mask，直接加载
    #                 mask = Image.open(mask_paths).convert('1')
    #                 mask = np.array(mask)
    #                 mask = self.transform_mask(mask)
    #
    #         image = Image.open(image_path).convert('RGB')
    #         image = self.resize_norm_transform(image)
    #
    #     return image, label, mask
    from PIL import Image
    import numpy as np

    def __getitem__(self, idx):
        if self.phase == 'train':
            image_path, label = self.img_paths[idx], self.labels[idx]
            mask = torch.zeros([1, 256, 256])  # 默认没有缺陷的mask
            image = Image.open(image_path).convert('RGB')
            image = self.resize_norm_transform(image)
        else:
            image_path, label, gt_folder = self.img_paths[idx], self.labels[idx], self.gt_paths[idx]

            if label == '0':  # 如果标签是 '0'（正常图像）
                mask = torch.zeros([1, 256, 256])  # 默认没有缺陷的mask
            elif gt_folder is None:  # 如果没有对应的 ground truth 文件夹
                mask = torch.zeros([1, 256, 256])  # 还是返回一个全零的mask
            else:
                # 加载多个 ground truth mask
                masks = []
                # 遍历该图像对应的 ground truth 文件夹，加载其中的所有 mask 文件
                gt_mask_paths = glob.glob(os.path.join(gt_folder, "*.png"))  # 假设每个 ground truth 文件夹下的 mask 是 PNG 文件
                for mask_path in gt_mask_paths:
                    mask_img = Image.open(mask_path).convert('1')  # 转为二值图
                    masks.append(np.array(mask_img))

                # 合并所有 masks，确保得到一个单一的 mask（可以通过逻辑或操作合并）
                combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
                for mask in masks:
                    combined_mask = np.logical_or(combined_mask, mask)

                mask = combined_mask.astype(np.uint8)  # 转换为 uint8 类型
                mask = Image.fromarray(mask)  # 将 numpy.ndarray 转换为 PIL.Image

                mask = self.transform_mask(mask)  # 对 mask 进行相应的变换（如 resize、归一化等）

            image = Image.open(image_path).convert('RGB')
            image = self.resize_norm_transform(image)

        return image, label, mask

