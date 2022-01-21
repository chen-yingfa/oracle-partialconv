import glob
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
# from gen_mask import gen_masks


class Places2(Dataset):
    def __init__(self, train=True, mask_dataset='mask', img_transform=None, mask_transform=None):
        super().__init__()
        self.root_dir = r'./data/places365_standard/'
        if mask_dataset == 'mask':
            self.mask_dir = r'./data/mask/'
        elif mask_dataset == 'mask_light':
            self.mask_dir = r'./data/mask_light/'
        elif mask_dataset == 'mask_lightest':
            self.mask_dir = r'./data/mask_lightest/'
        self.imgs_dir = self.root_dir + 'train.txt' if train else self.root_dir + 'val.txt'
        self.img_height, self.img_width = (256, 256)
        # 这两个文件里面有路径，其实就跟voc一样

        with open(self.imgs_dir, 'r', encoding='utf-8') as f:
            self.imgs_path = f.readlines()
            self.imgs_path = [self.root_dir + i.strip()
                              for i in self.imgs_path]
            #self.imgs_path = self.imgs_path[:100]
        self.masks_path = glob.glob(self.mask_dir + '*.png')

        self.imgs_cnt = len(self.imgs_path)
        self.masks_cnt = len(self.masks_path)

        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return self.imgs_cnt

    def __getitem__(self, index):
        img = Image.open(self.imgs_path[index % self.imgs_cnt]).convert('RGB')
        # 模只是保险，其实不会超的
        mask = Image.open(self.masks_path[np.random.randint(
            0, self.masks_cnt)]).convert('RGB')
        #mask = Image.open('./07772.png').convert('RGB')
        # mask = np.asarray(mask).copy()
        # 随便选一个，就只能在这些里面选
        # mask = np.expand_dims(mask,axis=-1)
        # anti_mask_img = img * (1 - mask)
        # anti_mask_img = Image.fromarray(anti_mask_img)
        # zero_mask = np.stack((np.reshape((mask == 0),(self.img_height,self.img_width)),) * 3,axis=-1)
        # mask_img[zero_mask] += 255
        # mask[mask == 0] += 255 # 本来是1和255，ToTensor会归一化，所以变成了0到1之间
        # mask = Image.fromarray(mask).convert('RGB')

        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        else:
            mask = np.expand_dims(mask, axis=-1)
        mask_img = img * mask
        return mask_img, mask, img  # img要返回的，因为要给loss


class OracleDataset(Dataset):
    def __init__(self, img_shape: tuple, data_dir: str, mask_dir: str='data/mask', train: bool=True, img_transform=None, mask_transform=None):
        super().__init__()

        self.root_dir = data_dir
        self.mask_dir = mask_dir
        self.is_train = train
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        self.img_height, self.img_width = img_shape
        self.masks_path = sorted(glob.glob(f'{mask_dir}/*.png'))
        self.img_files = sorted(glob.glob(f'{data_dir}/[0-9]*/*.png'))
        self.imgs_cnt = len(self.img_files)
        self.masks_cnt = len(self.masks_path)

    def __len__(self):
        return self.imgs_cnt

    def __getitem__(self, index):
        img = Image.open(self.img_files[index % self.imgs_cnt]).convert('RGB')
        # Pick a random mask
        rand_mask_file = self.masks_path[np.random.randint(0, self.masks_cnt)]
        mask = Image.open(rand_mask_file).convert('RGB')
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        else:
            mask = np.expand_dims(mask, axis=-1)
        masked_img = img * mask
        return masked_img, mask, img


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--img-h', type=int, default=128)
    p.add_argument('--img-w', type=int, default=128)
    args = p.parse_args()

    img_shape = (args.img_h, args.img_w)
    img_mean = np.array([0.485, 0.456, 0.406])
    img_std = np.array([0.229, 0.224, 0.225])

    img_transform = transforms.Compose([
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),

    ])
    mask_transform = transforms.Compose([
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor()
    ])
    # dataset = Places2(img_transform=img_transform,
    #                   mask_transform=mask_transform)
    dataset = OracleDataset(
        img_shape=img_shape,
        data_dir='../data/oracle', 
        mask_dir='data/mask',
        train=True,
        img_transform=img_transform,
        mask_transform=mask_transform)
    for eg in dataset:
        masked_img, mask, img = eg
        masked_img = transforms.ToPILImage()(masked_img)
        masked_img.show()
        #print(mask)
        mask = transforms.ToPILImage()(mask)
        mask.show()
        exit()


if __name__ == '__main__':
    main()
