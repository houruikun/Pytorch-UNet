from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import logging
from PIL import Image


def scaling(pil_img, scale):
    w, h = pil_img.size
    newW, newH = int(scale * w), int(scale * h)
    assert newW > 0 and newH > 0, 'Scale is too small'
    pil_img = pil_img.resize((newW, newH))

    return pil_img


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, use_noise, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.use_noise = use_noise
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids_imgs = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        self.ids_masks = [splitext(file)[0] for file in listdir(masks_dir)
                    if not file.startswith('.')]
        logging.info('Creating dataset with {0} examples'.format(len(self.ids_imgs)))

        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.ids_imgs)

    def __getitem__(self, i):
        idx_img = self.ids_imgs[i]
        idx_mask = self.ids_masks[i]
        mask_file = glob(self.masks_dir + idx_mask + '*')
        img_file = glob(self.imgs_dir + idx_img + '*')

        assert len(mask_file) == 1, \
            'Either no mask or multiple masks found for the ID {0}: {1}'.format(idx_mask, mask_file)
        assert len(img_file) == 1, \
            'Either no image or multiple images found for the ID {0}: {1}'.format(idx_img, img_file)
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            'Image and mask {0} should be the same size, but are {1} and {2}'.format(idx_img, img.size, mask.size)

        img = scaling(img, self.scale)
        mask = scaling(mask, self.scale)
        mask = np.array(mask)
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
        mask = (mask / 255).astype(np.int)

        if not self.use_noise:
            img = np.array(img)
        else:
            img = np.array(self.trancolor(img))
            choice = random.randint(0, 3)
            if choice == 0:
                img = np.fliplr(img)
                mask = np.fliplr(mask)
            elif choice == 1:
                img = np.flipud(img)
                mask = np.flipud(mask)
            elif choice == 2:
                img = np.fliplr(img)
                img = np.flipud(img)
                mask = np.fliplr(mask)
                mask = np.flipud(mask)

        img = np.transpose(img, (2, 0, 1))
        img = self.norm(torch.from_numpy(img.astype(np.float32)))
        mask = torch.from_numpy(mask.astype(np.int64))

        return {'image': img, 'mask': mask}
