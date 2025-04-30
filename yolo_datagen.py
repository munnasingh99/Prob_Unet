
import numpy as np
import flammkuchen as fl
import cv2
import albumentations as A
import random
import torch
from torch.utils.data import Dataset

class DataGeneratorDataset(Dataset):
    def __init__(self, fn, samples_per_epoch=50000, size=(1, 128, 128), 
                 target_resolution=None, augment=True, shuffle=True, 
                 seed=42, normalize=[0, 1], min_content=50,
                 hard_negative_rate=0.1):
        self.fn = fn
        self.augment = augment
        self.shuffle = shuffle
        self.seed = seed
        self.normalize = normalize
        self.samples_per_epoch = samples_per_epoch
        self.size = size
        self.target_resolution = target_resolution
        self.min_content = min_content
        self.hard_negative_rate = hard_negative_rate

        # Load data
        self.d = fl.load(self.fn)
        self.data = self.d['data']
        self.meta = self.d['meta']

        # Seed randomness
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Define augmentation
        self.aug = self._get_augmenter()

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        eps = 1e-5

        # Random hard negative decision
        force_negative = self.hard_negative_rate > 0 and random.random() < self.hard_negative_rate

        while True:
            image, dendrite, spines = self.getSample()

            # Hard negative mode: force near-zero spines
            if force_negative and spines.sum() > 5:
                continue

            # Normal sample mode
            if not force_negative and self.min_content > 0:
                if (dendrite.sum() + spines.sum()) < self.min_content:
                    continue

            break

        if self.augment:
            augmented = self.aug(
                image=image, 
                mask1=dendrite.astype(np.uint8), 
                mask2=spines.astype(np.uint8)
            )
            image = augmented['image']
            dendrite = augmented['mask1']
            spines = augmented['mask2']

        # Normalize image
        image = (image.astype(np.float32) - image.min()) / (image.max() - image.min() + eps)
        image = image * (self.normalize[1] - self.normalize[0]) + self.normalize[0]
        image = torch.from_numpy(image).float()
        if image.dim() == 2:
            image = image.unsqueeze(0)

        dendrite = torch.from_numpy(dendrite.astype(np.float32)).float()
        if dendrite.dim() == 2:
            dendrite = dendrite.unsqueeze(0)

        spines = torch.from_numpy(spines.astype(np.float32)).float()
        if spines.dim() == 2:
            spines = spines.unsqueeze(0)

        return image, (dendrite, spines)

    def _get_augmenter(self):
        return A.Compose([
            A.RandomBrightnessContrast(p=0.25),
            A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT, p=0.5),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ElasticTransform(p=0.3, alpha=1, sigma=50, alpha_affine=30),
            A.Blur(p=0.2),
            A.GaussNoise(p=0.3),
            A.RandomGamma(p=0.3)
        ], p=1,
        additional_targets={
            'mask1': 'mask',
            'mask2': 'mask'
        })

    def getSample(self, squeeze=True):
        while True:
            r = self._getSample(squeeze)
            if r is not None:
                return r

    def _getSample(self, squeeze=True):
        if len(self.size) == 2:
            size = (1,) + self.size
        else:
            size = self.size

        r_stack = np.random.choice(len(self.meta))
        target_h, target_w = size[1], size[2]

        if self.target_resolution is None:
            scaling = 1
            h, w = target_h, target_w
        else:
            scaling = self.target_resolution / self.meta.iloc[r_stack].Resolution_XY
            h = round(scaling * target_h)
            w = round(scaling * target_w)

        if self.meta.iloc[r_stack].Width - w <= 0 or self.meta.iloc[r_stack].Height - h <= 0:
            return

        x = np.random.choice(self.meta.iloc[r_stack].Width - w)
        y = np.random.choice(self.meta.iloc[r_stack].Height - h)
        r_plane = np.random.choice(self.meta.iloc[r_stack].Depth - size[0] + 1)
        z_begin, z_end = r_plane, r_plane + size[0]

        tmp_stack = self.data['stacks'][f'x{r_stack}'][z_begin:z_end, y:y+h, x:x+w]
        tmp_dendrites = self.data['dendrites'][f'x{r_stack}'][z_begin:z_end, y:y+h, x:x+w]
        tmp_spines = self.data['spines'][f'x{r_stack}'][z_begin:z_end, y:y+h, x:x+w]

        if scaling != 1:
            return_stack, return_dendrites, return_spines = [], [], []
            for i in range(tmp_stack.shape[0]):
                return_stack.append(cv2.resize(tmp_stack[i], (target_h, target_w)))
                return_dendrites.append(cv2.resize(tmp_dendrites[i].astype(np.uint8), (target_h, target_w)).astype(bool))
                return_spines.append(cv2.resize(tmp_spines[i].astype(np.uint8), (target_h, target_w)).astype(bool))
            return_stack = np.asarray(return_stack)
            return_dendrites = np.asarray(return_dendrites)
            return_spines = np.asarray(return_spines)
        else:
            return_stack = tmp_stack
            return_dendrites = tmp_dendrites
            return_spines = tmp_spines

        if squeeze:
            return return_stack.squeeze(), return_dendrites.squeeze(), return_spines.squeeze()
        else:
            return return_stack, return_dendrites, return_spines