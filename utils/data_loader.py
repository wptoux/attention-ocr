import os

from captcha.image import ImageCaptcha

import random
from PIL import Image, ImageOps
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Sampler


img_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=(0.229, 0.224, 0.225)),
])


class CaptchaDataset(Dataset):
    def __init__(self, size, n_chars=4, chars=None):
        self.gen = ImageCaptcha()
        self.size = size

        self.n_chars = n_chars

        if chars is None:
            self.chars = list('1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        else:
            self.chars = list(chars)

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        content = torch.randint(0, len(self.chars), (self.n_chars, ))

        s = ''.join([self.chars[i] for i in content.numpy()])

        d = self.gen.generate(s)
        d = Image.open(d)

        return img_trans(d), content

