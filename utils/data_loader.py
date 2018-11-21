import os
import random

from PIL import Image, ImageOps

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler


class OCRDataset(Dataset):
    def __init__(self, df, content_img_dir, chars, char_per_img,
                 img_transform=None):
        self._char_per_img = char_per_img

        df = df.copy()
        #         df = pd.read_csv(content_csv_file, encoding='utf-8')
        df['str_len'] = df.content.str.len()
        df = df[df.str_len == char_per_img].drop('str_len', axis=1)

        #         assert len(df) == len(os.listdir(content_img_dir)), 'Corrupted Data!'

        self._img_dir = content_img_dir

        keys = {}

        for i, c in enumerate(chars):
            keys[c] = i + 1  # 0 is for 'space'

        self._label_tensors = []
        self._fns = []
        for row in df.itertuples():
            t = torch.zeros(len(row.content, ), dtype=torch.int32)

            flag = True
            for i, c in enumerate(row.content):
                if c in chars:
                    t[i] = keys[c] + 1  # 0 reserved for SOS
                else:
                    t[i] = len(chars) + 1

            self._label_tensors.append(t)
            self._fns.append(row.filename)

        self._img_transform = img_transform

    def __len__(self):
        return len(self._fns)

    def __getitem__(self, idx):
        fn = self._fns[idx]
        img = Image.open(os.path.join(self._img_dir, fn + '.jpg'))

        if img.size[1] > img.size[0] * 2:
            img = img.rotate(90, expand=True)
        elif img.size[1] > img.size[0]:
            img = img.resize((img.size[0], img.size[0]))

        img = ImageOps.autocontrast(img)

        img = img.resize((img.height * self._char_per_img, img.height))
        #         a = np.array(img)
        #         a = cv2.medianBlur(a, 5)
        #         img = Image.fromarray(a)

        if self._img_transform is not None:
            img = self._img_transform(img)

        return img, self._label_tensors[idx]


class OnePicSampler(Sampler):
    def __init__(self, data_len, batch_size, df, chars):
        charset = set(chars)

        char_imgs = {}

        for i, row in enumerate(df.itertuples()):
            for c in row.content:
                if c in charset:
                    if c not in char_imgs:
                        char_imgs[c] = []

                    char_imgs[c].append(i)

        self._chars = chars
        self._char_imgs = char_imgs
        self._avail_chars = list(char_imgs.keys())
        self._n_data = data_len
        self._batch_size = batch_size

    def __len__(self):
        return self._n_data

    def __iter__(self):
        index = torch.zeros((self._n_data * self._batch_size,), dtype=torch.long)
        for i in range(self._n_data * self._batch_size):
            c = random.choice(self._avail_chars)
            idx = random.choice(self._char_imgs[c])

            index[i] = idx

        return iter(index)