import random
import time
import pickle

from tqdm import tqdm

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms

from model.attention_ocr import OCR
from utils.dataset import CaptchaDataset
from utils.train_util import train_batch, eval_batch

device = 'cuda'


def main():
    img_width = 160
    img_height = 60
    max_len = 4

    nh = 512

    teacher_forcing_ratio = 0.5

    batch_size = 32

    lr = 3e-4
    n_epoch = 100

    n_works = 8
    save_checkpoint_every = 5

    ds_train = CaptchaDataset(img_width, img_height, 10000, max_len)
    ds_test = CaptchaDataset(img_width, img_height, 1000, max_len)

    tokenizer = ds_train.tokenizer

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=n_works)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=n_works)

    model = OCR(img_width, img_height, nh, tokenizer.n_token,
                max_len + 1, tokenizer.SOS_token, tokenizer.EOS_token).to(device=device)

    load_weights = torch.load('./inception_v3_google-1a9a5a14.pth')

    names = set()
    for k, w in model.incept.named_children():
        names.add(k)

    weights = {}
    for k, w in load_weights.items():
        if k.split('.')[0] in names:
            weights[k] = w

    model.incept.load_state_dict(weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    crit = nn.NLLLoss().cuda()

    def train_epoch():
        sum_loss_train = 0
        n_train = 0
        sum_acc = 0
        sum_sentence_acc = 0

        for bi, batch in enumerate(tqdm(train_loader)):
            x, y = batch
            x = x.to(device=device)
            y = y.to(device=device)

            loss, acc, sentence_acc = train_batch(x, y, model, optimizer,
                                                  crit, teacher_forcing_ratio, max_len,
                                                  tokenizer)

            sum_loss_train += loss
            sum_acc += acc
            sum_sentence_acc += sentence_acc

            n_train += 1

        return sum_loss_train / n_train, sum_acc / n_train, sum_sentence_acc / n_train

    def eval_epoch():
        sum_loss_eval = 0
        n_eval = 0
        sum_acc = 0
        sum_sentence_acc = 0

        for bi, batch in enumerate(tqdm(test_loader)):
            x, y = batch
            x = x.to(device=device)
            y = y.to(device=device)

            loss, acc, sentence_acc = eval_batch(x, y, model, crit, max_len, tokenizer)

            sum_loss_eval += loss
            sum_acc += acc
            sum_sentence_acc += sentence_acc

            n_eval += 1

        return sum_loss_eval / n_eval, sum_acc / n_eval, sum_sentence_acc / n_eval

    for epoch in range(n_epoch):
        train_loss, train_acc, train_sentence_acc = train_epoch()
        eval_loss, eval_acc, eval_sentence_acc = eval_epoch()

        print("Epoch %d" % epoch)
        print('train_loss: %.4f, train_acc: %.4f, train_sentence: %.4f' % (train_loss, train_acc, train_sentence_acc))
        print('eval_loss:  %.4f, eval_acc:  %.4f, eval_sentence:  %.4f' % (eval_loss, eval_acc, eval_sentence_acc))

        if epoch % save_checkpoint_every == 0 and epoch > 0:
            print('saving checkpoint...')
            torch.save(model.state_dict(), './chkpoint/time_%s_epoch_%s.pth' % (time.strftime('%Y-%m-%d_%H-%M-%S'), epoch))


if __name__ == '__main__':
    main()
