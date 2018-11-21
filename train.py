import random
import time
import pickle

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler

from ignite.engine import Engine, Events, create_supervised_evaluator, _prepare_batch
from ignite.handlers import EarlyStopping, Timer

from torchvision import transforms

from model.attention_ocr import *
from utils.data_loader import OCRDataset, OnePicSampler
from utils.weight_init import weight_init

device = 'cuda'
use_pretrained_incept = False

img_height = 50
img_width = 256
n_channels = 3

preprocess_workers = 30

batch_size = 128
max_char_per_img = 5

MAX_ATTEN_LENGTH = 100

n_epoch = 50
print_interval = 2000

random_state = 42

teacher_forcing_ratio = 0.5

with open('chars.txt', 'r') as f:
    chars = f.readlines()[0]

n_classes = len(chars) + 3  # reserve for SOS, UNKNOWN and EOS
SOS_token = 0
EOS_token = len(chars) + 2

incept = MyIncept().to(device=device)
att = AttnDecoderRNN('dot', 288 + 3 + MAX_ATTEN_LENGTH // 3 + 1, n_classes).to(device=device)

onehot_x = OneHot(3).to(device=device)
onehot_y = OneHot(MAX_ATTEN_LENGTH // 3 + 1).to(device=device)

if use_pretrained_incept:
    # load pretrained weights
    incept.load_state_dict(torch.load('/path/'), restrict=False)
    incept.requires_grad = False
else:
    incept.appy(weight_init)

att.apply(weight_init)


def train_batch(input_tensor, target_tensor, optimizer, criterion, max_length=MAX_ATTEN_LENGTH):
    img_feature = incept(input_tensor)
    b, c, h, w = img_feature.size()
    img_feature = img_feature.permute(0, 2, 3, 1)  # h, w, c

    x, y = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device))

    h_loc = onehot_x(x)
    w_loc = onehot_y(y)

    loc = torch.cat([h_loc, w_loc], dim=2).unsqueeze(0).expand(b, -1, -1, -1)

    img_feature = torch.cat([img_feature, loc], dim=3)

    img_feature = img_feature.contiguous().view(b, h * w, -1).permute(1, 0, 2)

    tmp = torch.zeros((max_length, b, img_feature.size(2)), device=device)
    tmp[:h * w] = img_feature
    img_feature = tmp

    optimizer.zero_grad()

    # target has identical length
    b, w = target_tensor.size()
    tmp = torch.zeros((max_length, b), dtype=torch.long, device=device)
    tmp[:w, :] = target_tensor.permute(1, 0)
    tmp[w, :] = EOS_token

    target_tensor = tmp

    target_length = w + 1

    loss = 0

    decoder_input = torch.full((b,), SOS_token, dtype=torch.long, device=device)

    decoder_hidden = att.init_hidden(b)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = att(
                decoder_input, decoder_hidden, img_feature)

            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = att(
                decoder_input, decoder_hidden, img_feature)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input[0].item() == EOS_token:
                break

    loss.backward()

    optimizer.step()

    return loss.item() / target_length


def inference(input_tensor, max_length=MAX_ATTEN_LENGTH):
    att.eval()
    with torch.no_grad():
        img_feature = incept(input_tensor)
        b, c, h, w = img_feature.size()
        assert b == 1
        img_feature = img_feature.permute(0, 2, 3, 1)  # h, w, c

        x, y = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device))

        h_loc = onehot_x(x)
        w_loc = onehot_y(y)

        loc = torch.cat([h_loc, w_loc], dim=2).unsqueeze(0).expand(b, -1, -1, -1)

        img_feature = torch.cat([img_feature, loc], dim=3)

        img_feature = img_feature.contiguous().view(b, h * w, -1).permute(1, 0, 2)

        tmp = torch.zeros((max_length, b, img_feature.size(2)), device=device)
        tmp[:h * w] = img_feature
        img_feature = tmp

        decoder_input = torch.full((b,), SOS_token, dtype=torch.long, device=device)

        decoder_hidden = att.init_hidden(b)

        decoded_words = []
        #         decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden = att(
                decoder_input, decoder_hidden, img_feature)
            #             decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            decoded_words.append(topi.item())
            if topi.item() == EOS_token:
                break

            decoder_input = topi.squeeze().detach()

        return decoded_words


optimizer = optim.Adadelta(att.parameters())
# optimizer = optim.Adam(att.parameters(), lr=lr)
nll_loss = nn.NLLLoss().to(device=device)


def create_trainer(optimizer, loss_fn, device=device):
    def _update(engine, batch):
        incept.eval()
        att.train()
        x, y = _prepare_batch(batch, device=device)
        loss = train_batch(x, y, optimizer, loss_fn)
        engine.state.loss_total += loss
        return loss

    engine = Engine(_update)

    return engine


img_trans = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomRotation(15, resample=Image.NEAREST),
    transforms.Resize(img_height),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])


def train_on_char_per_img(df, char_per_img):
    print("=================================================================")
    print('Max char length per img: ', char_per_img)

    df_train = df[df.content.str.len() <= char_per_img]

    print('Dataset len: ', len(df))

    ds_train = OCRDataset(df_train, './imgs', chars, char_per_img,
                          img_transform=img_trans)

    train_loader = DataLoader(ds_train, batch_size=batch_size,
                              sampler=OnePicSampler(print_interval * 3, batch_size, df_train, chars),
                              shuffle=False, num_workers=preprocess_workers)

    for sample in train_loader:
        break

    "---------------------------------------------"

    trainer = create_trainer(optimizer, nll_loss, device=device)

    @trainer.on(Events.STARTED)
    def init_loss(trainer):
        trainer.state.loss_total = 0
        trainer.state.loss = 0

    def score_function(engine):
        loss = engine.state.loss
        return -loss

    es_handler = EarlyStopping(patience=500, score_function=score_function, trainer=trainer)

    trainer.add_event_handler(Events.COMPLETED, es_handler)

    timer = Timer(average=True)
    timer.attach(trainer,
                 start=Events.EPOCH_STARTED,
                 resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED,
                 step=Events.ITERATION_COMPLETED)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        if trainer.state.iteration % print_interval == 1:
            trainer.state.loss = trainer.state.loss_total / print_interval
            trainer.state.loss_total = 0
            print("Epoch[{}] Iter[{}] Loss: {:.2f} AvgTime {:.2f}".format(
                trainer.state.epoch, trainer.state.iteration, trainer.state.loss, timer.value()))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        print("Training Results - Epoch: {}  Avg loss: {:.2f}"
              .format(trainer.state.epoch, trainer.state.loss))

    print("==================================================================")
    print('Training...')
    trainer.run(train_loader, max_epochs=n_epoch)

    print('saving model...')
    ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    torch.save(att.state_dict(), './models/' + 'attn_pos_' + str(char_per_img) + '_' + ts + '.pth')


df = pickle.load(open('./df_select_after_seg.p', 'rb'))

for i in sorted(df.content.str.len().unique()):
    train_on_char_per_img(df, i)