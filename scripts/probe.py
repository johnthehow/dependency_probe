from config import *
from thehow.wordpiece2token.mapper import idx_mapper, matrix_denser, matrix_clssep_stripper
from thehow.tuda.depd_core import trees_gi
from collections import namedtuple
from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from load_dataset import prep_dataset_preload
import datetime


# 参考Hewitt2019, probe.py>>>TwoWordPSDProbe
class TwoWordDepdProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Parameter(data = torch.zeros(768, 768))
        nn.init.uniform_(self.proj, -0.05, 0.05)
    def forward(self,batch):
        transformed = torch.matmul(batch, self.proj) # 经过probe投射过的token向量和token的支配词向量 # (batch_size, 最大句长, 2, 32)
        diffs = torch.tensor([1,-1],dtype=torch.float)@transformed # token向量和token的支配词向量之差 # (batch_size, 最大句长, 32)
        squared_diffs = diffs.pow(2) # 差向量每个分量平方 # (batch_size, 最大句长, 32)
        # squared_diffs = diffs
        squared_distances = torch.sum(squared_diffs, -1) # 差向量平方的和 # (batch_size, 最大句长)
        return squared_distances

class loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred_batch, lab_batch,length_batch):
        loss_per_sent = torch.sum(torch.abs(pred_batch - lab_batch), axis=-1)
        normalized_loss_per_sent = loss_per_sent/length_batch
        batch_loss = torch.sum(normalized_loss_per_sent)/len(length_batch)
        return batch_loss

def train(probe,dataloader,loss_fn,optimizer):
    for feat_batch,lab_batch,length_batch in dataloader:
        probe.train()
        optimizer.zero_grad()
        pred_batch = probe(feat_batch) # (batch_size, 最大句长)
        batch_loss = loss_fn(pred_batch,lab_batch,length_batch)
        print(f'[probe] current train batch loss {batch_loss.detach().numpy()}')
        batch_loss.backward()
        optimizer.step()
    return

def dev(probe,dataloader,loss_fn):
    for feat_batch,lab_batch,length_batch in dataloader:
        probe.eval()
        pred_batch = probe(feat_batch)
        batch_loss = loss_fn(pred_batch, lab_batch, length_batch)
        print(f'[probe] current dev batch loss {batch_loss.detach().numpy()}')
    return

DEV_CONLL_PATH = CONLL_PATH.joinpath('en_gum-ud-dev.conllu')
TRAIN_CONLL_PATH = CONLL_PATH.joinpath('en_gum-ud-train.conllu')

# dataloader_dev = prep_dataset(CONLL_PATH.joinpath('en_ewt-ud-test.conllu'))
dataloader_dev = prep_dataset_preload(DEV_CONLL_PATH)
# dataloader_train = prep_dataset(CONLL_PATH.joinpath('en_ewt-ud-train.conllu'))
dataloader_train = prep_dataset_preload(TRAIN_CONLL_PATH)
probe = TwoWordDepdProbe()
loss_fn = loss()
optimizer = Adam(probe.parameters(),lr=LEARNING_RATE)
timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

def looper(epochs):
    for e in range(epochs):
        print(f'[probe] epoch {e}...')
        train(probe,dataloader_train,loss_fn,optimizer)
        dev(probe,dataloader_dev,loss_fn)
    torch.save(probe,PROBE_SAVEPATH.joinpath(f'{timestamp}.pth'))
    print(f'[probe] probe saved at {PROBE_SAVEPATH}')
    return

looper(EPOCHS)

print('done')

