from config import *
from torch import nn
import torch
from torch.optim import Adam
import datetime
from dataset import dataloaders_triad
from collections import defaultdict


# 参考Hewitt2019, probe.py>>>TwoWordPSDProbe
class TwoWordDepdProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Parameter(data = torch.zeros(NDIM_TOKEN_EMBEDDING, PROBE_RANK))
        nn.init.uniform_(self.proj, -0.05, 0.05)
    def forward(self,batch): # shape: batch_size, 最大句长, 768
        transformed = torch.matmul(batch, self.proj) # shape: batch_size, 最大句长，64
        transformed = transformed.unsqueeze(2) # shape: batch_size, 最大句长， 1， 64
        transformed = transformed.expand(-1, -1, transformed.shape[1], -1)
        transposed = transformed.transpose(1,2) # 等价于 transformed.unsqueeze(1).expand(-1,句长,-1,-1)
        diffs = transformed - transposed #
        squared_diffs = diffs.pow(PROBE_LOSS_POW)
        squared_distances = torch.sum(squared_diffs, -1)
        return squared_distances

class loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_pair_dims = (1,2)
    def forward(self, pred_batch, lab_batch,length_batch):
        labels_1s = (lab_batch != -1).float() # lab_batch@(batch_size, 最大句长, 最大句长)
        pred_batch_masked = pred_batch * labels_1s # (batch_size, 最大句长, 最大句长)
        labels_masked = lab_batch * labels_1s # (batch_size, 最大句长, 最大句长)
        total_sents = torch.sum((length_batch != 0)).float() # 一批的句子数, 默认是40 torch.tensor, shape:[]
        squared_lengths = length_batch.pow(2).float() # 各个句子句长的平方 # 长度为40的tensor
        if total_sents > 0:
            loss_per_sent = torch.sum(torch.abs(pred_batch_masked - labels_masked), dim=self.word_pair_dims) # 长度为40的tensor
            normalized_loss_per_sent = loss_per_sent / squared_lengths # 一批中, 每句话的标准化损失
            batch_loss = torch.sum(normalized_loss_per_sent) / total_sents # 一批中的总损失/总句子数
        else:
            batch_loss = torch.tensor(0.0)
        return batch_loss

def train(probe,dataloader,loss_fn,optimizer):
    total_batch_loss = 0
    batch_cnt = 0
    for feat_batch,lab_batch,length_batch,obs_batch in dataloader:
        probe.train()
        optimizer.zero_grad()
        pred_batch = probe(feat_batch) # (batch_size, 最大句长)
        batch_loss = loss_fn(pred_batch,lab_batch,length_batch)
        total_batch_loss += batch_loss.item()
        # print(f'[probe] current train batch loss {batch_loss.detach().numpy()}')
        batch_loss.backward()
        optimizer.step()
        batch_cnt += 1
    return total_batch_loss/batch_cnt

def dev(probe,dataloader,loss_fn):
    total_batch_loss = 0
    batch_cnt = 0
    for feat_batch,lab_batch,length_batch,obs_batch in dataloader:
        probe.eval()
        pred_batch = probe(feat_batch)
        batch_loss = loss_fn(pred_batch, lab_batch, length_batch)
        # print(f'[probe] current dev batch loss {batch_loss.detach().numpy()}')
        total_batch_loss += batch_loss.item()
        batch_cnt += 1
    return total_batch_loss/batch_cnt

probe = TwoWordDepdProbe()
loss_fn = loss()
optimizer = Adam(probe.parameters(),lr=LEARNING_RATE)
timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

dataloader_train, dataloader_dev, dataloader_test = dataloaders_triad

def looper(epochs):
    epoch_losses = defaultdict(list)
    for e in range(epochs):
        print(f'[PROBE] epoch {e}...')
        train_epoch_loss = train(probe,dataloader_train,loss_fn,optimizer)
        dev_epoch_loss = dev(probe,dataloader_dev,loss_fn)
        epoch_losses[e] = [train_epoch_loss,dev_epoch_loss]
        print(f'[PROBE] train_loss: {train_epoch_loss}, dev_loss: {dev_epoch_loss}')
    probe_filename = f'PROBE[MODEL]{MODEL_NAME}[LAYER]{HIDDEN_LAYER}[DATA]{TRAIN_CONLL_PATH.stem}[RANK]{PROBE_RANK}[EPOCHS]{EPOCHS}[BATCHSIZE]{BATCH_SIZE}[LR]{LEARNING_RATE}[LOSSPOW]{PROBE_LOSS_POW}.pth'
    torch.save(probe,PROBE_SAVEPATH.joinpath(probe_filename))
    print(f'[PROBE] probe saved at {PROBE_SAVEPATH} {probe_filename}')
    return probe,epoch_losses,probe_filename

