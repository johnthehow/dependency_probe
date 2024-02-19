from config import *
from report import reporter
from torch import nn
import torch
from torch.optim import Adam
import datetime
from load_dataset import dataloader_train, dataloader_dev
from collections import defaultdict


# 参考Hewitt2019, probe.py>>>TwoWordPSDProbe
class TwoWordDepdProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Parameter(data = torch.zeros(NDIM_TOKEN_EMBEDDING, PROBE_RANK))
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
    total_batch_loss = 0
    batch_cnt = 0
    for feat_batch,lab_batch,length_batch in dataloader:
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
    for feat_batch,lab_batch,length_batch in dataloader:
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

def looper(epochs):
    epoch_losses = defaultdict(list)
    epoch_dev_losses = []
    for e in range(epochs):
        print(f'[probe] epoch {e}...')
        train_epoch_loss = train(probe,dataloader_train,loss_fn,optimizer)
        dev_epoch_loss = dev(probe,dataloader_dev,loss_fn)
        epoch_losses[e] = [train_epoch_loss,dev_epoch_loss]
        print(f'[probe] train_loss: {train_epoch_loss}, dev_loss: {dev_epoch_loss}')
    probe_filename = f'probe_{MODEL_NAME}_ndim_{NDIM_TOKEN_EMBEDDING}_rank_{PROBE_RANK}_directed_{str(DEPD_DIRECTED)}_{timestamp}.pth'
    torch.save(probe,PROBE_SAVEPATH.joinpath(probe_filename))
    print(f'[probe] probe saved at {PROBE_SAVEPATH} {probe_filename}')
    return probe,epoch_losses,probe_filename

