from config import *
import config
from thehow.wordpiece2token.mapper import idx_mapper, matrix_denser, matrix_clssep_stripper
from thehow.tuda.depd_core import trees_gi
from collections import namedtuple
from torch import nn
import torch
from torch.optim import Adam
from load_dataset import prep_dataset_preload, prep_dataset_onsite
import datetime
from scipy.stats import spearmanr, pearsonr


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

def test(probe, dataloader, loss_fn):
    total_batch_loss = 0
    batch_cnt = 0
    for feat_batch,lab_batch,length_batch in dataloader:
        probe.eval()
        pred_batch = probe(feat_batch)
        batch_loss = loss_fn(pred_batch, lab_batch, length_batch)
        total_batch_loss += batch_loss.item()
        # print(f'[probe] current test batch loss {batch_loss.detach().numpy()}')
        batch_cnt += 1
    return total_batch_loss/batch_cnt

def report(probe, dataloader):
    preds = []
    labs = []
    for feat_batch, lab_batch, length_batch in dataloader:
        probe.eval()
        sent_cnt = 0
        for i in feat_batch:
            feat_sent = feat_batch[sent_cnt][:int(length_batch[sent_cnt].item())]
            lab_sent = lab_batch[sent_cnt][:int(length_batch[sent_cnt].item())]
            pred_sent = probe(feat_sent)
            preds.append(pred_sent)
            labs.append(lab_sent)
            sent_cnt += 1
    preds = torch.cat(preds).detach().numpy()
    labs = torch.cat(labs).detach().numpy()

    dspr = spearmanr(preds, labs)
    with open(REPORTS_PATH.joinpath(f'DSPR_{MODEL_NAME}_{HIDDEN_LAYER}.txt'), mode='w') as file:
        file.write(f'DSPR\t{str(dspr.correlation)}')
        file.write('\n')
        for k,v in (config.__dict__).items():
            if k.isupper() == True and k != 'HIDDEN_LAYER':
                file.write(f'{k}\t{str(v)}')
                file.write('\n')
    return dspr
DEV_CONLL_PATH = CONLL_PATH.joinpath('en_gum-ud-dev.conllu')
TRAIN_CONLL_PATH = CONLL_PATH.joinpath('en_gum-ud-train.conllu')
TEST_CONLL_PATH = CONLL_PATH.joinpath('en_gum-ud-test.conllu')

if ONSITE_EMBEDDINGS == True:
    dataloader_dev = prep_dataset_onsite(DEV_CONLL_PATH)
    dataloader_train = prep_dataset_onsite(TRAIN_CONLL_PATH)
    dataloader_test = prep_dataset_onsite(TEST_CONLL_PATH)
else:
    dataloader_dev = prep_dataset_preload(DEV_CONLL_PATH)
    dataloader_train = prep_dataset_preload(TRAIN_CONLL_PATH)
    dataloader_test = prep_dataset_preload(TEST_CONLL_PATH) 
probe = TwoWordDepdProbe()
loss_fn = loss()
optimizer = Adam(probe.parameters(),lr=LEARNING_RATE)
timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

def looper(epochs):
    for e in range(epochs):
        print(f'[probe] epoch {e}...')
        train_epoch_loss = train(probe,dataloader_train,loss_fn,optimizer)
        dev_epoch_loss = dev(probe,dataloader_dev,loss_fn)
        test_epoch_loss = test(probe,dataloader_test,loss_fn)
        print(f'[probe] train_loss: {train_epoch_loss}, dev_loss: {dev_epoch_loss}, test_loss: {test_epoch_loss}')
    torch.save(probe,PROBE_SAVEPATH.joinpath(f'probe_{MODEL_NAME}_ndim_{NDIM_TOKEN_EMBEDDING}_rank_{PROBE_RANK}_{timestamp}.pth'))
    print(f'[probe] probe saved at {PROBE_SAVEPATH}')
    return probe

probe_trained = looper(EPOCHS)

report(probe_trained, dataloader_test)

print('done')

