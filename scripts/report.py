from scipy.stats import spearmanr, pearsonr
from config import *
import config
import torch

def reporter(probe_trained, dataloader):
    probe = probe_trained[0]
    looper_losses = probe_trained[1]
    probe_filename = probe_trained[2]
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
    dpsr = pearsonr(preds, labs)[0]
    with open(REPORTS_PATH.joinpath(f'REPORT_{MODEL_NAME}_layer_{HIDDEN_LAYER}.txt'), mode='w') as file:
        file.write(f'[CORRELATION INDICES]')
        file.write('\n')
        file.write(f'Spearman\t{str(dspr.correlation)}')
        file.write('\n')
        file.write(f'Pearson\t{str(dpsr)}')
        file.write('\n')
        file.write('\n')
        file.write(f'[PROBE SETTINGS]')
        file.write('\n')
        for k,v in (config.__dict__).items():
            if k.isupper() == True and k != 'HIDDEN_LAYER':
                file.write(f'{k}\t{str(v)}')
                file.write('\n')
        file.write('\n')
        file.write(f'[PROBE ARCHIVE]')
        file.write(f'\n')
        file.write(f'\n')
        file.write(f'{probe_filename}')
        file.write('\n')
        file.write(f'[TRAINING LOSSES]')
        file.write('\n')
        for k,v in looper_losses.items():
            file.write(f'{k}\t{str(v)}')
            file.write('\n')

    return dpsr,dpsr