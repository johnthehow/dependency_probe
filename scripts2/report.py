from scipy.stats import spearmanr, pearsonr
from config import *
import config
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

def corrindex_reporter(probe_trained, dataloader):
    print(f'[report] calculating Spearman and Pearson corr index...')
    probe = probe_trained[0]
    preds = []
    labs = []
    for feat_batch, lab_batch, length_batch in dataloader:
        probe.eval()
        pred_batch = probe(feat_batch)
        for sent_pred,sent_lab, sent_length in zip(pred_batch,lab_batch,length_batch):
            sent_pred_trim = sent_pred[:int(sent_length.item()),:int(sent_length.item())]
            sent_lab_trim = sent_lab[:int(sent_length.item()),:int(sent_length.item())]
            for row in sent_pred_trim:
                preds.append(row)
            for row in sent_lab_trim:
                labs.append(row)

    preds = torch.cat(preds).detach().numpy()
    labs = torch.cat(labs).detach().numpy()

    dspr = spearmanr(preds, labs)
    dpsr = pearsonr(preds, labs)[0]
    return dspr,dpsr

def uuas_reporter(probe_trained, dataloader):
    print(f'[report] calculating UUAS...')
    probe = probe_trained[0]
    correct_edges = 0
    total_edges = 0
    for feat_batch, lab_batch, length_batch in dataloader:
        probe.eval()
        pred_batch = probe(feat_batch)
        for sent_pred, sent_lab, sent_length in zip(pred_batch,lab_batch,length_batch):
            sent_pred_trim = sent_pred[:int(sent_length.item()),:int(sent_length.item())]
            sent_lab_trim = sent_lab[:int(sent_length.item()),:int(sent_length.item())]
            pred_matrix = sent_pred_trim.detach().numpy()
            gold_matrix = sent_lab_trim.detach().numpy()
            pred_csr = csr_matrix(pred_matrix)
            pred_tcsr = minimum_spanning_tree(pred_csr)
            pred_aja = pred_tcsr.toarray()
            pred_aja_01 = (pred_aja!=0).astype(int)
            gold_csr = csr_matrix(gold_matrix)
            gold_tcsr = minimum_spanning_tree(gold_csr)
            gold_aja = gold_tcsr.toarray()
            gold_aja_01 = (gold_aja !=0).astype(int)
            pred_edges = []
            gold_edges = []
            for ridx, row in enumerate(pred_aja_01):
                for cidx, element in enumerate(row):
                    if element == 1:
                        pred_edges.append((ridx,cidx))
            pred_edges_reverse = [(i[1],i[0]) for i in pred_edges]
            pred_edges += pred_edges_reverse
            for ridx, row in enumerate(gold_aja_01):
                for cidx, element in enumerate(row):
                    if element == 1:
                        gold_edges.append((ridx,cidx))
            correct_edges += len(set(pred_edges).intersection(set(gold_edges)))
            total_edges += len(gold_edges)
    uuas = correct_edges / total_edges
    return uuas

def report_writer(probe_trained, dataloader):
    print(f'[report] writing report...')
    with open(REPORTS_PATH.joinpath(f'REPORT_method_pairwise_depd_corrindex_uuas_{MODEL_NAME}_layer_{HIDDEN_LAYER}_directed_{str(DEPD_DIRECTED)}.txt'), mode='w') as file:
        looper_losses = probe_trained[1]
        probe_filename = probe_trained[2]
        dspr,dpsr = corrindex_reporter(probe_trained, dataloader)
        uuas = uuas_reporter(probe_trained, dataloader)
        file.write(f'[CORRELATION INDICES]')
        file.write('\n')
        file.write(f'Spearman\t{str(dspr.correlation)}')
        file.write('\n')
        file.write(f'Pearson\t{str(dpsr)}')
        file.write('\n')
        file.write('\n')
        file.write(f'[UUAS]')
        file.write('\n')
        file.write(f'UUAS\t{str(uuas)}')
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
        file.write(f'{probe_filename}')
        file.write('\n')
        file.write('\n')
        file.write(f'[TRAINING LOSSES]')
        file.write('\n')
        for k,v in looper_losses.items():
            file.write(f'{k}\t{str(v)}')
            file.write('\n')   