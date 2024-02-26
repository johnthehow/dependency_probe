from collections import defaultdict
from scipy.stats import spearmanr
import numpy as np
from config import *
import config

'''
    prediction_batches: # list @ batch数量 [0,...,N_BATCH] # ndarray @(batch_size, 最大句长， 最大句长)
    dataset: # dataloader @batch数量 
      [0]: # tuple @ 4 
        [0]: * feat_batch # tensor @(batch_size, 最大句长，1024) 
        [1]: * lab_batch # tensor @(batch_size, 最大句长，最大句长） 
        [2]: * len_batch # tensor @ (batch_size,) 
        [3]: * obs_batch 一批句子 conll + feat_embeddings # list @ batch_size 
          [0,1,2,.., 39]: # tuple @ 2
            [0]: # observation<-namedtuple 
              .embeddings: * 内容同dataset[0]中一句话的内容，但是行数等于句长而不是批最大句长 @ （句长，1024)
              .extra_info
              .governance_relations
              .head_idx
              .index
              .lemma_sentence
              .morph
              .secondary_relations
              .sentence
              .upos
              .xpos
            [1]: * 句子中两次之间的路径距离， 内容类似dataset[1],但是行数等于句长而不是批最大句长 # tensor @ (句长，句长)
    '''

def report_spearmanr(probe_trained, dataset):
    prediction_batches = []
    probe = probe_trained[0]
    for feat_batch, lab_batch, length_batch, observation_batch in dataset:
        pred = probe(feat_batch)
        prediction_batches.append(pred)
    lengths_to_spearmanrs = defaultdict(list)
    for prediction_batch, (data_batch, label_batch, length_batch, observation_batch) in zip(prediction_batches, dataset):
        for prediction, label, length, (observation, _) in zip(prediction_batch, label_batch, length_batch, observation_batch):
            words = observation.sentence
            length = int(length)
            prediction = prediction[:length,:length].detach()
            label = label[:length,:length].cpu()
            spearmanrs = [spearmanr(pred, gold) for pred, gold in zip(prediction, label)]
            lengths_to_spearmanrs[length].extend([x.correlation for x in spearmanrs])
    mean_spearman_for_each_length = {length: np.mean(lengths_to_spearmanrs[length]) for length in lengths_to_spearmanrs}
    mean_spearman_for_sents_len_5_50 = np.mean([mean_spearman_for_each_length[x] for x in range(5,51) if x in mean_spearman_for_each_length])

    return mean_spearman_for_each_length, mean_spearman_for_sents_len_5_50

def report_writer(probe_trained, dataset):
    print(f'[REPORT] writing reports...')
    spearmanr_res = report_spearmanr(probe_trained, dataset)
    mean_spearman_for_each_length = spearmanr_res[0]
    mean_spearman_for_sents_len_5_50 = spearmanr_res[1]
    uuas_res = report_uuas(probe_trained, dataset)
    report_filepath = REPORTS_PATH.joinpath(f'REPORT[COR]SPEARMANR[ACC]UUAS[MODEL]{MODEL_NAME}[LAYER]{HIDDEN_LAYER}[EPOCHS]{EPOCHS}[BATCHSIZE]{BATCH_SIZE}[LR]{LEARNING_RATE}[LOSSPOW]{PROBE_LOSS_POW}[METHOD]{METHOD}{CONLL_ALIAS}[DIRECTED]{str(DEPD_DIRECTED).lower()}.txt')
    with open(report_filepath, 'w') as fout:
         fout.write(f'[SPEARMANR]\n')
         fout.write(f'sent_len\tmean_spearmanr\n')
         for length in sorted(mean_spearman_for_each_length):
             fout.write(str(length) + '\t' + str(mean_spearman_for_each_length[length]) + '\n')
         fout.write(f'mean spearmanr for sents of len 5-50\n')
         fout.write(f'{str(mean_spearman_for_sents_len_5_50)}\n')
         fout.write(f'\n[UUAS]\n')
         fout.write(f'{uuas_res}\n')
         fout.write(f'\n[SETTINGS]\n')
         for k,v in (config.__dict__).items():
             if k.isupper() == True:
                 fout.write(f'{k}\t{str(v)}\n')
    print(f'[REPORT] report saved at {report_filepath}')
    return
 



class UnionFind:
  '''
  Naive UnionFind implementation for (slow) Prim's MST algorithm

  Used to compute minimum spanning trees for distance matrices
  '''
  def __init__(self, n):
    self.parents = list(range(n))
  def union(self, i,j):
    if self.find(i) != self.find(j):
      i_parent = self.find(i)
      self.parents[i_parent] = j
  def find(self, i):
    i_parent = i
    while True:
      if i_parent != self.parents[i_parent]:
        i_parent = self.parents[i_parent]
      else:
        break
    return i_parent

def prims_matrix_to_edges(matrix, words, poses):
  '''
  Constructs a minimum spanning tree from the pairwise weights in matrix;
  returns the edges.

  Never lets punctuation-tagged words be part of the tree.
  '''
  pairs_to_distances = {}
  uf = UnionFind(len(matrix))
  for i_index, line in enumerate(matrix):
    for j_index, dist in enumerate(line):
      if poses[i_index] in ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]:
        continue
      if poses[j_index] in ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]:
        continue
      pairs_to_distances[(i_index, j_index)] = dist
  edges = []
  for (i_index, j_index), distance in sorted(pairs_to_distances.items(), key = lambda x: x[1]):
    if uf.find(i_index) != uf.find(j_index):
      uf.union(i_index, j_index)
      edges.append((i_index, j_index))
  return edges


def report_uuas(probe_trained, dataset):
    '''
    prediction_batches: # list @ batch数量 [0] # ndarray @(batch_size, 最大句长， 最大句长)
    dataset: # dataloader @batch数量 
        [0]: # tuple @ 4 
        [0]: * feat_batch # tensor @(batch_size, 最大句长，1024) 
        [1]: * lab_batch # tensor @(batch_size, 最大句长，最大句长） 
        [2]: * len_batch # tensor @ (batch_size,) 
        [3]: * obs_batch 一批句子 conll + feat_embeddings # list @ batch_size 
            [0,1,2,.., 39]: # tuple @ 2
            [0]: # observation<-namedtuple 
                .embeddings: * 内容同dataset[0]中一句话的内容，但是行数等于句长而不是批最大句长 @ （句长，1024)
                .extra_info
                .governance_relations
                .head_idx
                .index
                .lemma_sentence
                .morph
                .secondary_relations
                .sentence
                .upos
                .xpos
            [1]: * 句子中两次之间的路径距离， 内容类似dataset[1],但是行数等于句长而不是批最大句长 # tensor @ (句长，句长)
    '''
    prediction_batches = []
    probe = probe_trained[0]
    for feat_batch, lab_batch, length_batch, observation_batch in dataset:
        pred = probe(feat_batch)
        prediction_batches.append(pred)
    uspan_total = 0
    uspan_correct = 0
    total_sents = 0
    for prediction_batch, (data_batch, label_batch, length_batch, observation_batch) in zip(prediction_batches, dataset):
        for prediction, label, length, (observation, _) in zip(prediction_batch, label_batch,length_batch, observation_batch):
            words = observation.sentence
            poses = observation.xpos_sentence
            length = int(length)
            assert length == len(observation.sentence)
            prediction = prediction[:length,:length]
            label = label[:length,:length].cpu()
            gold_edges = prims_matrix_to_edges(label, words, poses)
            pred_edges = prims_matrix_to_edges(prediction, words, poses)
            uspan_correct += len(set([tuple(sorted(x)) for x in gold_edges]).intersection(set([tuple(sorted(x)) for x in pred_edges])))
            uspan_total += len(gold_edges)
            total_sents += 1
    uuas = uspan_correct / float(uspan_total)
    return uuas