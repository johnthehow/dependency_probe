from transformers import BertModel, BertTokenizer
from config import *
from thehow.wordpiece2token.mapper import idx_mapper, matrix_denser, matrix_clssep_stripper
from thehow.tuda.depd_core import trees_gi
from collections import namedtuple
import torch
from torch.utils.data import Dataset, DataLoader
import pickle

model = BertModel.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

def prep_dataset_onsite(conll_path):
	tgi = trees_gi(conll_path)
	trees = [tree for tree in tgi]
	mappable_trees = []
	densemats = []
	texts = []
	lengths = []
	user_tksents = []
	wdps_tksents = []
	idxmaps = []
	depdmats = []
	tree_cnt = 0
	print(f'[load_dataset] calculating word embeddings and assemblying into observation objects')
	for tree in trees:
		text = tree.text_lower
		user_tksent = [node.token.lower() for node in tree.nodes]
		bert_tksent = tokenizer(text, return_tensors='pt')
		wdps_tksent = tokenizer.convert_ids_to_tokens(bert_tksent['input_ids'][0][1:-1])
		idxmap = idx_mapper(wdps_tksent, user_tksent)
		if idxmap != None: # 筛选， 只允许可合并wordpiece的句子
			with torch.no_grad():
				bert_output = model(**bert_tksent, output_hidden_states = True)
			res_hidden = bert_output['hidden_states'][HIDDEN_LAYER][0] # 句长+2*768
			stripmat = matrix_clssep_stripper(res_hidden, scale=False)
			densemat = matrix_denser(stripmat, idxmap, rowwise=True)
			densemats.append(densemat)
			mappable_trees.append(tree)
			texts.append(text)
			lengths.append(tree.len)
			user_tksents.append(user_tksent)
			wdps_tksents.append(wdps_tksent)
			idxmaps.append(idxmap)
			depdmats.append(torch.tensor(tree.ajacency_matrix_weighted_absolute_full))
		else:
			# print(f'[load_dataset] {tree_cnt}-th wordpiece-tokenized sent not alignable')
			# print(f'[load_dataset] {text}')
			pass
		tree_cnt += 1
	max_sentlen = max(lengths)
	feats_tensor = torch.zeros(len(mappable_trees), max_sentlen, NDIM_TOKEN_EMBEDDING)
	for sent_idx,densemat in enumerate(densemats):
		n_densemat_rows = densemat.shape[0]
		n_densemat_cols = densemat.shape[1]
		feats_tensor[sent_idx,:n_densemat_rows,:n_densemat_cols] = densemat
	print('[LOAD_DATASET] pickling dataset feats tensor...')
	with open(DATASET_PKL_PATH.joinpath(f'feats_tensor_method_pairwise_depd_model_{MODEL_NAME}_ndim_{NDIM_TOKEN_EMBEDDING}_conll_{conll_path.stem}_layer_{HIDDEN_LAYER}.pkl'), mode='wb') as file:
		pickle.dump(feats_tensor, file)
	labs_tensor = torch.zeros(len(mappable_trees), max_sentlen, max_sentlen)
	labs_tensor = labs_tensor-1
	for sent_idx,depdmat in enumerate(depdmats):
		n_depdmat_rows = depdmat.shape[0]
		n_depdmat_cols = depdmat.shape[1]
		labs_tensor[sent_idx,:n_depdmat_rows,:n_depdmat_cols] = depdmat
	print('[LOAD_DATASET] pickling dataset labs tensor...')
	with open(DATASET_PKL_PATH.joinpath(f'labs_tensor_method_pairwise_depd_model_{MODEL_NAME}_ndim_{NDIM_TOKEN_EMBEDDING}_conll_{conll_path.stem}_layer_{HIDDEN_LAYER}.pkl'), mode='wb') as file:
		pickle.dump(labs_tensor, file)
	lengths_tensor = torch.tensor(lengths,dtype=torch.float)
	print('[LOAD_DATASET] pickling dataset lengths tensor...')
	with open(DATASET_PKL_PATH.joinpath(f'lengths_tensor_method_pairwise_depd_model_{MODEL_NAME}_ndim_{NDIM_TOKEN_EMBEDDING}_conll_{conll_path.stem}_layer_{HIDDEN_LAYER}.pkl'), mode='wb') as file:
		pickle.dump(lengths_tensor, file)
	class mydataset(Dataset):
		def __init__(self,feats,labs,lengths):
			self.feats = feats
			self.labs = labs
			self.lengths = lengths
		def __len__(self):
			return len(self.labs)
		def __getitem__(self,idx):
			return self.feats[idx], self.labs[idx], self.lengths[idx]

	ds = mydataset(feats_tensor,labs_tensor, lengths_tensor)
	dl = DataLoader(ds,batch_size=BATCH_SIZE,shuffle=False)
	print(f'[load_dataset] dataloader instantiated: observations: {len(dl.dataset)}, batches: {len(dl)}')
	return dl

def prep_dataset_preload(conll_path):
	print(f'[load_dataset] loading feats_{conll_path.stem}.pkl')
	with open(DATASET_PKL_PATH.joinpath(f'feats_tensor_method_pairwise_depd_model_{MODEL_NAME}_ndim_{NDIM_TOKEN_EMBEDDING}_conll_{conll_path.stem}_layer_{HIDDEN_LAYER}.pkl'), mode='rb') as file:
		feats_tensor = pickle.load(file)
	print(f'[load_dataset] loading labs_{conll_path.stem}.pkl')
	with open(DATASET_PKL_PATH.joinpath(f'labs_tensor_method_pairwise_depd_model_{MODEL_NAME}_ndim_{NDIM_TOKEN_EMBEDDING}_conll_{conll_path.stem}_layer_{HIDDEN_LAYER}.pkl'), mode='rb') as file:
		labs_tensor = pickle.load(file)
	print(f'[load_dataset] loading lengths_{conll_path.stem}.pkl')	
	with open(DATASET_PKL_PATH.joinpath(f'lengths_tensor_method_pairwise_depd_model_{MODEL_NAME}_ndim_{NDIM_TOKEN_EMBEDDING}_conll_{conll_path.stem}_layer_{HIDDEN_LAYER}.pkl'), mode='rb') as file:
		lengths_tensor = pickle.load(file)
	class mydataset(Dataset):
		def __init__(self,feats,labs,lengths):
			self.feats = feats
			self.labs = labs
			self.lengths = lengths
		def __len__(self):
			return len(self.labs)
		def __getitem__(self,idx):
			return self.feats[idx], self.labs[idx], self.lengths[idx]

	dataset = mydataset(feats_tensor, labs_tensor, lengths_tensor)
	dataloader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=False)
	print(f'[load_dataset] {conll_path.stem} dataloader instantiated: observations: {len(dataloader.dataset)}, batches: {len(dataloader)}')
	return dataloader

if ONSITE_EMBEDDINGS == True:
    dataloader_dev = prep_dataset_onsite(DEV_CONLL_PATH)
    dataloader_train = prep_dataset_onsite(TRAIN_CONLL_PATH)
    dataloader_test = prep_dataset_onsite(TEST_CONLL_PATH)
else:
    dataloader_dev = prep_dataset_preload(DEV_CONLL_PATH)
    dataloader_train = prep_dataset_preload(TRAIN_CONLL_PATH)
    dataloader_test = prep_dataset_preload(TEST_CONLL_PATH) 