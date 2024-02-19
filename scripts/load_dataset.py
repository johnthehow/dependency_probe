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
	user_tksents = []
	wdps_tksents = []
	idxmaps = []
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
			user_tksents.append(user_tksent)
			wdps_tksents.append(wdps_tksent)
			idxmaps.append(idxmap)
		else:
			# print(f'[load_dataset] {tree_cnt}-th wordpiece-tokenized sent not alignable')
			# print(f'[load_dataset] {text}')
			pass
		tree_cnt += 1

	all_pairs = []
	tree_cnt = 0
	for tree in mappable_trees:
		tree_pairs = []
		for node in tree.nodes:
			if not node.isroot:
				tree_pairs.append((node.token.lower(), tree.get_nodes_by_ids([node.headid])[0].token.lower(), node.id-1, node.headid-1, node.depd_abs, node.depd_directed, tree.len, densemats[tree_cnt][node.id-1], densemats[tree_cnt][node.headid-1]))
		all_pairs.append(tree_pairs)
		tree_cnt += 1

	observation_class = namedtuple('observation', field_names = ['token', 'head', 'token_pos', 'head_pos', 'depd_abs', 'depd_directed', 'sent_len','token_emb', 'head_emb'])

	all_obs = [[observation_class(*node) for node in tree] for tree in all_pairs]

	print(f'[load_dataset] total trees instantiated as observation objects: {len(all_obs)}')
	print(f'[load_dataset] assembling observation objects for feats')
	max_sentlen = max([len(sent) for sent in all_obs])
	feats = [torch.zeros(max_sentlen,2,NDIM_TOKEN_EMBEDDING) for sent in all_obs]
	for idx_sent, sent in enumerate(all_obs):
		for idx_token, token in enumerate(sent):
			feats[idx_sent][idx_token][0] = token.token_emb.detach()
			feats[idx_sent][idx_token][1] = token.head_emb.detach()
	feats_tensor = torch.stack(feats,dim=0)
	print(f'[load_dataset] shape of feats_tensor: {feats_tensor.shape}')
	with open(DATASET_PKL_PATH.joinpath(f'feats_{conll_path.stem}_{MODEL_NAME}_layer_{HIDDEN_LAYER}.pkl'), mode='wb') as file:
		pickle.dump(feats_tensor,file)
	print(f'[load_dataset] assembling observation objects for labs')
	labs = [torch.zeros(max_sentlen) for sent in all_obs]
	if DEPD_DIRECTED == False:
		for idx_sent, sent in enumerate(all_obs):
			for idx_token, token in enumerate(sent):
				labs[idx_sent][idx_token] = token.depd_abs
	else:
		for idx_sent, sent in enumerate(all_obs):
			for idx_token, token in enumerate(sent):
				labs[idx_sent][idx_token] = token.depd_directed		
	labs_tensor = torch.stack(labs,dim=0)
	with open(DATASET_PKL_PATH.joinpath(f'labs_{conll_path.stem}_{MODEL_NAME}_layer_{HIDDEN_LAYER}.pkl'), mode='wb') as file:
		pickle.dump(labs_tensor,file)
	print(f'[load_dataset] shape of labs_tensor: {labs_tensor.shape}')
	lengths = [len(sent) for sent in all_obs]
	lengths_tensor = torch.tensor(lengths,dtype=torch.float)
	with open(DATASET_PKL_PATH.joinpath(f'lengths_{conll_path.stem}_{MODEL_NAME}_layer_{HIDDEN_LAYER}.pkl'), mode='wb') as file:
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
	with open(DATASET_PKL_PATH.joinpath(f'feats_{conll_path.stem}_{MODEL_NAME}_layer_{HIDDEN_LAYER}.pkl'), mode='rb') as file:
		feats_tensor = pickle.load(file)
	print(f'[load_dataset] loading labs_{conll_path.stem}.pkl')
	with open(DATASET_PKL_PATH.joinpath(f'labs_{conll_path.stem}_{MODEL_NAME}_layer_{HIDDEN_LAYER}.pkl'), mode='rb') as file:
		labs_tensor = pickle.load(file)
	print(f'[load_dataset] loading lengths_{conll_path.stem}.pkl')	
	with open(DATASET_PKL_PATH.joinpath(f'lengths_{conll_path.stem}_{MODEL_NAME}_layer_{HIDDEN_LAYER}.pkl'), mode='rb') as file:
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