from transformers import BertModel, BertTokenizer
from config import *
from thehow.wordpiece2token.mapper import idx_mapper, matrix_denser, matrix_clssep_stripper
from thehow.tuda.depd_core import trees_gi
from collections import namedtuple
from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from torch.optim import Adam

model = BertModel.from_pretrained(BERT_LOCAL_PATH)
tokenizer = BertTokenizer.from_pretrained(BERT_LOCAL_PATH)

def prep_dataset(conll_path):
	tgi = trees_gi(conll_path)
	trees = [tree for tree in tgi]
	mappable_trees = []
	densemats = []
	texts = []
	user_tksents = []
	wdps_tksents = []
	idxmaps = []
	tree_cnt = 0
	for tree in trees:
		text = tree.text_lower
		user_tksent = [node.token.lower() for node in tree.nodes]
		bert_tksent = tokenizer(text, return_tensors='pt')
		wdps_tksent = tokenizer.convert_ids_to_tokens(bert_tksent['input_ids'][0][1:-1])
		idxmap = idx_mapper(wdps_tksent, user_tksent)
		if idxmap != None:
			bert_output = model(**bert_tksent)
			last_hidden = bert_output['last_hidden_state'][0] # 句长+2*768
			stripmat = matrix_clssep_stripper(last_hidden, scale=False)
			densemat = matrix_denser(stripmat, idxmap, rowwise=True)
			densemats.append(densemat)
			mappable_trees.append(tree)
			texts.append(text)
			user_tksents.append(user_tksent)
			wdps_tksents.append(wdps_tksent)
			idxmaps.append(idxmap)
		else:
			print(f'{tree_cnt}')
			print(f'{text}')
		tree_cnt += 1

	all_pairs = []
	tree_cnt = 0
	for tree in mappable_trees:
		tree_pairs = []
		for node in tree.nodes:
			if not node.isroot:
				tree_pairs.append((node.token.lower(), tree.get_nodes_by_ids([node.headid])[0].token.lower(), node.id-1, node.headid-1, node.depd_abs, tree.len, densemats[tree_cnt][node.id-1], densemats[tree_cnt][node.headid-1]))
		all_pairs.append(tree_pairs)
		tree_cnt += 1

	observation_class = namedtuple('observation', field_names = ['token', 'head', 'token_pos', 'head_pos', 'depd_abs', 'sent_len','token_emb', 'head_emb'])

	all_obs = [[observation_class(*node) for node in tree] for tree in all_pairs]

	max_sentlen = max([len(sent) for sent in all_obs])

	feats = [torch.zeros(max_sentlen,2,768) for sent in all_obs]

	for idx_sent, sent in enumerate(all_obs):
		for idx_token, token in enumerate(sent):
			feats[idx_sent][idx_token][0] = token.token_emb.detach()
			feats[idx_sent][idx_token][1] = token.head_emb.detach()

	feats_tensor = torch.stack(feats,dim=0)

	labs = [torch.zeros(max_sentlen) for sent in all_obs]

	for idx_sent, sent in enumerate(all_obs):
		for idx_token, token in enumerate(sent):
			labs[idx_sent][idx_token] = token.depd_abs

	labs_tensor = torch.stack(labs,dim=0)

	lengths = [len(sent) for sent in all_obs]

	lengths_tensor = torch.tensor(lengths,dtype=torch.float)
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

	dl = DataLoader(ds,batch_size=20,shuffle=False)
	return dl