from transformers import BertModel, BertTokenizer
from config import *
from thehow.wordpiece2token.mapper import idx_mapper, matrix_denser, matrix_clssep_stripper
from thehow.tuda.depd_core import trees_gi
from collections import namedtuple, defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from torch import nn
import time

class timer(object):
	def __enter__(self):
		self.time_start = time.time()
	def __exit__(self,exc_type, exc_val,exc_tb):
		self.time_end = time.time()
		print(f'duration: {self.time_end - self.time_start}')


model = BertModel.from_pretrained(MODEL_PATH)
model.to('cuda')
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

def save_treeobjs():
	for conll_path in [TRAIN_CONLL_PATH, DEV_CONLL_PATH, TEST_CONLL_PATH]:
		tgi = trees_gi(conll_path)
		trees = [tree for tree in tgi]
		res_trees = []
		raw_tree_cnt = 0
		print(f'[DATASET] filtering mappable trees...')
		for tree in trees:
			print(f'prcessing {raw_tree_cnt}-th sent', end='\x1b\r')
			text = tree.text_lower
			user_tksent = [node.token.lower() for node in tree.nodes]
			bert_tksent = tokenizer(text, return_tensors='pt')
			wdps_tksent = tokenizer.convert_ids_to_tokens(bert_tksent['input_ids'][0][1:-1])
			idxmap = idx_mapper(wdps_tksent, user_tksent)
			if idxmap != None: # 筛选， 只允许可合并wordpiece的句子
				res_trees.append(tree)
			else:
				pass
			raw_tree_cnt += 1
		print(f'[DATASET] total trees mappable: {len(res_trees)}')
		print(f'[DATSSET] pickling mappable trees as tuda tree objects...')
		save_path = DATASET_PKL_PATH.joinpath('treeobjs').joinpath(f'MAPPABLE_TREEOBJS[DATA]{conll_path.stem}[MODEL]{MODEL_NAME}.pkl')
		with open(save_path, mode='wb') as file:
			pickle.dump(res_trees, file)
		print(f'[DATASET] treeobjs pickled at {save_path}')
	return

def save_depdmats():
	for conll_path in [TRAIN_CONLL_PATH, DEV_CONLL_PATH, TEST_CONLL_PATH]:
		treeobjs_path = DATASET_PKL_PATH.joinpath('treeobjs').joinpath(f'MAPPABLE_TREEOBJS[DATA]{conll_path.stem}[MODEL]{MODEL_NAME}.pkl')
		with open(treeobjs_path, mode='rb') as file:
			treeobjs = pickle.load(file)
		depd_mats = []
		tree_cnt = 0
		print(f'[DATASET] generating depd mats for mappable trees...')
		for tree in treeobjs:
			print(f'prcessing {tree_cnt}-th tree', end='\x1b\r')
			depd_mats.append(torch.tensor(tree.ajacency_matrix_weighted_absolute_full))
			tree_cnt += 1
		save_path = DATASET_PKL_PATH.joinpath('depdmats').joinpath(f'DEPDMATS[DATA]{conll_path.stem}[MODEL]{MODEL_NAME}.pkl')
		print(f'[DATSSET] pickling depd mats...')
		with open(save_path, mode='wb') as file:
			pickle.dump(depd_mats, file)
		print(f'[DATASET] depd mats pickled at {save_path}')
	return

def save_psdmats():
	for conll_path in [TRAIN_CONLL_PATH, DEV_CONLL_PATH, TEST_CONLL_PATH]:
		treeobjs_path = DATASET_PKL_PATH.joinpath('treeobjs').joinpath(f'MAPPABLE_TREEOBJS[DATA]{conll_path.stem}[MODEL]{MODEL_NAME}.pkl')
		with open(treeobjs_path, mode='rb') as file:
			treeobjs = pickle.load(file)
		depd_mats = []
		tree_cnt = 0
		print(f'[DATASET] generating psd mats for mappable trees...')
		for tree in treeobjs:
			print(f'prcessing {tree_cnt}-th tree', end='\x1b\r')
			depd_mats.append(torch.tensor(tree.ajacency_matrix_weighted_absolute_full))
			tree_cnt += 1
		save_path = DATASET_PKL_PATH.joinpath('psdmats').joinpath(f'PSDMATS[DATA]{conll_path.stem}[MODEL]{MODEL_NAME}.pkl')
		print(f'[DATSSET] pickling psd mats...')
		with open(save_path, mode='wb') as file:
			pickle.dump(depd_mats, file)
		print(f'[DATASET] psd mats pickled at {save_path}')
	return
	
def save_lengths():
	for conll_path in [TRAIN_CONLL_PATH, DEV_CONLL_PATH, TEST_CONLL_PATH]:
		treeobjs_path = DATASET_PKL_PATH.joinpath('treeobjs').joinpath(f'MAPPABLE_TREEOBJS[DATA]{conll_path.stem}[MODEL]{MODEL_NAME}.pkl')
		with open(treeobjs_path, mode='rb') as file:
			treeobjs = pickle.load(file)
		lengths = []
		tree_cnt = 0
		for tree in treeobjs:
			print(f'[DATASET] processing {tree_cnt}-th tree', end='\x1b\r')
			lengths.append(tree.len)
			tree_cnt += 1
		save_path = DATASET_PKL_PATH.joinpath('lengths').joinpath(f'LENGTHS[DATA]{conll_path.stem}[MODEL]{MODEL_NAME}.pkl')
		print(f'[DATSSET] pickling lengths...')
		with open(save_path, mode='wb') as file:
			pickle.dump(lengths, file)
		print(f'[DATASET] lengths pickled at {save_path}')
	return

def save_embeddings():
	print(f'[DATASET] calculating and saving embeddings')
	for conll_path in [TRAIN_CONLL_PATH, DEV_CONLL_PATH, TEST_CONLL_PATH]:
		treeobjs_path = DATASET_PKL_PATH.joinpath('treeobjs').joinpath(f'MAPPABLE_TREEOBJS[DATA]{conll_path.stem}[MODEL]{MODEL_NAME}.pkl')
		with open(treeobjs_path, mode='rb') as file:
			treeobjs = pickle.load(file)
		tree_cnt = 0
		texts = []
		lengths_token = []
		user_tksents = []
		bert_tksents = []
		wdps_tksents = []
		lengths_wdps = []
		idxmaps = []
		for tree in treeobjs:
			print(f'[DATASET] processing {tree_cnt}-th sent', end='\x1b\r')
			text = tree.text_lower
			texts.append(text)
			lengths_token.append(tree.len)
			user_tksent = [node.token.lower() for node in tree.nodes]
			user_tksents.append(user_tksent)
			bert_tksent = tokenizer(text, return_tensors='pt')
			bert_tksents.append(bert_tksent)
			wdps_tksent = tokenizer.convert_ids_to_tokens(bert_tksent['input_ids'][0][1:-1])
			wdps_tksents.append(wdps_tksent)
			lengths_wdps.append(len(wdps_tksent)+2)
			idxmap = idx_mapper(wdps_tksent, user_tksent)
			idxmaps.append(idxmap)
			tree_cnt += 1
		print(f'tokenizing with bert tokenizer')
		batch_tksents = tokenizer(texts, padding=True, return_tensors='pt')
		batch_tksents.to('cuda')
		class embed_dataset(Dataset):
			def __init__(self, tksents):
				self.tksents = tksents
				self.input_ids = self.tksents['input_ids']
				self.token_type_ids = self.tksents['token_type_ids']
				self.attention_mask = self.tksents['attention_mask']
			def __len__(self):
				return len(self.input_ids)
			def __getitem__(self,idx):
				seg = dict(input_ids = self.input_ids[idx], token_type_ids = self.token_type_ids[idx], attention_mask = self.attention_mask[idx])
				return seg
		print(f'instantiating dataset object')
		embed_ds = embed_dataset(batch_tksents)
		embed_dl = DataLoader(embed_ds, batch_size = 800, shuffle=False)
		embeds = defaultdict(list)
		batch_cnt = 0
		for batch in embed_dl:
			with timer() as tm:
				print(f'calculating {batch_cnt}-th batch')
				with torch.no_grad():
					batch_output = model(**batch, output_hidden_states = True)
				batch_cnt += 1
				torch.cuda.empty_cache()
				for i in range(MODEL_NLAYER+1):
					embeds[i].append(batch_output['hidden_states'][i].to('cpu'))
		embeds_cats = [torch.cat(embeds[i], axis=0) for i in range(MODEL_NLAYER+1)]
		embeds_cats_zigzag = defaultdict(list)
		print('densing and saving embeddings')
		for layer in range(MODEL_NLAYER+1):
			print(f'layer {layer}')
			for idx, mat in enumerate(embeds_cats[layer]):
				raw_sent_mat = mat[:lengths_wdps[idx]]
				stripmat = matrix_clssep_stripper(raw_sent_mat, scale=False)
				densemat = matrix_denser(stripmat, idxmaps[idx], rowwise=True)
				embeds_cats_zigzag[i].append(densemat)
		return

def load_treeobjs():
	print(f'[DATASET] loading treeobjs for {TRAIN_CONLL_PATH.stem}, {DEV_CONLL_PATH.stem} and {TEST_CONLL_PATH.stem}')
	treeobjs = []
	for conll_path in [TRAIN_CONLL_PATH, DEV_CONLL_PATH, TEST_CONLL_PATH]:
		filename =  DATASET_PKL_PATH.joinpath('treeobjs').joinpath(f'MAPPABLE_TREEOBJS[DATA]{conll_path.stem}[MODEL]{MODEL_NAME}.pkl')
		with open(filename, mode='rb') as file:
			treeobjs.append(pickle.load(file))
	return treeobjs

def load_depdmats():
	print(f'[DATASET] loading depdmats for {TRAIN_CONLL_PATH.stem}, {DEV_CONLL_PATH.stem} and {TEST_CONLL_PATH.stem}')
	depdmats = []
	for conll_path in [TRAIN_CONLL_PATH, DEV_CONLL_PATH, TEST_CONLL_PATH]:
		filename =  DATASET_PKL_PATH.joinpath('depdmats').joinpath(f'DEPDMATS[DATA]{conll_path.stem}[MODEL]{MODEL_NAME}.pkl')
		with open(filename, mode='rb') as file:
			depdmats.append(pickle.load(file))
	return depdmats

def load_psdmats():
	print(f'[DATASET] loading psdmats for {TRAIN_CONLL_PATH.stem}, {DEV_CONLL_PATH.stem} and {TEST_CONLL_PATH.stem}')
	psdmats = []
	for conll_path in [TRAIN_CONLL_PATH, DEV_CONLL_PATH, TEST_CONLL_PATH]:
		filename =  DATASET_PKL_PATH.joinpath('psdmats').joinpath(f'PSDMATS[DATA]{conll_path.stem}[MODEL]{MODEL_NAME}.pkl')
		with open(filename, mode='rb') as file:
			psdmats.append(pickle.load(file))
	return psdmats

def load_lengths():
	print(f'[DATASET] loading lengths for {TRAIN_CONLL_PATH.stem}, {DEV_CONLL_PATH.stem} and {TEST_CONLL_PATH.stem}')
	lengths = []
	for conll_path in [TRAIN_CONLL_PATH, DEV_CONLL_PATH, TEST_CONLL_PATH]:
		filename =  DATASET_PKL_PATH.joinpath('lengths').joinpath(f'LENGTHS[DATA]{conll_path.stem}[MODEL]{MODEL_NAME}.pkl')
		with open(filename, mode='rb') as file:
			lengths.append(pickle.load(file))
	return lengths

def load_embeddings():
	print(f'[DATASET] loading embeddings for {TRAIN_CONLL_PATH.stem}, {DEV_CONLL_PATH.stem} and {TEST_CONLL_PATH.stem}')
	embeddings = []
	for conll_path in [TRAIN_CONLL_PATH, DEV_CONLL_PATH, TEST_CONLL_PATH]:
		filename =  DATASET_PKL_PATH.joinpath('embeddings').joinpath(f'EMBEDDINGS[MODEL]{MODEL_NAME}[DATA]{conll_path.stem}[LAYER]{HIDDEN_LAYER}.pkl')
		with open(filename, mode='rb') as file:
			embeddings.append(pickle.load(file))
	return embeddings

def assemble_datasets():
	class mydataset(Dataset):
		def __init__(self, observations):
			self.observations = observations
		def __len__(self):
			return len(self.observations)
		def __getitem__(self,idx):
			return self.observations[idx]
		
	def custom_pad(batch_observations):
		seqs = [x[0].embeddings for x in batch_observations]
		lengths = torch.tensor([len(x) for x in seqs])
		seqs = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
		label_shape = batch_observations[0][1].shape
		maxlen = int(max(lengths))
		label_maxshape = [maxlen for x in label_shape]
		labels = [-torch.ones(*label_maxshape) for x in seqs]
		for index, x in enumerate(batch_observations):
			length = x[1].shape[0]
			if len(label_shape) == 1:
				labels[index][:length] = x[1]
			elif len(label_shape) == 2:
				labels[index][:length,:length] = x[1]
			else:
				raise ValueError("Labels must be either 1D or 2D right now; got either 0D or >3D")
		labels = torch.stack(labels)
		return seqs, labels, lengths, batch_observations
	
	observation_class = namedtuple('Observation', ['index', 'sentence', 'lemma_sentence', 'upos_sentence', 'xpos_sentence', 'morph', 'head_indices', 'governance_relations', 'secondary_relations', 'extra_info', 'embeddings'])

	treeobjs_triad = load_treeobjs()
	if PATH_DISTANCE == False:
		labmats_triad = load_depdmats()
	else:
		labmats_triad = load_psdmats()
	embeddings_triad = load_embeddings()
	dataloaders_triad = []
	print(f'[DATASET] assembling observations for {TRAIN_CONLL_PATH.stem}, {DEV_CONLL_PATH.stem} and {TEST_CONLL_PATH.stem}')
	for treeobjs, depdmats, embeddings in zip(treeobjs_triad, labmats_triad, embeddings_triad):
		observations = []
		for idx, tree in enumerate(treeobjs):
			obs_nodes = tree.nodes
			obs_index = tuple([i.id for i in obs_nodes])
			obs_sentence = tuple([i.token for i in obs_nodes])
			obs_lemma_sentence = tuple([i.lemma for i in obs_nodes])
			obs_upos = tuple([i.upos for i in obs_nodes])
			obs_xpos = tuple([i.xpos for i in obs_nodes])
			obs_morph = tuple([i.feats for i in obs_nodes])
			obs_head_indices = tuple([i.headid for i in obs_nodes])
			obs_governance_relations = tuple([i.deprel for i in obs_nodes])
			obs_secondary_relations = tuple([i.deps for i in obs_nodes])
			obs_extra_info = tuple([i.misc for i in obs_nodes])
			densemat = embeddings[idx].type(torch.float32)
			depdmat = depdmats[idx].type(torch.float32)
			tree_observation = observation_class(obs_index,obs_sentence,obs_lemma_sentence,obs_upos,obs_xpos,obs_morph,obs_head_indices,obs_governance_relations, obs_secondary_relations,obs_extra_info, densemat)
			observations.append((tree_observation,depdmat))
		ds = mydataset(observations)
		dl = DataLoader(ds,batch_size=BATCH_SIZE,collate_fn=custom_pad, shuffle=False)
		dataloaders_triad.append(dl)
		print(f'[DATASET] dataloader instantiated: observations: {len(dl.dataset)}, batches: {len(dl)}')
	return dataloaders_triad

# dataloaders_triad = assemble_datasets()

if __name__ == '__main__':
	# save_treeobjs()
	# save_depdmats()
	# save_lengths()
	# save_embeddings()
	save_psdmats()