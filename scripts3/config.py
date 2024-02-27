from pathlib import Path

# [GENERAL SETTINGS]
ROOT_PATH = Path(r'C:\Program Files\Python311\Lib\site-packages\thehow\dependency_probe')

# [MODEL SETTINGS]
# MODEL_NAME = 'bert_base_uncased'
MODEL_NAME = 'bert_large_uncased'
if MODEL_NAME == 'bert_large_uncased':
    MODEL_PATH = Path(r'D:\transformers\bert\bert_large_uncased')
    MODEL_NLAYER = 24
    NDIM_TOKEN_EMBEDDING = 1024
elif MODEL_NAME == 'bert_base_uncased':
    MODEL_PATH = Path(r'D:\transformers\bert\bert_base_uncased\mixed')
    NDIM_TOKEN_EMBEDDING = 768
    MODEL_NLAYER = 12
MODEL_CUDA = False
ONSITE_EMBEDDINGS = False

# [PROBE SETTINGS]
HIDDEN_LAYER = 2
ONSITE_PROBES = True
LEARNING_RATE = 0.001
PROBE_RANK = 128
PROBE_CUDA = False
BATCH_SIZE = 100
PROBE_SAVEPATH = ROOT_PATH.joinpath("probes")
EPOCHS = 20
PROBE_LOSS_POW = 2

# [DATA SETTINGS]
# CONLL_ALIAS = 'ewt'
# CONLL_ALIAS = 'gum'
CONLL_ALIAS = 'wsj'
CONLL_PATH = ROOT_PATH.joinpath(f"datasets\conll\{CONLL_ALIAS}")
TRAIN_CONLL_PATH = CONLL_PATH.joinpath(f'en_{CONLL_ALIAS}-ud-train.conllu')
DEV_CONLL_PATH = CONLL_PATH.joinpath(f'en_{CONLL_ALIAS}-ud-dev.conllu')
TEST_CONLL_PATH = CONLL_PATH.joinpath(f'en_{CONLL_ALIAS}-ud-test.conllu')
DEPD_DIRECTED = False
DATASET_PKL_PATH = ROOT_PATH.joinpath("datasets\pkl")

# [REPORT SETTINGS]
REPORTS_PATH = ROOT_PATH.joinpath("reports")

# [METHOD]
METHOD = 'pairwisedepd'
# METHOD = 'deprelonly'
