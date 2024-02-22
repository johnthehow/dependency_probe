from pathlib import Path
# MODEL_PATH = Path(r'D:\transformers\bert\bert_base_uncased\mixed')
MODEL_PATH = Path(r'D:\transformers\bert\bert_large_uncased')
# MODEL_NAME = 'bert_base_uncased'
MODEL_NAME = 'bert_large_uncased'
# NDIM_TOKEN_EMBEDDING = 768
NDIM_TOKEN_EMBEDDING = 1024
HIDDEN_LAYER = 24
ONSITE_EMBEDDINGS = True
LEARNING_RATE = 0.001
PROBE_RANK = 64
BATCH_SIZE = 100
DEPD_DIRECTED = False
CONLL_PATH = Path(r"C:\Program Files\Python37\Lib\site-packages\thehow\dependency_probe\datasets\conll\gum")
# CONLL_PATH = Path(r"C:\Program Files\Python37\Lib\site-packages\thehow\dependency_probe\datasets\conll\ewt")
TRAIN_CONLL_PATH = CONLL_PATH.joinpath('en_gum-ud-train.conllu')
DEV_CONLL_PATH = CONLL_PATH.joinpath('en_gum-ud-dev.conllu')
TEST_CONLL_PATH = CONLL_PATH.joinpath('en_gum-ud-test.conllu')
# TRAIN_CONLL_PATH = CONLL_PATH.joinpath('en_ewt-ud-train.conllu')
# DEV_CONLL_PATH = CONLL_PATH.joinpath('en_ewt-ud-dev.conllu')
# TEST_CONLL_PATH = CONLL_PATH.joinpath('en_ewt-ud-test.conllu')
DATASET_PKL_PATH = Path(r"C:\Program Files\Python37\Lib\site-packages\thehow\dependency_probe\datasets\pkl")
PROBE_SAVEPATH = Path(r"C:\Program Files\Python37\Lib\site-packages\thehow\dependency_probe\models")
EPOCHS = 20
REPORTS_PATH = Path(r"C:\Program Files\Python37\Lib\site-packages\thehow\dependency_probe\reports")