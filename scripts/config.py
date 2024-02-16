from pathlib import Path
BERT_LOCAL_PATH = Path(r'D:\transformers\bert\bert_base_uncased\mixed')
NDIM_TOKEN_EMBEDDING = 768
LEARNING_RATE = 0.001
PROBE_RANK = 768
BATCH_SIZE = 100
# CONLL_PATH = Path(r'C:\Program Files\Python37\Lib\site-packages\thehow\papercodes\dependency_probe\datasets\conll')
# CONLL_PATH = Path('E:/同步空间/科研数据_镜像/语料库_20210320191310/UD_Universal_Dependencies_20231101125202/UD2.8/ud-treebanks-v2.8/ud-treebanks-v2.8/English-GUM/')
CONLL_PATH = Path(r"C:\Program Files\Python37\Lib\site-packages\thehow\dependency_probe\datasets\conll\gum")
DATASET_PKL_PATH = Path(r"C:\Program Files\Python37\Lib\site-packages\thehow\dependency_probe\datasets\pkl")
PROBE_SAVEPATH = Path(r"C:\Program Files\Python37\Lib\site-packages\thehow\dependency_probe\models")
EPOCHS = 20