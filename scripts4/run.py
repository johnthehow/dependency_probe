import torch
from probe import looper
from report import report_writer
from dataset import dataloaders_triad
from config import *
from collections import defaultdict

dataloader_train, dataloader_dev, dataloader_test = dataloaders_triad

if ONSITE_PROBES == True:
    probe_trained = looper(EPOCHS)
else:
    probe_filename = f'PROBE[MODEL]{MODEL_NAME}[LAYER]{HIDDEN_LAYER}[DATA]{TRAIN_CONLL_PATH.stem}[RANK]{PROBE_RANK}[MEHOD]{METHOD}[DEPDDIRECTED]{str(DEPD_DIRECTED).lower}[EPOCHS]{EPOCHS}[BATCHSIZE]{BATCH_SIZE}[LR]{LEARNING_RATE}[LOSSPOW]{PROBE_LOSS_POW}.pth'
    probe_filepath = PROBE_SAVEPATH.joinpath(probe_filename)
    print(f'[RUN] loading saved probe: {probe_filename}')
    probe = torch.load(probe_filepath)
    epoch_losses = defaultdict(list)
    probe_trained = (probe, epoch_losses, probe_filename)


report_writer(probe_trained, dataloader_test)