import torch
from probe import looper
from report import report_writer
from config import *
from load_dataset import dataloader_test
from collections import defaultdict

if ONSITE_PROBES == True:
    probe_trained = looper(EPOCHS)
else:
    probe_filename = f'probe_{MODEL_NAME}_ndim_{NDIM_TOKEN_EMBEDDING}_rank_{PROBE_RANK}_directed_{str(DEPD_DIRECTED)}.pth'
    probe_filepath = PROBE_SAVEPATH.joinpath(probe_filename)
    print(f'[report] loading saved probe: {probe_filename}')
    probe = torch.load(probe_filepath)
    epoch_losses = defaultdict(list)
    probe_trained = (probe, epoch_losses, probe_filename)


report_writer(probe_trained, dataloader_test)