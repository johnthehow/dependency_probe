from probe import looper
from report import reporter
from config import *
from load_dataset import dataloader_test

probe_trained = looper(EPOCHS)

reporter(probe_trained, dataloader_test)