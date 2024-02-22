import subprocess
from config import *
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

for layer in range(MODEL_NLAYER+1):
    cmd = f'sed -i -E "s/(HIDDEN_LAYER = )(.+?)/\\1{layer}/g" config.py'
    subprocess.run(cmd)
    subprocess.run(f'python run.py')