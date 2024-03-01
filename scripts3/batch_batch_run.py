import subprocess
from config import *
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

for rank in [4,2,1]:
    cmd = f'sed -i -E "s/(PROBE_RANK = )(.+?)/\\1{rank}/g" config.py'
    print(f'[BATCH_BATCH_RUN] experimenting with rank {rank}')
    subprocess.run(cmd)
    subprocess.run(f'python batch_run.py')