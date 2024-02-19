import subprocess
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# for layer in range(0,13):
for layer in range(0,25):
    cmd = f'sed -i -E "s/(HIDDEN_LAYER = )(.+?)/\\1{layer}/g" config.py'
    print(cmd)
    subprocess.run(cmd)
    subprocess.run(f'python ui.py')