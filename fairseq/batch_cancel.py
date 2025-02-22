import sys
import os

for line in open(sys.argv[1], 'r').readlines():
    # print(f'scancel {line}')
    print(f"cancelling {line}")
    os.system(f'scancel {line}')
        