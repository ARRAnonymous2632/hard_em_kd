import sys
import subprocess
import shlex
import os

local_path = sys.argv[1]
hdfs_path = sys.argv[2]


cmd = f'ls -l {local_path}'
local_file_list = subprocess.check_output(shlex.split(cmd)).decode('utf-8').split('\n')
local_file_list = [line.split()[-1] for line in local_file_list if len(line) > 0 and line.split()[-1].startswith('checkpoint')]
# print(local_file_list)

cmd = f'hdfs dfs -ls {hdfs_path}'
remote_file_list  = subprocess.check_output(shlex.split(cmd)).decode('utf-8').split('\n')
remote_file_list = [line.split()[-1] for line in remote_file_list if len(line) > 0 and line.split()[-1].startswith('hdfs')]

base_path = os.path.dirname(remote_file_list[0])
remote_file_list = [os.path.basename(file_name) for file_name in remote_file_list]
# print(remote_file_list)

for remote_file_name in remote_file_list:
    if remote_file_name not in local_file_list:
        if "_COPYING_" in remote_file_name:
            continue
        if ".log" in remote_file_name:
            continue
        print(f"Removeing {remote_file_name} from HDFS")
        cmd = f'hdfs dfs -rm {os.path.join(base_path, remote_file_name)} '
        subprocess.run(shlex.split(cmd))
        # print(cmd)
