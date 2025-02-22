#!/bin/bash

# Get extra packages



# Install torch+cuda
export CUDA_HOME=/usr/local/cuda
# pip3 install torch-1.10.1+cu113-cp37-cp37m-linux_x86_64.whl
# sudo pip3 install torch-1.10.1+cu113-cp37-cp37m-linux_x86_64.whl
# rm -rf torch-1.10.1+cu113-cp37-cp37m-linux_x86_64.whl
# pip3 install torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Compile FairSeq
sudo pip3 install --editable .

sudo chmod -R 1777 .
python3 setup.py build_ext --inplace

## install ctcdecode
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install .
cd ..

## Install apex
pip install Ninja
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..

# Install dag search
cd dag_search
bash install.sh
cd ..

## Others
pip install hydra-core==1.1.1

pip install sacremoses
pip install 'fuzzywuzzy[speedup]'
pip install git+https://github.com/dugu9sword/lunanlp.git
pip install omegaconf
pip install nltk
pip install sacrebleu==1.5.1
pip install sacrebleu[ja]
pip install scikit-learn scipy
pip install bitarray
pip install tensorboardX
pip install langdetect
# pip install git+https://github.com/chenyangh/sacrebleu.git@1.5.1

pip install scipy
unset http_proxy
unset https_proxy

echo "======================= CURRENT COMMIT ===================="
echo pulling recent commit 
git pull

git log --pretty=oneline -1


$@
