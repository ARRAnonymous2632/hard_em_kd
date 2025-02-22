#!/bin/bash

cuda_home=${1-/usr/local/cuda}


# Install torch+cuda
export CUDA_HOME=$cuda_home




pip3 install tensorboardX
pip install numba tensorflow==2.15.1 transformers==4.34.0


pip3 install Ninja packaging
python -m pip install Cython
pip3 install hydra-core==1.1.1
pip3 install sacremoses
pip3 install 'fuzzywuzzy[speedup]'
pip3 install git+https://github.com/dugu9sword/lunanlp.git
pip3 install omegaconf
pip3 install nltk
pip3 install sacrebleu==1.5.1
# pip3 install sacrebleu[ja]
pip3 install scikit-learn scipy
pip3 install bitarray
# pip3 install tensorflow==2.4.1
# pip install git+https://github.com/chenyangh/sacrebleu.git@1.5.1

pip3 install scipy
pip3 install wandb
pip install rouge
pip install rouge_score

# pip install streamlit==0.62.0
# pip install pyarrow==0.12.0
# pip install --extra-index-url https://pypi.fury.io/arrow-nightlies/ --prefer-binary --pre pyarrow

pip install git+https://github.com/tagucci/pythonrouge.git
pip install https://github.com/kpu/kenlm/archive/master.zip
# pip install evaluate #  
pip install bert-score
#cconda upgrade numpy

pip install numpy==1.23.0 # +computecanada


python setup.py build_ext --inplace
# python setup.py install
pip3 install --editable .

## install ctcdecode
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode 
mkdir -p ctcdecode/_ext
pip install .
cd ..

## Install apex
# pip3 install Ninja
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 6943fd26e04c59327de32592cf5af68be8f5c44e
pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..

cd dag_search
bash install.sh
cd ..

## Others


# wandb login 6dc39255e119c5d8d2e39a809e9edfe896c3633e

# module load boost
# module load cmake
# module load StdEnv/2020 gcc/11.3.0
# module load cuda/12.2


# module load python-build-bundle/2023b

python setup.py build_ext --inplace
# python setup.py install
pip3 install --editable .

pip install --upgrade pip

pip install sacrebleu # [ja]

pip install git+https://github.com/dugu9sword/lunanlp.git

pip install git+https://github.com/tagucci/pythonrouge.git

pip install --upgrade setuptools


# module load cuda/11.7
# export CUDA_HOME=/cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/cudacore/11.7.0


# module load cuda/12.2
# export CUDA_HOME=/cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/cudacore/12.2.2


# module load arrow/13.0.0