#!/bin/bash

# Quantum Computers' based Early Diseases Detection

# https://developer.nvidia.com/cuda-11-8-0-download-archive

# https://www.tensorflow.org/quantum/install

#echo 'export CUDNN_INCLUDE_DIR=/usr/local/cuda/include' >> ~/.bashrc
#echo 'export CUDNN_LIBRARY=/usr/local/cuda/lib64' >> ~/.bashrc
#echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
#source ~/.bashrc

source utils/colors.sh

sudo apt update && sudo apt upgrade -y
sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev htop
sudo apt install -y build-essential dkms linux-headers-$(uname -r)
sudo apt install -y unzip zip libcairo2 libcairo2-dev libpng-dev libfreetype6-dev python3-cairocffi
sudo apt install -y libcairo2-dev libpango1.0-dev fonts-dejavu-core graphviz
fc-cache -fv

sudo nvidia-persistenced --verbose
sudo systemctl enable nvidia-persistenced

ssh-keygen -t rsa -b 4096 -C "vgarnaga@suki.ai"
mkdir -p ~/.ssh
ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts
# git@github.com:GoogleCloudPlatform/compute-gpu-installation.git

echo -e "${CYAN}Checking if pyenv is installed...${NO_COLOR}"
if ! command -v pyenv &> /dev/null; then
    echo -e "${RED}pyenv is not installed. Installing pyenv...${NO_COLOR}"
    curl https://pyenv.run | bash
    echo -e 'export PYENV_ROOT="$HOME/.pyenv"\nexport PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo -e 'eval "$(pyenv init --path)"\neval "$(pyenv init -)"' >> ~/.bashrc
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init --path)"
    eval "$(pyenv init -)"
fi

source ~/.bashrc
pyenv install 3.10.16
pyenv global 3.10.16

pip install --upgrade pip
pip install nvidia-pyindex
pip install tensorrt-libs==8.6.1 --upgrade
pip install tensorflow[and-cuda]==2.15.0 --upgrade
python -c "import tensorflow as tf; print('GPUs: ', tf.config.list_physical_devices('GPU'))"
pip install pydot
pip install graphviz
pip install ucimlrepo
pip install scikit-learn
pip install seaborn
pip install statsmodels
pip install imblearn

### Install from sources 
# https://www.tensorflow.org/quantum/install

wget https://github.com/bazelbuild/bazel/releases/download/6.5.0/bazel_6.5.0-linux-x86_64.deb
sudo dpkg -i bazel_6.5.0-linux-x86_64.deb
sudo apt-mark hold bazel
bazel --version

pip install cirq pathos tensorflow-quantum cairosvg --upgrade

#pip install -U tfq-nightly
#pip install --force-reinstall "protobuf<3.21"
pushd .
cd ..
git clone https://github.com/tensorflow/quantum.git
cd quantum
pip install -r requirements.txt
bazel clean
bazel shutdown
./configure.sh
bazel build -c opt --cxxopt="-O3" --cxxopt="-march=native" --cxxopt="-std=c++17" --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=1" release:build_pip_package
bazel-bin/release/build_pip_package /tmp/tfquantum/
python3 -m pip install /tmp/tfquantum/name_of_generated_wheel.whl

popd
###


