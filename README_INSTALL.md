安装说明:

按照以下的顺序来进行安装：


```
einopsextensions/rl_labc#我自己使用的操作系统是Ubuntu24.04，当然22.04或者20.04也是可以的。
#如果你没有conda，我非常推荐使用miniconda3来安装整个虚拟环境
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh



#在你安装好conda之后，你可以创建
conda create -n genesis python=3.10
conda activate genesis

#请检查自己的nvida显卡驱动版本以及cuda版本，从而确定pytorch的安装版本：
nvidia-smi
nvcc -V
#现在我们可以安装pytorch了
#cuda11版本
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#cuda12版本
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#在安装完这些之后，我们可以下载这个仓库并安装
git clone https://github.com/Huisouan/Genesis
cd Genesis
pip install -e .

#如果你想要使用强化学习，还需要安装两个本地库：
cd extensions/rl_lab
pip install -e.
cd ..
cd rsl_rl
pip install -e.

#最后我们需要安装几个额外的pip包来保证所有代码都能正常运行

pip install tensorboard,einops,open3d


```
