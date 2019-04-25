# cuda安装
./cuda_9.0.176_384.81_linux.run

export PATH=$HOME/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/cuda-9.0/lib64/

nvcc -V

# cuDNN安装
tar -zxvf cudnn-9.0-linux-x64-v7.5.0.56.tgz

cp cuda/include/cudnn.h ~/cuda-9.0/include/

cp cuda/lib64/libcudnn* ~/cuda-9.0/lib64

chmod 777 ~/cuda-9.0/include/cudnn.h ~/cuda-9.0/lib64/libcudnn*

cat ~/cuda-9.0/include/cudnn.h | grep CUDNN_MAJOR -A5