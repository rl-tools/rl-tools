CUDACXX=/home/jonas/.local/opt/cuda-12.8.2/bin/nvcc CUDAARCHS=90 cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$HOME/.local/opt/cudnn"
