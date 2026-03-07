set -e
(cd tests && ./download_data.sh)
(cd src/nn_models/port_checkpoint/example && ./setup.sh)
(cd src/nn_models/port_checkpoint/raptor && ./setup.sh)
CUDACXX=/usr/local/cuda-13.1/bin/nvcc cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DRL_TOOLS_ENABLE_TESTS=ON \
  -DRL_TOOLS_EXPERIMENTAL=ON \
  -DRL_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO=ON \
  -DRL_TOOLS_NUMERIC_TYPES_ENABLE_BF16=ON \
  -DRL_TOOLS_ENABLE_TAR=ON
cmake --build build -j10
