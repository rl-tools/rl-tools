set -e
(cd tests && ./download_data.sh)
(cd src/nn_models/port_checkpoint/example && ./setup.sh)
(cd src/nn_models/port_checkpoint/raptor && ./setup.sh)
if [ "$(uname)" = "Darwin" ]; then
cmake -B build -DCMAKE_BUILD_TYPE=Release
else
CUDACXX=/usr/local/cuda-13.1/bin/nvcc cmake -B build -DCMAKE_BUILD_TYPE=Release
fi
