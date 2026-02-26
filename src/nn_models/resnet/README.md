
```
g++ -I include -I /usr/include/hdf5/serial/ -I .dependencies/build/highfive-src/include -I .dependencies/build/stb-src/ src/nn_models/resnet/resnet_inference.cpp -std=c++17 -Ofast -DRL_TOOLS_BACKEND_ENABLE_OPENBLAS -DRL_TOOLS_ENABLE_HDF5=ON -lblas -L /usr/lib/x86_64-linux-gnu/hdf5/serial/ -lhdf5
```
```
./a.out tests/data/IMG_8734_224x224.png tests/data/resnet18_test_data.h5 tests/data/imagenet-1k-classes.txt
```


Clang AST Dump
```
clang++ -I include -I /usr/include/hdf5/serial/ -I .dependencies/build/highfive-src/include -I .dependencies/build/stb-src/ src/nn_models/resnet/resnet_inference.cpp -std=c++17 -DRL_TOOLS_BACKEND_ENABLE_OPENBLAS -DRL_TOOLS_ENABLE_HDF5=ON -Xclang -ast-dump -Xclang -ast-dump-filter=rl_tools -fsyntax-only > resnet_ast.txt
```



Training ImageNet
```
CUDACXX=$HOME/.local/opt/cuda/bin/nvcc cmake -B build -DCMAKE_PREFIX_PATH="$HOME/.local/opt/cudnn"
cmake --build build -j8 --target nn_models_resnet_imagenet_training_cuda
 ./build/src/nn_models/resnet/cuda/nn_models_resnet_imagenet_training_cuda --dataset-dir /scr/jonas/imagenet-1k
```