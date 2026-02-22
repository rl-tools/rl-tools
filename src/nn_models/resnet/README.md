
```
g++ -I include -I /usr/include/hdf5/serial/ -I .dependencies/build/highfive-src/include -I .dependencies/build/stb-src/ src/nn_models/resnet/resnet_inference.cpp -std=c++17 -Ofast -DRL_TOOLS_BACKEND_ENABLE_OPENBLAS -DRL_TOOLS_ENABLE_HDF5=ON -o a.out -lblas -L /usr/lib/x86_64-linux-gnu/hdf5/serial/ -lhdf5
```
```
./a.out tests/data/IMG_8734_224x224.png tests/data/resnet18_test_data.h5 tests/data/imagenet-1k-classes.txt
```