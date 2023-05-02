## Environment setup on Ubuntu (tested with 20.04)
### Basic requirements
```
apt-get update
apt-get install -y cmake build-essential
```
With this setup you can already run the basic rl example:

[__00 - Basic RL__: Dependency-free RL](../examples/00_BASIC_RL.MD)

### Extended requirements
Requirements for HDF5 and Tensorboard logging:
```
apt-get install -y libhdf5-dev libboost-all-dev
apt-get install -y protobuf-compiler libprotobuf-dev
```
Installing Tensorboard
```
apt-get install -y python3 python3-pip
pip3 install --upgrade pip
pip3 install tensorboard==2.12.2
```

Intel MKL: BLAS library to speed up matrix multiplications using dedicated CPU instructions. The following adds the official Intel repositories to `apt` and installs the MKL library:
```
apt-get update
apt-get install -y wget gnupg
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
| gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list
apt-get update
apt-get install -y intel-oneapi-mkl-devel-2023.1.0
```

MuJoCo UI: The MuJoCo UI depends on `glfw3` (this is only required if you want to use the MuJoCo UI):
```
apt-get update && apt-get install -y libglfw3-dev
```

From here you can continue with the instructions for the accelerated examples:

[__01 - BLAS__: Accelerated RL](../examples/01_BLAS.MD)
