FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y wget gnupg
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
| gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
RUN echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list
RUN apt-get update && apt-get install -y intel-oneapi-mkl-devel-2023.1.0
RUN apt-get update && apt-get install -y cmake build-essential libhdf5-dev libboost-all-dev protobuf-compiler libprotobuf-dev python3 python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install tensorboard==2.12.2
RUN echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-16 main" >> /etc/apt/sources.list.d/llvm.list \
    && echo "deb-src http://apt.llvm.org/jammy/ llvm-toolchain-jammy-16 main" >> /etc/apt/sources.list.d/llvm.list
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
RUN apt-get update && apt-get install -y clang-16
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-16 100
RUN update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-16 100
RUN update-alternatives --set clang /usr/bin/clang-16
RUN apt-get remove --purge --auto-remove -y cmake
RUN apt-get update && apt-get install -y software-properties-common lsb-release
RUN apt-get clean all
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
RUN apt-get update && apt-get install -y cmake git # git is required for MuJoCo (it uses "findOrFetch" in its CMakeLists.txt)
RUN mkdir build
WORKDIR /build
