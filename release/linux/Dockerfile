FROM nvidia/cuda:11.8.0-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y wget gnupg
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
| gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
RUN echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list
RUN apt-get update
RUN apt-get install -y intel-oneapi-mkl-devel-2023.1.0
RUN apt-get install -y cmake build-essential git vim tmux
RUN apt-get install -y libhdf5-dev libboost-all-dev
RUN apt-get install -y protobuf-compiler libprotobuf-dev
RUN apt-get install -y python3 python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install tensorboard==2.12.2
#RUN apt-get install -y clang
ENV UBUNTU_VERSION=focal
RUN echo "deb http://apt.llvm.org/$UBUNTU_VERSION/ llvm-toolchain-$UBUNTU_VERSION-16 main" >> /etc/apt/sources.list.d/llvm.list \
    && echo "deb-src http://apt.llvm.org/$UBUNTU_VERSION/ llvm-toolchain-$UBUNTU_VERSION-16 main" >> /etc/apt/sources.list.d/llvm.list
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
RUN apt-get update
RUN apt-get install -y clang-16
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-16 100
RUN update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-16 100
RUN update-alternatives --set clang /usr/bin/clang-16
RUN apt-get remove --purge --auto-remove -y cmake
RUN apt-get update
RUN apt-get install -y software-properties-common lsb-release
RUN apt-get clean all
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
RUN apt-get update
RUN apt-get install -y cmake
RUN apt-get update
RUN apt-get install -y libglfw3-dev

# Conan
RUN pip3 install conan==2.0.4
# Glfw dependencies (copied from conan error message when they were not installed)
RUN apt-get install -y libxcomposite-dev libxcursor-dev libxdamage-dev libxfixes-dev libxi-dev libxinerama-dev libxkbfile-dev libxmuu-dev libxres-dev libxss-dev libxtst-dev libxv-dev libxvmc-dev libxxf86vm-dev libxcb-render0-dev libxcb-render-util0-dev libxcb-xkb-dev libxcb-icccm4-dev libxcb-image0-dev libxcb-keysyms1-dev libxcb-randr0-dev libxcb-shape0-dev libxcb-sync-dev libxcb-xfixes0-dev libxcb-xinerama0-dev libxcb-dri3-dev uuid-dev libxcb-cursor-dev
RUN apt-get install -y libxcb-util-dev libxcb-util0-dev
RUN apt-get install -y pkg-config
RUN apt-get install -y libx11-xcb-dev libfontenc-dev libice-dev libsm-dev libxaw7-dev libxmu-dev libxpm-dev libxt-dev

RUN mkdir build
WORKDIR /build
