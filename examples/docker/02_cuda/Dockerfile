FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y wget gnupg
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
| gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
RUN echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list
RUN apt-get update
RUN apt-get install -y intel-oneapi-mkl-devel
RUN apt-get install -y cmake build-essential git vim tmux
RUN apt-get install -y libhdf5-dev libboost-all-dev
RUN apt-get install -y protobuf-compiler libprotobuf-dev
RUN apt-get install -y python3 python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install tensorboard
RUN mkdir build
WORKDIR /build
