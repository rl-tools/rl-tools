FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y wget gnupg
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
| gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
RUN echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list
RUN apt-get update && apt-get install -y intel-oneapi-mkl-devel-2023.1.0
RUN apt-get update && apt-get install -y cmake build-essential git vim tmux libhdf5-dev libboost-all-dev protobuf-compiler libprotobuf-dev python3 python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install tensorboard==2.12.2 six==1.16.0
RUN mkdir build
WORKDIR /build
