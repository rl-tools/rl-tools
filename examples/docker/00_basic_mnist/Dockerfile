FROM ubuntu:20.04 as BUILD
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install numpy==1.22.4 datasets==2.11.0 h5py==3.7.0
RUN pip3 install Pillow==9.4.0
COPY fetch_and_convert_mnist.py /fetch_and_convert_mnist.py
RUN python3 /fetch_and_convert_mnist.py


FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y cmake build-essential
RUN apt-get install -y libhdf5-dev libboost-all-dev
RUN apt-get install -y protobuf-compiler libprotobuf-dev
RUN mkdir build
COPY --from=BUILD /mnist.hdf5 /build/mnist.hdf5
WORKDIR /build
