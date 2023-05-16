FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y cmake build-essential libglfw3-dev libhdf5-dev libboost-all-dev
RUN mkdir build
WORKDIR /build
