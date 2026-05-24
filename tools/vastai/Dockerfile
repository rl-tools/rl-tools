# FROM vastai/base-image:cuda-13.1.0-auto
FROM vastai/base-image:cuda-13.1.1-cudnn-devel-ubuntu24.04-py314-2026-03-26
COPY rl-tools rl-tools/
WORKDIR rl-tools/
RUN apt-get update
RUN apt-get install -y libassimp-dev protobuf-compiler libprotobuf-dev
RUN ls
RUN bash -c "(cd tests && ./download_data.sh)"
RUN bash -c "(cd src/nn_models/port_checkpoint/example && ./setup.sh)"
RUN bash -c "(cd src/nn_models/port_checkpoint/raptor && ./setup.sh)"
RUN apt-get update
RUN apt-get install -y libhdf5-dev
RUN cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DRL_TOOLS_ENABLE_TARGETS=ON \
  -DRL_TOOLS_ENABLE_TESTS=ON \
  -DRL_TOOLS_EXPERIMENTAL=ON \
  -DRL_TOOLS_ENABLE_TAR=ON \
  -DRL_TOOLS_RENDERING_ENABLE_RAYTRACING=ON
RUN cmake --build build/ -j10 --target nn_models_port_checkpoint_raptor
RUN ./build/src/nn_models/port_checkpoint/raptor/nn_models_port_checkpoint_raptor
RUN cmake --build build/ -j10 --target rl_environments_l2f_visual_yaw_cuda
RUN git clone https://github.com/rl-tools/procthor-train-200-glb.git glb
WORKDIR /
