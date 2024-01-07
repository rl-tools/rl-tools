FROM ubuntu:22.04 as builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git wget nlohmann-json3-dev libboost-all-dev
RUN apt-get update && apt-get install -y build-essential cmake

RUN git clone https://github.com/rl-tools/rl-tools rl_tools && echo 7
WORKDIR /rl_tools
RUN git submodule update --init -- external/json
WORKDIR /rl_tools/include/rl_tools/ui_server
RUN ./download_dependencies.sh
RUN mkdir /build
WORKDIR /build
RUN cmake ../rl_tools -DCMAKE_BUILD_TYPE=Release -DRL_TOOLS_ENABLE_JSON=ON -DRL_TOOLS_ENABLE_BOOST_BEAST=ON
RUN cmake --build . --target ui_server

FROM ubuntu:22.04
LABEL authors="Jonas Eschmann"

COPY --from=builder /build/src/ui_server/ui_server /usr/local/bin/ui_server
COPY --from=builder /rl_tools/include/rl_tools/ui_server/static/ /include/rl_tools/ui_server/static
ENTRYPOINT ["/usr/local/bin/ui_server"]
CMD ["0.0.0.0", "8000", "multirotor", "default"]
