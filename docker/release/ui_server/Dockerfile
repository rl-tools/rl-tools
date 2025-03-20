FROM ubuntu:22.04 as builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git wget nlohmann-json3-dev libboost-all-dev
RUN apt-get update && apt-get install -y build-essential cmake

RUN #git clone https://github.com/rl-tools/rl-tools rl_tools && echo 7
RUN mkdir /rl-tools
COPY .git /rl-tools/.git
WORKDIR /rl-tools
RUN git reset --hard HEAD && git clean -fdx
RUN git submodule update --init -- external/json
RUN git submodule update --init -- external/cli11
WORKDIR /rl-tools
RUN mkdir /build
WORKDIR /build
RUN cmake /rl-tools -DCMAKE_BUILD_TYPE=Release -DRL_TOOLS_ENABLE_TARGETS=ON -DRL_TOOLS_ENABLE_JSON=ON -DRL_TOOLS_ENABLE_BOOST_BEAST=ON -DRL_TOOLS_ENABLE_CLI11=ON
RUN cmake --build . --target ui_server
WORKDIR /rl-tools/static/ui_server/generic
RUN ./download_dependencies.sh

FROM ubuntu:22.04
LABEL authors="Jonas Eschmann"

COPY --from=builder /build/src/ui_server/ui_server /usr/local/bin/ui_server
COPY --from=builder /rl-tools/static/ui_server/ /static/ui_server/
WORKDIR /
CMD ["/usr/local/bin/ui_server", "--ip", "0.0.0.0"]
