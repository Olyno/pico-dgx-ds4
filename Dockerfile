# DS4 (DeepSeek-V4.1-Flash) engine image — built and run by start.sh / stop.sh.
# One image holds the engine (dsgb10_rank, sm_121a / GB10) and the Rust
# front-end (dsfront). It runs with host networking + /dev/infiniband so the
# ConnectX-7 ring keeps working; service.sh inside it drives the four ranks
# exactly as it does on bare metal (it copies the binary to the workers over
# ssh, so those boxes need the same CUDA driver and rdma-core).
# build: docker build -t pico-dgx-ds4:latest -f Dockerfile .
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/root \
    CMAKE_PREFIX_PATH=/usr/local/cuda-13.0 \
    PATH=/usr/local/cuda-13.0/bin:/usr/local/cargo/bin:$PATH \
    CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake g++ python3 python3-pip libibverbs-dev rdma-core libnuma-dev \
    curl ca-certificates openssh-client rsync iproute2 kmod \
  && rm -rf /var/lib/apt/lists/* \
  && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
       | sh -s -- -y --profile minimal --default-toolchain 1.97 \
  && cargo --version \
  && python3 -m pip install --break-system-packages --no-cache-dir huggingface_hub hf_xet \
  && (command -v hf || command -v huggingface-cli)

WORKDIR /work

# The engine's CMake configure copies ../fleet/homelab.conf beside the
# binaries, so the fleet directory must be here before the build.
COPY ds41f-4xgx10-switchless/fleet/ ds41f-4xgx10-switchless/fleet/
COPY ds41f-4xgx10-switchless/engine/ ds41f-4xgx10-switchless/engine/
COPY ds41f-4xgx10-switchless/frontend/ ds41f-4xgx10-switchless/frontend/
RUN cmake -S ds41f-4xgx10-switchless/engine -B ds41f-4xgx10-switchless/engine/build \
 && cmake --build ds41f-4xgx10-switchless/engine/build -j4 \
 && (cd ds41f-4xgx10-switchless/frontend && cargo build --release -j4)

# service.sh start/stop/status is the entry point (see start.sh).
WORKDIR /work/ds41f-4xgx10-switchless
CMD ["engine/server/service.sh", "status"]
