# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Build with `docker build -f docker/aie.dockerfile -t allo-ci:aie .`
#               `docker run -it  --device /dev/accel/accel0:/dev/accel/accel0  --ulimit memlock=-1 allo-ci:aie bash`

FROM chhzh123/allo:latest

# Copy the scripts/ directory from the root of the allo project to /ryzers/scripts/ in the container
COPY scripts/ /ryzers/scripts/

ARG PATCH_FILE="/ryzers/scripts/mlir-aie-patch.diff"

# required by rocm driver
RUN groupadd -f render && usermod -aG render root

WORKDIR /ryzers

# Suppress prompts in scripts
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
	git \
	curl \
	bash \
	wget

RUN apt install -y software-properties-common && \
    add-apt-repository ppa:amd-team/xrt && \
    apt update && \
    apt install -y libxrt2 libxrt-npu2 libxrt-dev libxrt-utils libxrt-utils-npu amdxdna-dkms

RUN eval "$(/root/miniconda/bin/conda shell.bash hook)" && \
    conda activate allo && \
    python3 -m pip install mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.0 && \
    python3 -m pip install llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly

RUN git clone https://github.com/Xilinx/mlir-aie.git "/ryzers/mlir-aie" && \
    cd "/ryzers/mlir-aie" && \
    git checkout 07320d6 && \
    patch -p1 < "$PATCH_FILE"
