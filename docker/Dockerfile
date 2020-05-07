FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ENV  DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y tzdata
ENV TZ=Asia/Tokyo

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake python3-opencv libssl-dev wget \
    libjpeg8-dev liblapack-dev libblas-dev gfortran \
    python3 python3-pip openmpi-bin python3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    git && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install setuptools wheel && pip3 install -U pip
RUN pip3 install tqdm torch pycuda

RUN pip3 install -U pip && \
    pip3 uninstall -y numpy && pip3 install numpy tensorflow-cpu

# install cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.17.0/cmake-3.17.0.tar.gz && \
    tar zxvf cmake-3.17.0.tar.gz && cd cmake-3.17.0 && cmake . && make -j && make install && \
    apt purge -y cmake && cd .. && rm -rf cmake-3.17.0

# install TensorRT
ARG TENSORRT_VERSION=7.0.0.11
COPY nv-tensorrt-repo-ubuntu1804-cuda10.2-trt${TENSORRT_VERSION}-ga-20191216_1-1_amd64.deb .

RUN apt install ./nv-tensorrt-repo-ubuntu1804-cuda10.2-trt${TENSORRT_VERSION}-ga-20191216_1-1_amd64.deb&& \
    apt-get update && \
    apt-get install -y tensorrt && \
    apt-get install -y python-libnvinfer-dev && \
    apt-get install -y python3-libnvinfer-dev && \
    apt-get install -y uff-converter-tf
    # && \
    #rm nv-tensorrt-repo-ubuntu1804-cuda10.1-trt${TENSORRT_VERSION}-ga-20191216_1-1_amd64.deb

RUN git clone https://github.com/kuroko1t/nne.git && cd nne && python3 -m pip install -e .
WORKDIR /nne