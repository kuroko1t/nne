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
COPY TensorRT-${TENSORRT_VERSION}.*.tar.gz .

RUN tar -xvf TensorRT-${TENSORRT_VERSION}.*.tar.gz && \
    cd TensorRT-${TENSORRT_VERSION}/ && \
    cp lib/lib* /usr/lib/x86_64-linux-gnu/ && \
    rm /usr/lib/x86_64-linux-gnu/libnv*.a && \
    cp include/* /usr/include/x86_64-linux-gnu/ && \
    cp bin/* /usr/bin/ && \
    mkdir /usr/share/doc/tensorrt && \
    cp -r doc/* /usr/share/doc/tensorrt/ && \
    mkdir /usr/src/tensorrt && \
    cp -r samples /usr/src/tensorrt/  && \
    pip3 install python/tensorrt-${TENSORRT_VERSION}-cp36-none-linux_x86_64.whl && \
    pip3 install uff/uff-*-py2.py3-none-any.whl && \
    cd ../ && \
    rm -rf TensorRT-${TENSORRT_VERSION}*

RUN git clone https://github.com/kuroko1t/nne.git && cd nne && python3 -m pip install -e .
WORKDIR /nne