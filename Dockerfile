FROM ubuntu:20.04

# common dependencies
RUN apt-get update && apt-get install -y  \
    python3-pip \
    git

RUN apt-get clean autoclean && \
    apt-get autoremove -y

RUN pip install matplotlib && \
    pip install tensorflow

WORKDIR /opt
RUN git clone https://github.com/gabrielecastellano/Scission

WORKDIR /opt/Scission

ENTRYPOINT python3 ./scission_benchmark.py $0 $@
