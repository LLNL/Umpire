FROM ghcr.io/llnl/radiuss:gcc-13-ubuntu-24.04

RUN sudo apt-get -qq update && sudo apt-get install -y --no-install-recommends abigail-tools perl-base binutils libtool pkg-config elfutils libelf-dev python3-pip

RUN apt-get update && apt-get install -y cmake

RUN git clone https://github.com/lvc/vtable-dumper && cd vtable-dumper \
    && sudo make install

RUN git clone https://github.com/lvc/abi-dumper && cd abi-dumper \
    && sudo make install

RUN git clone https://github.com/lvc/abi-compliance-checker && cd abi-compliance-checker \
    && sudo make install

COPY entrypoint.sh /entrypoint.sh
USER root

ENTRYPOINT ["/entrypoint.sh"]
