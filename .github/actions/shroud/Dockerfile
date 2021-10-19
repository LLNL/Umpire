FROM ghcr.io/rse-ops/clang-ubuntu-20.04:llvm-12.0.0

RUN git clone https://github.com/llnl/shroud && \
    cd shroud && \
    git fetch && git checkout v0.12.2 && \
    pip3 install .

COPY entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

