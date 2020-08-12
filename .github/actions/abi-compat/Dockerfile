FROM axom/compilers:gcc-8

RUN sudo apt-get -qq update && sudo apt-get install -y --no-install-recommends abigail-tools perl-base binutils libtool pkg-config elfutils libelf-dev

RUN git clone https://github.com/lvc/vtable-dumper && cd vtable-dumper \
    && sudo make install

RUN git clone https://github.com/lvc/abi-dumper && cd abi-dumper \
    && sudo make install

RUN git clone https://github.com/lvc/abi-compliance-checker && cd abi-compliance-checker \
    && sudo make install

COPY entrypoint.sh /entrypoint.sh
USER root

ENTRYPOINT ["/entrypoint.sh"]
