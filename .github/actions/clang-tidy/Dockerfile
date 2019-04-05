FROM axom/compilers:clang-6

LABEL "name"="sh"
LABEL "maintainer"="David Beckingsale <david@llnl.gov>"
LABEL "version"="1.0.0"

LABEL "com.github.actions.name"="Clang Tidy"
LABEL "com.github.actions.description"="Run clang-tidy on a CMake project"
LABEL "com.github.actions.icon"="trash-2"
LABEL "com.github.actions.color"="green"

COPY entrypoint.sh /entrypoint.sh
COPY run-clang-tidy.py /run-clang-tidy.py

USER root

ENTRYPOINT ["/entrypoint.sh"]
