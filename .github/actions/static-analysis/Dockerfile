FROM axom/compilers:clang-6

LABEL "name"="sh"
LABEL "maintainer"="David Beckingsale <david@llnl.gov>"
LABEL "version"="1.0.0"

LABEL "com.github.actions.name"="Clang Static Analysis for CMake"
LABEL "com.github.actions.description"="Run scan-build on a CMake project"
LABEL "com.github.actions.icon"="aperture"
LABEL "com.github.actions.color"="orange"

COPY --chown=axom:axom entrypoint.sh /home/axom/entrypoint.sh

USER axom
WORKDIR /home/axom

ENTRYPOINT ["/home/axom/entrypoint.sh"]
