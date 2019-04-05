FROM debian:stable-slim

LABEL "name"="sh"
LABEL "maintainer"="David Beckingsale <david@llnl.gov>"
LABEL "version"="1.0.0"

LABEL "com.github.actions.name"="Better Shell for GitHub Actions"
LABEL "com.github.actions.description"="Runs one or more commands in an Action"
LABEL "com.github.actions.icon"="terminal"
LABEL "com.github.actions.color"="gray-dark"

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        git && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

COPY LICENSE /

COPY entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
