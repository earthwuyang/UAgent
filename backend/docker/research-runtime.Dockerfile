# Minimal OpenHands Runtime for UAgent
FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    wget \
    vim \
    nano \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python3 as default python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Create OpenHands directory structure
RUN mkdir -p /workspace /openhands/code

# Install OpenHands and dependencies
RUN pip3 install openhands

# Create a simple entry point script
RUN echo '#!/bin/bash\n\
echo "OpenHands runtime starting..."\n\
cd /workspace\n\
exec python3 -m openhands.runtime.action_execution_server "$@"\n\
' > /openhands/entrypoint.sh && chmod +x /openhands/entrypoint.sh

# Set working directory
WORKDIR /workspace

# Entry point
ENTRYPOINT ["/openhands/entrypoint.sh"]