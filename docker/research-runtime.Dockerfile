# Generic Research Runtime Docker Image
# OpenHands runtime extended with comprehensive build tools for compiling ANY software from source

FROM ghcr.io/all-hands-ai/runtime:0.57-nikolaik

# Install comprehensive build tools and development libraries
# Split into multiple RUN commands for better error handling
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    autoconf \
    automake \
    libtool \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    zip \
    unzip \
    bzip2 \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    libssl-dev \
    zlib1g-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    libbz2-dev \
    liblzma-dev \
    libncurses5-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-pip \
    vim \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Install common Python packages for scientific computing
# Try multiple Python locations (poetry venv or system python3)
RUN if [ -f /openhands/poetry/openhands-ai-5O4_aCHf-py3.12/bin/pip ]; then \
        /openhands/poetry/openhands-ai-5O4_aCHf-py3.12/bin/pip install --no-cache-dir \
            numpy scipy pandas matplotlib scikit-learn requests; \
    else \
        pip3 install --no-cache-dir \
            numpy scipy pandas matplotlib scikit-learn requests; \
    fi

# Create workspace with proper permissions
RUN mkdir -p /workspace && chmod 777 /workspace

# Add build information
RUN echo "Generic Research Runtime - Built: $(date)" > /BUILD_INFO && \
    echo "Includes: gcc, g++, cmake, autotools, git, and common dev libraries" >> /BUILD_INFO && \
    echo "" >> /BUILD_INFO && \
    echo "Build Tools: gcc, g++, cmake, autoconf, automake, libtool, make" >> /BUILD_INFO && \
    echo "Version Control: git, wget, curl" >> /BUILD_INFO && \
    echo "Development Libraries: libssl-dev, zlib1g-dev, libreadline-dev, libffi-dev, libsqlite3-dev" >> /BUILD_INFO && \
    echo "Python: numpy, scipy, pandas, matplotlib, scikit-learn, requests" >> /BUILD_INFO && \
    echo "" >> /BUILD_INFO && \
    echo "This is a GENERIC image. Domain-specific software should be built from source." >> /BUILD_INFO

WORKDIR /workspace

# Verify installations
RUN gcc --version && cmake --version && git --version && python3 --version

# No domain-specific software pre-installed
# Users should clone and build from source as needed