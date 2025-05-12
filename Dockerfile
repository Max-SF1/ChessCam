# Base image
FROM harel316/ds7:humble

# Install sudo and other tools
RUN apt-get update && apt-get install -y sudo \
    && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Set working directory
WORKDIR /workspace

# Default command
CMD ["bash"]
