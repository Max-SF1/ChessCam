# Base image
FROM harel316/ds7:humble

# Add sudo and other tools
RUN apt-get update && apt-get install -y sudo \
    && rm -rf /var/lib/apt/lists/*


# Set the working directory
WORKDIR /workspace

# Default command
CMD ["bash"]
