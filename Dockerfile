# Use an Ubuntu base image
FROM ubuntu:latest

# Install Python
RUN apt-get update && apt-get install -y python3

# Set the working directory
WORKDIR /app

# Copy your Python script into the container
COPY app.py .

# Set the default command to run the Python script
CMD ["python3", "app.py"]
