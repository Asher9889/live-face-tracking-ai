FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app
# Do not ask for user input during package installation
ENV DEBIAN_FRONTEND=noninteractive

COPY ./* /app/



