FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# Install Python + system deps
RUN apt-get update && apt-get install -y \
    build-essential \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    curl \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Symlink python
# Make python command point to 3.10
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Copy requirements first
COPY requirement.txt .

RUN pip install -r requirement.txt

# Copy code
COPY . .

CMD ["python", "-m", "app.main"]