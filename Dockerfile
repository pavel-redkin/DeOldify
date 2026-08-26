# =============================================================================
# DeOldify Dockerfile - Python 3.14 + PyTorch 2.5+ (CUDA 12.4)
# =============================================================================
# Build:
#   docker build -t deoldify:latest .
#   docker build --build-arg CUDA_VERSION=12.4.0 -t deoldify:latest .
#
# Run (GPU):
#   docker run --gpus all -p 8888:8888 -v ./models:/app/models deoldify:latest
#
# Run (CPU only):
#   docker run --build-arg INSTALL_CUDA=false -p 8888:8888 deoldify:latest
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Base with system dependencies
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04 AS base

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.14

# System dependencies
# Add deadsnakes PPA for Python 3.14
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-dev \
    curl \
    wget \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgtk2.0-0 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.14 as default python/python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.14 \
    && python3.14 -m pip install --no-cache-dir --upgrade pip setuptools wheel packaging

# ---------------------------------------------------------------------------
# Stage 2: Python dependencies
# ---------------------------------------------------------------------------
FROM base AS deps

WORKDIR /app

# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install PyTorch with CUDA 12.4 support, then remaining dependencies
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    torch>=2.5.0 \
    torchvision>=0.20.0 \
    torchaudio>=2.5.0 \
    && pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 3: Final image
# ---------------------------------------------------------------------------
FROM base AS runtime

WORKDIR /app

# Copy installed Python packages from deps stage
# Note: ARG values are not available in COPY --from, so hardcode python3.14
COPY --from=deps /usr/local/lib/python3.14/dist-packages /usr/local/lib/python3.14/dist-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy project code
COPY . .

# Create models directory (for pretrained weights)
RUN mkdir -p /app/models

# Create a non-root user
RUN groupadd -r deoldify && useradd -r -g deoldify -d /app -s /bin/bash deoldify \
    && chown -R deoldify:deoldify /app
USER deoldify

# Expose JupyterLab port
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8888')" || exit 1

# Default command: start JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
