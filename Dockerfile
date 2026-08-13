FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# 1. System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# 3. Python dependencies
RUN pip install --no-cache-dir \
    opencv-python-headless \
    boto3 \
    python-dotenv \
    requests \
    runpod \
    gdown \
    scipy \
    pyyaml \
    "pymongo[srv]" \
    google-auth

# 4. Install BasicSR & RealESRGAN without dependencies
RUN pip install --no-cache-dir --no-deps basicsr
RUN pip install --no-cache-dir --no-deps realesrgan

# PATCH: Bulletproof BasicSR TorchVision 0.16+ functional_tensor fix
RUN python -c "import basicsr, os; p = os.path.join(os.path.dirname(basicsr.__file__), 'data', 'degradations.py'); open(p, 'w').write(open(p).read().replace('functional_tensor', 'functional'))"

# 5. Lock NumPy to 1.26.4
RUN pip uninstall -y numpy && pip install --force-reinstall --no-cache-dir "numpy==1.26.4"

# 6. Pre-download RealESRGAN model weights
RUN mkdir -p /app/weights && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -O /app/weights/RealESRGAN_x4plus.pth

# 7. Copy application script
COPY async_handler.py /app/async_handler.py

CMD ["python", "-u", "/app/async_handler.py"]