FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 wget curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    opencv-python-headless boto3 python-dotenv requests runpod gdown scipy pyyaml "pymongo[srv]" google-auth

RUN pip install --no-cache-dir --no-deps basicsr
RUN pip install --no-cache-dir --no-deps realesrgan

# Patch BasicSR PyTorch 2.x compatibility bug
RUN python -c "import basicsr, os; p = os.path.join(os.path.dirname(basicsr.__file__), 'data', 'degradations.py'); open(p, 'w').write(open(p).read().replace('functional_tensor', 'functional'))"

RUN pip uninstall -y numpy && pip install --force-reinstall --no-cache-dir "numpy==1.26.4"

RUN mkdir -p /app/weights && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -O /app/weights/RealESRGAN_x4plus.pth

COPY async_handler.py /app/async_handler.py

CMD ["python", "-u", "/app/async_handler.py"]