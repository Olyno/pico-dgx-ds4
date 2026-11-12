# The image encoder's runtime: the checkpoint's own inference/vision.py imports
# torch, and the processor needs PIL; the engine image has neither. start.sh
# builds this on first use (VISION=1, the default) and runs vision_encoder.py
# from it against the bundle in $VISION_DIR.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
RUN python3 -m pip install --no-cache-dir pillow safetensors
WORKDIR /app
COPY . /app
ENTRYPOINT []
CMD ["python3", "/app/vision_encoder.py", "--host", "127.0.0.1", "--port", "18950"]
