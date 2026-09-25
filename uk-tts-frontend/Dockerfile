# CPU image. The verbalizer checkpoint is mounted, not baked in, so a
# checkpoint change is a redeploy of configuration rather than of the image.
FROM python:3.11-slim

WORKDIR /app

# 2.6 is the floor, not a preference: the verbalizer checkpoint ships as
# pytorch_model.bin rather than safetensors, and transformers refuses to
# torch.load one on anything older (CVE-2025-32434).
RUN pip install --no-cache-dir \
      torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[service,ui]"

ENV UKTTS_VERBALIZER_CHECKPOINT=/models/verbalizer \
    UKTTS_VERBALIZER_DEVICE=cpu \
    UKTTS_STRESS_URL=http://api:8080 \
    HF_HOME=/tmp/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8000
USER 65534:65534
CMD ["uvicorn", "--factory", "uktts.service:create_app", "--host", "0.0.0.0", "--port", "8000"]
